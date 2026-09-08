"""Matched optimization; no early stopping, score substitution, or regularizer."""

import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rbsqmc.src.model.rbsqmc.model_rbsqmc import run_filter_sqmc
from rbsqmc.src.model.ekf import model as ekf
from rbsqmc.src.utils.helpers import default_init_params, encode_EM_params, decode_EM_params


def jsonable(value):
    if hasattr(value, "_asdict"):
        return jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return np.asarray(value).tolist()
    return value


def save_json(path, value):
    """Atomic strict JSON: incomplete writes never masquerade as checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_finite(label, tree):
    for index, leaf in enumerate(jax.tree.leaves(tree)):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError(f"{label}: non-finite values in leaf {index}")


def slice_inputs(inputs, end):
    return jax.tree.map(lambda x: x[:end], inputs)


class Methods:
    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        base = default_init_params(len(data.teams), data.teams)
        self.fixed_mean = base.mean_0
        self.initial = {
            "ekf": ekf.initial_raw(base.gamma_0[0, 0] * base.B, base.alpha, base.beta, base.kappa),
            "sqmc": dict(model=encode_EM_params(base), friendly_scale=ekf.inverse_positive(2.0)),
        }

    def params(self, method, raw):
        if method == "ekf":
            return ekf.constrain(raw)
        return dict(model=decode_EM_params(raw["model"], self.fixed_mean),
                    friendly_scale=ekf.positive(raw["friendly_scale"]))

    def filter(self, method, raw, key, end):
        params = self.params(method, raw)
        if method == "ekf":
            return ekf.run_filter(slice_inputs(self.data.inputs, end), params, len(self.data.teams))
        # Baseline per-match scale comes from config (default 1); friendly
        # fixtures further multiply by the shared learned friendly_scale so
        # both methods use identical observation scaling.
        base = self.cfg.get("match_scale", 1.0)
        scales = jnp.where(self.data.inputs.friendly[:end], params["friendly_scale"], base)[:, None]
        result, augmented = run_filter_sqmc(key, slice_inputs(self.data.sqmc, end),
                                            params["model"], self.cfg["n_particles"],
                                            self.cfg["max_goals"], match_scales=scales)
        return dict(logz=result["log_normalizing_constant"], result=result, augmented=augmented)

    def train(self, method, output):
        cfg, data = self.cfg, self.data
        raw = self.initial[method]
        root = jax.random.PRNGKey(cfg["seed"])
        optimizer = optax.adam(optax.cosine_decay_schedule(cfg["learning_rate"], cfg["n_epochs"]))
        opt_state = optimizer.init(raw)
        # Only training outcomes enter the differentiated objective.
        loss_grad = jax.jit(jax.value_and_grad(
            lambda r, key: -self.filter(method, r, key, data.train_count)["logz"][-1]))
        def score(r, key):
            logz = self.filter(method, r, key, data.train_count + data.test_count)["logz"]
            # One continuous realization makes test logZ conditional on training.
            return jnp.array([logz[data.train_count], logz[-1] - logz[data.train_count]])
        scorer = jax.jit(score)
        started = time.perf_counter()
        gradient_fn = loss_grad.lower(raw, root).compile()
        score_fn = scorer.lower(raw, root).compile()
        compile_sec = time.perf_counter() - started
        print(f"{method}: compiled in {compile_sec:.1f}s", flush=True)
        score_key = jax.random.fold_in(root, 1_000_000)
        baseline = score_fn(raw, score_key)
        require_finite(f"{method} baseline", baseline)
        history = []
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        execution_start = time.perf_counter()
        for epoch in range(cfg["n_epochs"]):
            tick = time.perf_counter()
            replicas = cfg["n_reps"] if method == "sqmc" else 1
            gradient = jax.tree.map(jnp.zeros_like, raw)
            loss = 0.0
            # Sequential accumulation is the same mean gradient as vmap, while
            # keeping peak storage independent of the number of replicas.
            for replica in range(replicas):
                key = jax.random.fold_in(jax.random.fold_in(root, epoch), replica)
                value, grad = gradient_fn(raw, key)
                require_finite(f"{method} epoch {epoch + 1} replica {replica}", (value, grad))
                gradient = jax.tree.map(lambda a, b: a + b / replicas, gradient, grad)
                loss += float(value) / replicas
            norm = float(optax.global_norm(gradient))
            updates, opt_state = optimizer.update(gradient, opt_state, raw)
            raw = optax.apply_updates(raw, updates)
            require_finite(f"{method} updated parameters", (raw, self.params(method, raw)))
            scores = np.asarray(score_fn(raw, score_key))
            require_finite(f"{method} epoch {epoch + 1} scores", scores)
            row = dict(epoch=epoch + 1, train_logz=float(scores[0]), test_logz=float(scores[1]),
                       gradient_norm=norm, pre_update_loss=loss,
                       elapsed_sec=time.perf_counter() - tick)
            history.append(row)
            print(f"{method} epoch {epoch + 1}/{cfg['n_epochs']}: train {scores[0]:.4f}, "
                  f"test {scores[1]:.4f}, |grad| {norm:.4f}, {row['elapsed_sec']:.1f}s", flush=True)
        execution_sec = time.perf_counter() - execution_start
        best = max(history, key=lambda row: row["test_logz"])
        summary = dict(n_epochs_completed=len(history), baseline=baseline,
                       compilation_sec=compile_sec, execution_sec=execution_sec,
                       elapsed_sec=compile_sec + execution_sec,
                       final_train_logz=history[-1]["train_logz"],
                       final_test_logz=history[-1]["test_logz"],
                       best_test_logz=best["test_logz"], best_test_epoch=best["epoch"],
                       checkpoint_policy="final_epoch", learning_rate_schedule="cosine",
                       gradient_replicas=replicas, score_replicas=1)
        save_json(output / "summary.json", summary)
        return raw, history, summary
