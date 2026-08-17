import os
os.environ["RBSQMC_CONFIG"] = "rbpf/scripts/config/smoothing_gpu_config.json"

from rbpf.src.smoothing import _load_run_config
cfg = _load_run_config()
print("=== smoothing.py (adam) ===")
print("start_date:", cfg.get("start_date"))
print("n_particles:", cfg.get("n_particles"))
print("n_epochs:", cfg.get("n_epochs"))
print("learning_rate:", cfg.get("learning_rate"))

os.environ["RBSQMC_CONFIG"] = "rbpf/scripts/config/smoothing_bfgs_gpu_config.json"
from rbpf.src.smoothing_bfgs import _load_run_config as _load_bfgs
cfg2 = _load_bfgs()
print("\n=== smoothing_bfgs.py (bfgs) ===")
print("start_date:", cfg2.get("start_date"))
print("n_particles:", cfg2.get("n_particles"))
print("n_epochs:", cfg2.get("n_epochs"))

# Test fallback (no env var)
del os.environ["RBSQMC_CONFIG"]
from rbpf.src.smoothing import _load_run_config as _load_default
cfg3 = _load_default()
print("\n=== fallback (no RBSQMC_CONFIG) ===")
print("start_date:", cfg3.get("start_date", "2000-01-01 (default)"))
print("n_particles:", cfg3.get("n_particles", 1000))