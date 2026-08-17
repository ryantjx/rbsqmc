import json, numpy as np

with open("rbpf/outputs/smoothing/optimized_params.json") as f:
    p = json.load(f)

g0 = np.array(p["gamma_0"])
gT = np.array(p.get("gamma_T", p["gamma_0"]))  # no gamma_T saved, use gamma_0

# Correlation from gamma_0 (the prior that EM optimizes)
std0 = np.sqrt(np.diag(g0))
std0_safe = np.where(std0 > 1e-10, std0, 1.0)
corr0 = g0 / np.outer(std0_safe, std0_safe)
corr0 = np.clip(corr0, -1, 1)

print("=== gamma_0 (prior) correlation ===")
print(f"diag (variances): {np.diag(g0)[:5]}")
print(f"off-diag mean: {np.mean(corr0[np.triu_indices(48, k=1)]):.4f}")
print(f"off-diag max: {np.max(corr0[np.triu_indices(48, k=1)]):.4f}")
print(f"off-diag min: {np.min(corr0[np.triu_indices(48, k=1)]):.4f}")
print(f"n off-diag > 0.01: {np.sum(np.abs(corr0[np.triu_indices(48, k=1)]) > 0.01)}")

# Check if the posterior gamma would have correlation
# (we don't have gamma_T saved, but we can check the structure)
print(f"\n=== gamma_0 structure ===")
print(f"min eig: {np.linalg.eigvalsh(g0)[0]:.6f}")
print(f"max eig: {np.linalg.eigvalsh(g0)[-1]:.6f}")
print(f"condition number: {np.linalg.eigvalsh(g0)[-1] / np.linalg.eigvalsh(g0)[0]:.2f}")
print(f"trace: {np.trace(g0):.4f}")
print(f"mean off-diag (raw): {np.mean(g0[np.triu_indices(48, k=1)]):.6f}")