import json, numpy as np

with open("rbpf/outputs/smoothing/optimized_params.json") as f:
    p = json.load(f)

print("=== Optimized Parameters ===")
print(f"kappa: {p['kappa']}")
print(f"alpha: {p['alpha']}")
print(f"beta: {p['beta']}")

g = np.array(p["gamma_0"])
print(f"gamma_0 shape: {g.shape}")
print(f"gamma_0 diag[:5]: {np.diag(g)[:5]}")
print(f"gamma_0 min eig: {np.linalg.eigvalsh(g)[0]:.6f}")
print(f"gamma_0 max eig: {np.linalg.eigvalsh(g)[-1]:.6f}")

B = np.array(p["B"])
print(f"B: {B.tolist()}")
print(f"B eig: {np.linalg.eigvalsh(B).tolist()}")

# Check for degeneracy
print(f"\n=== Degeneracy Check ===")
print(f"kappa reasonable (0.001-1.0): {0.001 < p['kappa'] < 1.0}")
print(f"alpha reasonable (-5 to 5): {-5 < p['alpha'] < 5}")
print(f"beta reasonable (-10 to 0): {-10 < p['beta'] < 0}")
print(f"gamma_0 diag not collapsed (all > 0.01): {np.all(np.diag(g) > 0.01)}")
print(f"B not collapsed (all eig > 0.01): {np.all(np.linalg.eigvalsh(B) > 0.01)}")