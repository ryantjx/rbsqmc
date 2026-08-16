import json
import numpy as np

teams = json.load(open("rbpf_v3/data/worldcup2026.json"))
team_id_to_name = {i: name for i, name in enumerate(sorted(teams))}
print("n teams:", len(team_id_to_name))

d = np.load("rbpf_v3/outputs/smoothing/optimal_filter/filter_states.npz")
means = d["means"]  # (T,N,M,2)
logw = d["log_weights"]

w = np.exp(logw[-1] - logw[-1].max())
w = w / w.sum()
final = means[-1]  # (N,M,2)
wm = np.einsum("n,nm...->m...", w, final)  # (M,2) attack,defense

print("\n=== FINAL FILTERED STRENGTHS (weighted mean over particles) ===")
print("rank | team | attack | defense | total")
order = np.argsort(-(wm[:, 0] + wm[:, 1]))
for r, i in enumerate(order[:15]):
    print(f"{r+1:3d} | {team_id_to_name[i]:22s} | {wm[i,0]:+.3f} | {wm[i,1]:+.3f} | {wm[i,0]+wm[i,1]:+.3f}")

print("\n=== TOP ATTACK ===")
for r, i in enumerate(np.argsort(-wm[:, 0])[:10]):
    print(f"{r+1:2d} {team_id_to_name[i]:22s} att={wm[i,0]:+.3f}")

print("\n=== TOP DEFENSE (most negative = best) ===")
for r, i in enumerate(np.argsort(wm[:, 1])[:10]):
    print(f"{r+1:2d} {team_id_to_name[i]:22s} def={wm[i,1]:+.3f}")

print("\n=== BOTTOM TOTAL (weakest) ===")
for r, i in enumerate(order[-8:]):
    print(f"{r+1:2d} {team_id_to_name[i]:22s} total={wm[i,0]+wm[i,1]:+.3f}")
