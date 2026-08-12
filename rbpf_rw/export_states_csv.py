"""Convert the saved final filter states .npy to a CSV with team_name, attack, defence.

Reads:
  - final_filter_states.npy  (M, 2)  [attack, defence]
  - team_names.npy           (M,)    team names

Writes:
  - final_filter_states.csv  with columns: team_name, attack, defence
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "outputs_gpu", "trained")

final_states = np.load(os.path.join(OUT_DIR, "final_filter_states.npy"))  # (M, 2)
team_names = np.load(os.path.join(OUT_DIR, "team_names.npy"), allow_pickle=True)  # (M,)

assert final_states.shape[0] == len(team_names), "state/team count mismatch"

attack = final_states[:, 0]
defence = final_states[:, 1]

csv_path = os.path.join(OUT_DIR, "final_filter_states.csv")
with open(csv_path, "w") as f:
    f.write("team_name,attack,defence\n")
    for name, a, d in zip(team_names, attack, defence):
        f.write(f"{name},{a:.6f},{d:.6f}\n")

print(f"Wrote {csv_path} with {len(team_names)} rows")
