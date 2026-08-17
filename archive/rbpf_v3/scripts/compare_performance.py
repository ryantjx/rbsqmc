"""Compare timing summaries emitted by the two standalone v3 runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(cuthbert_path: Path, direct_path: Path) -> dict:
    cuthbert = json.loads(cuthbert_path.read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    shape_keys = ("D", "N", "S", "M", "seed", "device")
    mismatches = {
        key: (cuthbert.get(key), direct.get(key))
        for key in shape_keys
        if cuthbert.get(key) != direct.get(key)
    }
    if mismatches:
        raise ValueError(f"performance runs are not comparable: {mismatches}")
    c_backward = cuthbert["final_e_step"]["backward_seconds"]
    d_backward = direct["final_e_step"]["backward_seconds"]
    return {
        "configuration": {key: cuthbert.get(key) for key in shape_keys},
        "cuthbert": {
            "backward_seconds": c_backward,
            "training_seconds": cuthbert["training_seconds"],
            "total_seconds": cuthbert["total_seconds"],
        },
        "noncuthbert": {
            "backward_seconds": d_backward,
            "training_seconds": direct["training_seconds"],
            "total_seconds": direct["total_seconds"],
        },
        "backward_speedup_direct_over_cuthbert": c_backward / d_backward,
        "total_speedup_direct_over_cuthbert": cuthbert["total_seconds"]
        / direct["total_seconds"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cuthbert", type=Path)
    parser.add_argument("noncuthbert", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.cuthbert, args.noncuthbert)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
