from __future__ import annotations
import json
import subprocess
import sys
import argparse
from typing import Dict, Any, List

# Run three modes of src.train and collect printed JSON-like summary
# Modes:
#  - none: no aux (load_balance_coef=0, boids_off)
#  - normal: load_balance on, no boids
#  - boids: load_balance on, boids on


def run_once(args: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    lines = proc.stdout.splitlines()
    # find last JSON-ish line printed by train.py (dict via print)
    parsed: Dict[str, Any] | None = None
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                parsed = json.loads(ln.replace("'", '"'))
                break
            except Exception:
                continue
    if parsed is None:
        raise RuntimeError("Could not parse metrics from run output.\nLast 20 lines:\n" + "\n".join(lines[-20:]))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "full"], default="fast")
    args = ap.parse_args()

    if args.preset == "full":
        model_size = "small"; num_experts = "8"; seq_len = "512"; micro_batch = "1"; accum_steps = "2"
    else:
        model_size = "tiny"; num_experts = "4"; seq_len = "128"; micro_batch = "1"; accum_steps = "1"

    base = [sys.executable, "-m", "src.train",
            "--task", "squad", "--dataset_size", "small",
            "--model_size", model_size,
            "--num_experts", num_experts, "--top_k", "1",
            "--seq_len", seq_len, "--train_epochs", "1",
            "--micro_batch", micro_batch, "--accum_steps", accum_steps,
            "--backend", "tiny",
            "--seed", "42",
    ]

    # none: no aux losses
    none_args = base + ["--load_balance_coef", "0.0", "--boids_on", "false"]
    # normal: load balance only
    normal_args = base + ["--load_balance_coef", "0.01", "--boids_on", "false"]
    # boids: load balance + boids
    boids_args = base + ["--load_balance_coef", "0.01", "--boids_on", "true"]

    print("[compare] running: none")
    m_none = run_once(none_args)
    print("[compare] running: normal")
    m_norm = run_once(normal_args)
    print("[compare] running: boids")
    m_boids = run_once(boids_args)

    out = {
        "none": m_none,
        "normal": m_norm,
        "boids": m_boids,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
