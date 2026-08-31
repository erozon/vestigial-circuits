"""Are the doubly trained seeds really one trajectory recorded twice? (2026-08-28)

Sixteen (order, seed) pairs were trained in BOTH the prime/composite sweep
(checkpoints every 20,000 epochs) and the dense-checkpoint sweep (every 1,000):
D_47 seeds 0-4, D_61 seeds 0-7, D_63 seeds 0-2. The provenance paragraph of the
paper (app:runs) claims each pair is a single deterministic trajectory, so the
two directories must hold bit-identical weights at every checkpoint epoch they
share. This instrument checks exactly that, for every pair and every shared
epoch, by comparing all tensors of the two model_state_dicts.

If any tensor ever differs, the paper's "counted once" accounting is wrong and
the population must treat the copies as distinct runs. Report the failure; do
not paper over it.

Run: python -m instruments.verify_twin_runs
"""
import glob
import os
import re

import torch

from instruments.population import run_dir
from instruments.results_io import save

SWEEPS = ("prime_composite_sweep", "dense_checkpoint_sweep")


def ckpt_epochs(d):
    out = {}
    for p in glob.glob(os.path.join(d, "checkpoints", "checkpoint_epoch_*.pt")):
        out[int(re.search(r"_(\d+)\.pt$", p).group(1))] = p
    return out


def weights(path):
    return torch.load(path, map_location="cpu",
                      weights_only=False)["model_state_dict"]


def main():
    a_dirs = {os.path.basename(d): d
              for d in glob.glob(run_dir(f"{SWEEPS[0]}/d*_seed*"))
              if os.path.isdir(d)}
    b_dirs = {os.path.basename(d): d
              for d in glob.glob(run_dir(f"{SWEEPS[1]}/d*_seed*"))
              if os.path.isdir(d)}
    twins = sorted(set(a_dirs) & set(b_dirs))
    print(f"  {len(twins)} seeds trained in both sweeps")
    print(f"  {'run':<12} {'shared epochs':>26} {'identical':>10} {'max|diff|':>10}")

    rows, bad = [], 0
    for name in twins:
        n = int(name.split("_")[0][1:])
        ea, eb = ckpt_epochs(a_dirs[name]), ckpt_epochs(b_dirs[name])
        shared = sorted(set(ea) & set(eb))
        worst = 0.0
        for ep in shared:
            wa, wb = weights(ea[ep]), weights(eb[ep])
            assert wa.keys() == wb.keys(), f"{name}@{ep}: key mismatch"
            for k in wa:
                d = (wa[k] - wb[k]).abs().max().item()
                worst = max(worst, d)
        ok = worst == 0.0
        bad += not ok
        span = f"{shared[0]//1000}k-{shared[-1]//1000}k x{len(shared)}"
        print(f"  {name:<12} {span:>26} {str(ok):>10} {worst:>10.2e}")
        rows.append((name, n, len(shared), shared[0], shared[-1],
                     int(ok), worst))

    print(f"\n  {'ALL IDENTICAL' if bad == 0 else f'{bad} PAIRS DIFFER'}")
    save("verify_twin_runs",
         "run,n,shared_epochs,first_epoch,last_epoch,identical,max_abs_diff",
         rows)


if __name__ == "__main__":
    main()
