"""The same two edits at convergence, in the same units as the early ones. (2026-08-28)

The lifecycle argument compares the cost of removing the type contrast against a
magnitude-matched control that leaves it intact, early in training and at the
end. Early, that comparison is in TYPE accuracy (early_bit_ablation.py,
matched_acc_bit_ablation.py, chance 0.50). At convergence, the only measurement
of the pair was dc_surgical_ablation.py, which reports OVERALL accuracy.

Those are different quantities, and a figure that puts the early points and the
converged points on one axis is comparing type accuracy against overall accuracy.
This closes the gap: the same two edits, the same type-accuracy measurement, at
each run's final checkpoint.

  base       : untouched
  surgical   : the type-constant contrast removed, per-row deviations kept
  noise_only : balanced +/- offsets of the same magnitude along t_hat, which
               leaves the contrast exactly intact

Population is the 25 generalized dense-checkpoint runs, so the converged point
belongs to the same runs as the trajectory that leads to it. That is the
comparison the lifecycle claim needs; the 16-run class-A population mean in
Section 3.3 answers a different question, in different units, and is not a
substitute.

Reading, stated before running:
  - the gap closes to nothing => the contrast is superseded, which is the claim.
  - the gap stays open => the contrast is still load-bearing at convergence, and
    Section 3.3's overall-accuracy result would need reconciling with this one.

Run: python -m instruments.converged_type_ablation
"""
import glob
import os
from collections import defaultdict

import numpy as np

from instruments.dc_component_ablation import truth
from instruments.early_bit_ablation import final_class
from instruments.matched_acc_bit_ablation import ablate
from instruments.population import run_dir
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

SUB = 3000


def main():
    print("  The Section-3.3 edit pair at CONVERGENCE, measured in type accuracy.")
    print("  base / surgical (contrast removed) / noise (same magnitude, contrast "
          "intact). Chance = 0.50.\n")
    cache, rows, agg = {}, [], defaultdict(list)
    for rd in sorted(d for d in glob.glob(run_dir("dense_checkpoint_sweep/d*"))
                     if os.path.isdir(d)):
        n = int(os.path.basename(rd).split("_")[0][1:])
        cls = final_class(rd, n)
        if cls is None:
            continue
        if n not in cache:
            pairs, _, _, _ = all_pairs_typed(n)
            _, tt = truth(pairs, n)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pairs), SUB, replace=False)
            cache[n] = ([pairs[i] for i in idx], tt[idx])
        sp, tt = cache[n]

        ck = os.path.join(rd, "checkpoints", "best_model.pt")
        base, surg, noi = ablate(ck, n, sp, tt)
        agg[cls].append((base, surg, noi))
        rows.append((os.path.basename(rd), n, cls, f"{base:.6f}", f"{surg:.6f}",
                     f"{noi:.6f}"))
        print(f"  measured {os.path.basename(rd)} ({cls})", flush=True)

    save("converged_type_ablation", "run,n,cls,base,surgical,noise_only", rows)

    print(f"\n  {'cls':>3} {'k':>3} | {'base':>6} {'surgical':>9} {'noise':>7} | "
          f"{'surg cost':>10} {'noise cost':>11} {'difference':>11}")
    for cls in ("A", "B"):
        v = np.array(agg[cls])
        if not len(v):
            continue
        cs, cn = v[:, 0] - v[:, 1], v[:, 0] - v[:, 2]
        print(f"  {cls:>3} {len(v):>3} | {v[:,0].mean():>6.3f} {v[:,1].mean():>9.3f} "
              f"{v[:,2].mean():>7.3f} | {cs.mean():>10.3f} {cn.mean():>11.3f} "
              f"{(cs - cn).mean():>11.3f}")
    va = np.array(agg["A"])
    d = (va[:, 0] - va[:, 1]) - (va[:, 0] - va[:, 2])
    boot = np.array([np.random.default_rng(i).choice(d, len(d)).mean()
                     for i in range(10000)])
    print(f"\n  class A difference: {d.mean():+.4f}, bootstrap 95% CI "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
    print(f"  runs where removing the contrast costs more: {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
