"""Adversarial checks on the redundant-bit finding (2026-08-05).

Written to attack our own result, not to confirm it. Four checks:

CHECK 1 -- NORMALIZATION ARTIFACT.
  The claim "class A's frequency-based type margin equals class B's" (2.16 vs 2.15)
  normalized each margin by that model's logit standard deviation. But class A's
  logits INCLUDE the bit's contribution, which inflates its denominator and
  deflates its reported frequency margin. The agreement could be manufactured.
  Three normalizations, which fail differently:
    n_logit : sd of the model's own element logits            (the original)
    n_perp  : sd of logits recomputed with t removed from U   (bit excluded)
    n_geom  : ||x|| * mean_c||D_c||, pure geometry, no logits at all
  If the classes stay equal under all three, the result is not a normalizer effect.

CHECK 2 -- IS THE ZERO-PARAMETER READOUT TOO WEAK?
  "Class B has no global type direction" comes from projecting onto t, which is
  fixed in advance. A reviewer will say a FITTED direction would find one. So fit
  a linear probe on the FINAL RESIDUAL (the vector that feeds the output layer --
  the strongest place to look) with a train/test split, and report held-out
  accuracy. If class B sits at chance with a fitted probe, the absence is real.

CHECK 3 -- POOLING ACROSS GROUP ORDERS.
  Class composition differs by order (D_47 is mostly class A, D_63 mostly class B),
  so a pooled comparison could manufacture equality. Report per order.

CHECK 4 -- IS IT ACTUALLY BIMODAL?
  The classes are threshold-defined (auc >= 0.9 vs <= 0.7). Report the sorted
  distributions of auc_t and dc_frac over all 35 runs with the largest gap, so a
  continuum would be visible rather than hidden by the binning.

Run: python -m instruments.adversarial_checks
"""
import os
from collections import defaultdict

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_margin_decomposition import resid_and_logits
from instruments.type_xor_probe import all_pairs_typed, lin_probe, split
from instruments.unembed_type_direction_readout import auc


def main():
    print("  ADVERSARIAL CHECKS on the redundant-bit finding.\n")
    print(f"  {'run':24} {'cls':>3} {'n':>3} | {'perp/logit':>10} {'perp/perp':>10} "
          f"{'perp/geom':>10} | {'fitprobe':>9} {'auc_t':>7} {'dc_frac':>8}")
    rows = []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)
        m, tok = build(p, n)
        x, logits = resid_and_logits(m, tok, pairs, n)

        U = m.fc_out.weight.detach().numpy()
        D = U[:n] - U[n:2 * n]
        t = D.mean(0)
        th = t / (np.linalg.norm(t) + 1e-12)
        dc_frac = float(t @ t) / float((D * D).sum(1).mean())

        sign = np.where(tt == 0, 1.0, -1.0)
        Dc = D[ti]
        margin = sign * (x * Dc).sum(1)
        dc = sign * (x @ th) * (Dc @ th)
        perp = margin - dc

        # normalizer 1: the model's own logit spread (original)
        n1 = logits.std(1) + 1e-12
        # normalizer 2: logits with the bit's direction removed from U entirely
        U_perp = U - np.outer(U @ th, th)
        n2 = (x @ U_perp.T).std(1) + 1e-12
        # normalizer 3: pure geometry, no logits
        n3 = np.linalg.norm(x, axis=1) * np.linalg.norm(Dc, axis=1) + 1e-12

        # fitted probe on the FINAL RESIDUAL for output type
        tr, te = split(len(pairs))
        fit = lin_probe(torch.tensor(x, dtype=torch.float32), tt, tr, te)

        score = x @ th
        a = auc(score[tt == 0], score[tt == 1])
        a = max(a, 1 - a)

        print(f"  {os.path.basename(path):24} {cls:>3} {n:>3} | "
              f"{(perp / n1).mean():>10.2f} {(perp / n2).mean():>10.2f} "
              f"{(perp / n3).mean():>10.3f} | {fit:>8.0f}% {a:>7.3f} {dc_frac:>8.3f}")
        rows.append((cls, n, (perp / n1).mean(), (perp / n2).mean(),
                     (perp / n3).mean(), fit, a, dc_frac,
                     os.path.basename(path)))

    print()
    # run name is the LAST field of each row so the positional indices the checks
    # below use (r[2]..r[7]) keep their meaning.
    save("adversarial_checks",
         "run,n,cls,perp_logit,perp_perp,perp_geom,fit_probe,auc_t,dc_frac",
         [(r[8], r[1], r[0]) + tuple(f"{v:.6f}" for v in r[2:8]) for r in rows])

    print("\n  CHECK 1 + 3 -- frequency margin by class, three normalizations, "
          "pooled and per order")
    print(f"  {'group':>8} {'cls':>3} {'k':>3} {'perp/logit':>11} {'perp/perp':>10} "
          f"{'perp/geom':>10}")
    for key in ["pooled"] + sorted({r[1] for r in rows}):
        for cls in ("A", "B"):
            sel = [r for r in rows if r[0] == cls and (key == "pooled" or r[1] == key)]
            if not sel:
                continue
            a = np.array([[r[2], r[3], r[4]] for r in sel])
            print(f"  {str(key):>8} {cls:>3} {len(sel):>3} {a[:,0].mean():>11.2f} "
                  f"{a[:,1].mean():>10.2f} {a[:,2].mean():>10.3f}")

    print("\n  CHECK 2 -- fitted linear probe on the final residual (chance = 50%)")
    for cls in ("A", "B", "I"):
        v = np.array([r[5] for r in rows if r[0] == cls])
        if len(v):
            print(f"    class {cls} (n={len(v)}): {v.mean():.1f}% "
                  f"[{v.min():.0f}, {v.max():.0f}]")

    print("\n  CHECK 4 -- sorted distributions, largest gap marked")
    for name, i in (("auc_t", 6), ("dc_frac", 7)):
        v = np.sort(np.array([r[i] for r in rows]))
        gaps = np.diff(v)
        j = int(gaps.argmax())
        print(f"    {name}: " + " ".join(f"{x:.3f}" for x in v[:j + 1]) +
              f"  <<GAP {gaps[j]:.3f}>>  " + " ".join(f"{x:.3f}" for x in v[j + 1:]))


if __name__ == "__main__":
    main()
