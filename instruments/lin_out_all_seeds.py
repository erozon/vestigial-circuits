"""lin_out decodability across ALL grokked seeds -- the bimodality check (2026-07-13).

The robust "localized vs non-localized" fork (output type globally linearly
decodable from the MLP hidden: 100% vs chance) was established on only 6 hand-picked
models, while the "23-seed bimodality" claim lives on the COARSER override/k_loc
metric. This runs the SHARP metric (lin_out) on the SAME grokked population the
override analysis used, to settle: is lin_out actually BIMODAL across seeds, or a
binarized continuum?

For every grokked checkpoint (base acc >= 0.99), on the MLP hidden layer over the
full dataset:
  lin_out : linear decode of OUTPUT type (=op1 XOR op2)  -- the sharp fork metric
  lin_op1 : linear decode of operand-1 type              -- input feature (context)
  lin_op2 : linear decode of operand-2 type
  override: the coarse sign-gate localization metric      -- for cross-check

Bimodal => lin_out clusters near 100 and near 50 with a gap => fork is real on the
real population. Spread across 60/70/80 => continuum => reframe to "degree of
type-linearization".

SCOPE FIX (2026-08-13): this originally ran over a glob of the prime/composite
sweep plus three hand-listed D_59 seeds -- 23 runs, missing every run that lives
only in the dense checkpoint sweep. That was the population before population.py
existed, and it is not the population the paper reports on. It now iterates
population.RUNS, so lin_out is measured on all 35 generalized runs rather than on
whichever subset happened to be globbed. The class labels in population.py were
set partly from the old subset; the printed comparison against `cls` is therefore
worth reading as a check on those labels, not as a foregone agreement.

Run: python -m instruments.lin_out_all_seeds
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import act_resid, build
from instruments.population import RUNS, ckpt
from instruments.prime_composite_analyze import is_prime, override_score
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed, lin_probe, split


def lin_out_metrics(ck, n):
    m, tok = build(ck, n)
    pairs, o1, o2, out = all_pairs_typed(n)
    actFull, _ = act_resid(m, tok, pairs)
    X = actFull
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    tr, te = split(X.shape[0])
    return (lin_probe(X, out, tr, te), lin_probe(X, o1, tr, te), lin_probe(X, o2, tr, te))


def main():
    torch.manual_seed(0)
    rows = []
    for path, n, cls in RUNS:
        ck = ckpt(path)
        if not os.path.exists(ck):
            print(f"  MISSING {ck}"); continue
        try:
            base, ovr, _ = override_score(ck, n)
        except Exception as e:
            print(f"  ERR {ck}: {e}"); continue
        if base < 0.99:                                 # every RUNS entry should
            print(f"  UNEXPECTED base={base:.3f} for {path}")   # clear this
            continue
        lo, l1, l2 = lin_out_metrics(ck, n)
        lab = os.path.basename(path)
        rows.append((lab, n, cls, is_prime(n), base, ovr, lo, l1, l2))
        print(f"  ...{lab:<24} {cls} base={base:.2f} override={ovr:>6.1f} "
              f"lin_out={lo:>5.0f}%", flush=True)

    rows.sort(key=lambda r: -r[6])                      # sort by lin_out
    print(f"\n  {'run':<24} {'n':>3} {'cls':>3} {'type':>9} {'override':>8} "
          f"{'lin_out':>7} {'lin_op1':>7} {'lin_op2':>7}")
    for lab, n, cls, isp, base, ovr, lo, l1, l2 in rows:
        print(f"  {lab:<24} {n:>3} {cls:>3} {'prime' if isp else 'composite':>9} "
              f"{ovr:>8.1f} {lo:>6.0f}% {l1:>6.0f}% {l2:>6.0f}%")

    save("lin_out_all_seeds",
         "run,n,cls,is_prime,base,override,lin_out,lin_op1,lin_op2",
         [(r[0], r[1], r[2], int(r[3])) + tuple(f"{v:.4f}" for v in r[4:])
          for r in rows])

    los = np.array([r[6] for r in rows])
    hi = int((los >= 90).sum()); lo_ = int((los <= 60).sum()); mid = int(((los > 60) & (los < 90)).sum())
    print(f"\n  n_grokked={len(rows)}   lin_out>=90: {hi}   lin_out<=60: {lo_}   "
          f"60<mid<90: {mid}")
    print(f"  lin_out distribution: {np.round(np.sort(los))}")
    print("  bimodal (gap, mid~0) => fork real on real population.  spread => continuum.")

    # Does lin_out agree with the class labels on the runs that were NOT used to
    # set them? Disagreement here is a finding, not a nuisance.
    print(f"\n  {'cls':>4} {'k':>3} {'lin_out mean':>13} {'min':>6} {'max':>6}")
    for c in ("A", "B", "I"):
        v = np.array([r[6] for r in rows if r[2] == c])
        if len(v):
            print(f"  {c:>4} {len(v):>3} {v.mean():>12.1f}% {v.min():>5.0f}% "
                  f"{v.max():>5.0f}%")
    odd = [(r[0], r[2], r[6]) for r in rows
           if (r[2] == "A" and r[6] < 90) or (r[2] == "B" and r[6] > 60)]
    print(f"  runs where lin_out contradicts the label: "
          f"{odd if odd else 'none'}")


if __name__ == "__main__":
    main()
