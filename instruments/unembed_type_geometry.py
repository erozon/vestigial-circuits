"""Output-side geometry of the type distinction (2026-08-04).

Every measurement of the localized/non-localized fork so far lives in the MLP
hidden layer. This one looks at the OTHER end: the unembedding rows, which is
what actually decides the answer.

At the result position the choice between the two tokens sharing index c is
governed entirely by

    D_c = U[r_c] - U[s_c]

The answer is a rotation at index c iff the residual has a positive component
along D_c. So:

  * if the D_c are near-PARALLEL across c, there is ONE global "rotation-ness"
    direction in output space. A single hidden feature writing to it raises every
    rotation logit and lowers every reflection logit at once -- a type bit is
    worth computing, and a linear probe of the hidden layer will find it.
  * if the D_c point in DIFFERENT directions per c, no such global direction
    exists. A feature pushing toward rotation at one index pushes toward
    reflection at another; nothing is gained by holding a separate type bit, and
    the network can only emit the right token directly.

PREDICTION under that account: common_frac (below) is HIGH in runs where output
type is globally linearly decodable from the hidden layer (lin_out ~100) and LOW
in runs where it is at chance (lin_out ~50). If both classes show the same output
geometry, the account is WRONG and the fork lives elsewhere.

Metrics per run (all scale-invariant, weights only -- no forward passes):
  common_frac : ||mean_c D_c||^2 / mean_c ||D_c||^2, in [0,1]. Fraction of the
                per-index type-difference energy carried by the SHARED component.
                Chance reference for unstructured rows is ~1/n (see printout).
  mean_cos    : mean_c cos(D_c, mean D). 1 = every index separates its two tokens
                along the same direction; ~0 = index-specific.
  min_cos     : the worst index (does ANY index disagree in sign?).
  sv1_frac    : top singular value share of D -- how close D is to rank 1
                (rank-1 = one shared axis, possibly with per-index scaling).
  sv1_cos_t   : |cos| between the top singular direction and the shared component
                (separates "one axis, sign flips per index" from "one direction").
  bias_gap    : if the output layer has a bias, mean_c(b_r_c - b_s_c) normalized
                by its own std across c -- a constant type offset can also live
                here rather than in the rows.

SCOPE FIX (2026-08-13): this ran over a glob of the prime/composite sweep plus
three D_59 seeds, filtered by a hardcoded LIN_OUT dict copied from a 2026-08-04
run of lin_out_all_seeds -- 23 runs, and a table of numbers that could silently
fall out of date. It now iterates population.RUNS (35 runs) and reads lin_out from
results/lin_out_all_seeds.csv if that instrument has been run, leaving it blank
otherwise. The correlation against lin_out is computed on whichever runs have it.

Run: python -m instruments.unembed_type_geometry
"""
import os

import numpy as np
import torch

from instruments.population import RUNS, ckpt
from instruments.results_io import column, save


def load_unembed(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    U = sd["fc_out.weight"].float().numpy()
    b = sd.get("fc_out.bias")
    return U, (b.float().numpy() if b is not None else None)


def geometry(U, b, n):
    D = U[:n] - U[n:2 * n]                       # (n, d) per-index type difference
    t = D.mean(0)
    common = float(t @ t) / float((D * D).sum(1).mean())
    tn = t / (np.linalg.norm(t) + 1e-12)
    cos = (D @ tn) / (np.linalg.norm(D, axis=1) + 1e-12)
    sv = np.linalg.svd(D, compute_uv=False)
    sv1 = float(sv[0] ** 2 / (sv ** 2).sum())
    v1 = np.linalg.svd(D, full_matrices=False)[2][0]
    sv1_cos_t = abs(float(v1 @ tn))
    if b is not None:
        bd = b[:n] - b[n:2 * n]
        bias_gap = float(bd.mean() / (bd.std() + 1e-12))
    else:
        bias_gap = float("nan")
    return common, float(cos.mean()), float(cos.min()), sv1, sv1_cos_t, bias_gap


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    try:
        lin_out = column("lin_out_all_seeds", "lin_out")
    except FileNotFoundError:
        lin_out = {}
        print("  (no results/lin_out_all_seeds.csv -- lin_out columns left blank; "
              "run lin_out_all_seeds first for the correlation)\n")

    rows = []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        run = os.path.basename(path)
        U, b = load_unembed(p)
        rows.append((run, n, cls, lin_out.get(run)) + geometry(U, b, n))

    print("  Output-side type geometry.  D_c = U[r_c] - U[s_c].")
    print("  common_frac chance reference for unstructured rows ~ 1/n "
          f"(~{1/59:.3f} at n=59).\n")
    print(f"  {'run':24} {'n':>3} {'cls':>3} {'lin_out':>7} | {'common':>7} "
          f"{'mean_cos':>8} {'min_cos':>8} {'sv1':>6} {'sv1~t':>6} {'bias_gap':>8}")
    for r in sorted(rows, key=lambda x: (x[2], -x[4])):
        lo_s = f"{r[3]:>6.0f}%" if r[3] is not None else f"{'--':>7}"
        print(f"  {r[0]:24} {r[1]:>3} {r[2]:>3} {lo_s} | {r[4]:>7.3f} {r[5]:>8.3f} "
              f"{r[6]:>8.3f} {r[7]:>6.3f} {r[8]:>6.3f} {r[9]:>8.2f}")

    save("unembed_type_geometry",
         "run,n,cls,lin_out,common_frac,mean_cos,min_cos,sv1_frac,sv1_cos_t,bias_gap",
         [(r[0], r[1], r[2], "" if r[3] is None else f"{r[3]:.4f}") +
          tuple(f"{v:.6f}" for v in r[4:]) for r in rows])

    print(f"\n  by class:")
    for c in ("A", "B", "I"):
        sel = [r for r in rows if r[2] == c]
        if not sel:
            continue
        for j, name in [(4, "common_frac"), (5, "mean_cos"), (7, "sv1_frac")]:
            v = [r[j] for r in sel]
            print(f"    {c} {name:12} mean={np.mean(v):.3f} "
                  f"[{min(v):.3f},{max(v):.3f}]  (k={len(sel)})")

    have = [r for r in rows if r[3] is not None]
    if len(have) > 2:
        lin = [r[3] for r in have]
        print(f"\n  Pearson against lin_out, on the {len(have)} runs that have it:")
        for j, name in [(4, "common_frac"), (5, "mean_cos"), (7, "sv1_frac")]:
            print(f"    {name:12} {pearson(lin, [r[j] for r in have]):+.2f}")
    print("\n  READ: high common/mean_cos in lin-high runs only => output geometry "
          "explains the fork.\n        overlapping ranges => it does not.")


if __name__ == "__main__":
    main()
