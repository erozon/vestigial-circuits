"""Does the SIGN of t_hat . x give the type, with the threshold fixed at zero?
(2026-08-16)

unembed_type_direction_readout.py reports AUC, which is threshold-free, and
acc_thr, which picks the best threshold by scanning -- one fitted scalar. Neither
licenses the sentence "the sign of t.x gives the type", because AUC = 1.000 says
only that SOME threshold separates the two types perfectly. It could sit far from
zero, in which case the readout needs a fitted offset after all and the paper must
say "a threshold on t.x" instead of "the sign of t.x".

This settles it. Same construction as the AUC instrument --

    t_hat = sum_c (U[r_c] - U[s_c]) / || . ||     [weights only]
    beta  = t_hat . x                             [x = final residual, fc_out input]

-- but scored at the fixed threshold zero, with no fitting anywhere. Reported:

  acc_sign0 : accuracy of sign(beta) against the true output type. Orientation is
              resolved by taking the better of the two global sign conventions,
              which is a choice of labelling, not a fitted parameter.
  sep_thr   : midpoint of the separating gap when the classes are linearly
              separable along beta, else blank. This is where the perfect
              threshold actually sits.
  thr_sd    : sep_thr in units of the sd of beta -- how far the ideal threshold is
              from zero on the scale of the data.

Result (2026-08-16, all 35 runs): class A is 1.0000 in 15 of 16 runs and 0.9999 in
d59_30pct_seed4_resumed, which misses one input of 13924. The separating threshold
sits within 0.38 sd of zero in every class A run. So the sign alone carries it.
Class B averages 0.517 with one run at 0.616; report class B by AUC (0.505), not
by this statistic, since sign accuracy at a fixed threshold can exceed the AUC
when the two score distributions differ in spread.

Run: python -m instruments.sign_readout
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.unembed_type_geometry import load_unembed
from src.dihedral import multiply


def beta_and_labels(ck, n):
    """Projection of every input's final residual onto t_hat, and the true types."""
    m, tok = build(ck, n)
    U, _ = load_unembed(ck)
    t = torch.tensor(U[:n] - U[n:2 * n], dtype=torch.float32).mean(0)
    t = t / t.norm()

    toks = [f"r{i}" for i in range(n)] + [f"s{i}" for i in range(n)]
    pairs = [(a, b) for a in toks for b in toks]
    is_rot = np.array([multiply(a, b, n).startswith("r") for a, b in pairs])

    grabbed = {}
    h = m.fc_out.register_forward_pre_hook(
        lambda mod, inp: grabbed.__setitem__("x", inp[0].detach()))
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                        for a, b in pairs])
    with torch.no_grad():
        m(ids)
    h.remove()
    return (grabbed["x"][:, 2, :] @ t).numpy(), is_rot


def main():
    print("  Type accuracy of sign(t_hat . x), threshold fixed at zero.\n")
    print(f"  {'run':26} {'cls':>3} {'acc_sign0':>10} {'n_wrong':>8} "
          f"{'sep_thr':>9} {'thr_sd':>8}")
    rows = []
    for path, n, cls in RUNS:
        ck = ckpt(path)
        run = os.path.basename(path)
        if not os.path.exists(ck):
            print(f"  {run:26} {cls:>3}  (no checkpoint)")
            continue
        beta, is_rot = beta_and_labels(ck, n)
        # Orientation is a labelling convention, not a fitted parameter.
        acc = max(((beta > 0) == is_rot).mean(), ((beta <= 0) == is_rot).mean())
        n_wrong = int(round((1 - acc) * len(beta)))

        lo, hi = beta[is_rot].min(), beta[~is_rot].max()
        lo2, hi2 = beta[~is_rot].min(), beta[is_rot].max()
        if lo > hi:
            thr = (lo + hi) / 2
        elif lo2 > hi2:
            thr = (lo2 + hi2) / 2
        else:
            thr = None
        sd = beta.std()

        ts = f"{thr:.4f}" if thr is not None else ""
        td = f"{thr / sd:.3f}" if thr is not None else ""
        print(f"  {run:26} {cls:>3} {acc:>10.4f} {n_wrong:>8} {ts:>9} {td:>8}")
        rows.append((run, n, cls, f"{acc:.6f}", n_wrong, len(beta), ts, td))
    print()
    save("sign_readout", "run,n,cls,acc_sign0,n_wrong,n_inputs,sep_thr,thr_sd", rows)


if __name__ == "__main__":
    main()
