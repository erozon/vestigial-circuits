"""Is class B's 50% at the final residual the model's property or the optimizer's?
(2026-08-13)

The claim "nothing linear in the final residual predicts the output type in class
B" rests on adversarial_checks.py CHECK 2, which fits a logistic probe with Adam
(lr 0.05, 1000 steps) on RAW residual features -- no standardization, no
regularization, no learning-rate search. The hidden-layer probe in
lin_out_all_seeds.py standardizes its inputs; this one does not. Class A reaches
100% either way, but class A is an easy target: one direction, enormous margin. A
weak signal on badly-scaled features is exactly what an unregularized fixed-step
optimizer fails to find, and it would report chance while doing so.

So: same site, same labels, five fits of increasing strength. If class B stays at
chance through all of them, the absence is the model's.

  raw_adam  : the original -- raw features, Adam lr 0.05, 1000 steps
  std_adam  : identical but standardized features
  std_sweep : standardized, best held-out accuracy over lr in {0.01, 0.05, 0.2}
              at 3000 steps (optimistically biased ON PURPOSE -- picking the best
              of several fits inflates the number, which is the direction that
              makes a chance result meaningful)
  ridge     : closed-form least squares on +-1 labels, standardized, small ridge.
              No optimizer at all, so it cannot underfit.
  mlp       : one hidden layer of 64 units. Not a linear probe -- it answers the
              different question of whether the type is recoverable at this site
              AT ALL, which a reviewer will ask.

Reading, stated before running:
  - class B at chance under ridge and the sweep => the linear absence is real and
    not an artifact of how the probe was fit.
  - class B rising under any of them => adversarial_checks CHECK 2 understated the
    signal and every sentence resting on it needs revising.
  - mlp well above linear in class B => the type IS present at the residual but
    not linearly, which is a different claim from "not present" and would have to
    be stated that way.

Run: python -m instruments.residual_probe_robustness
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_margin_decomposition import resid_and_logits
from instruments.type_xor_probe import (all_pairs_typed, lin_probe,
                                                    mlp_probe, split)

LRS = [0.01, 0.05, 0.2]
SWEEP_STEPS = 3000


def ridge_acc(X, y, tr, te, lam=1e-3):
    """Closed-form least squares to +-1 targets; held-out sign accuracy."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    yt = np.where(y == 1, 1.0, -1.0)
    A = Xa[tr]
    G = A.T @ A + lam * np.trace(A.T @ A) / A.shape[1] * np.eye(Xa.shape[1])
    w = np.linalg.solve(G, A.T @ yt[tr])
    return float(100 * ((Xa[te] @ w > 0) == (yt[te] > 0)).mean())


def main():
    torch.manual_seed(0)
    print("  Probing the FINAL RESIDUAL for output type, five ways. Chance = 50.\n")
    print(f"  {'run':24} {'cls':>3} | {'raw_adam':>8} {'std_adam':>8} "
          f"{'std_sweep':>9} {'ridge':>6} {'mlp':>6}")
    rows = []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        _, tt = truth(pairs, n)
        m, tok = build(p, n)
        x, _ = resid_and_logits(m, tok, pairs, n)

        tr, te = split(len(pairs))
        Xr = torch.tensor(x, dtype=torch.float32)
        xs = (x - x.mean(0)) / (x.std(0) + 1e-6)
        Xs = torch.tensor(xs, dtype=torch.float32)

        raw = lin_probe(Xr, tt, tr, te)
        std = lin_probe(Xs, tt, tr, te)
        sweep = max(lin_probe(Xs, tt, tr, te, steps=SWEEP_STEPS) for _ in [0])
        # lin_probe hardcodes its lr, so vary the scale of the features instead:
        # scaling X by c is equivalent to scaling the effective step size.
        for c in (LRS[0] / LRS[1], LRS[2] / LRS[1]):
            sweep = max(sweep, lin_probe(Xs * c, tt, tr, te, steps=SWEEP_STEPS))
        rg = ridge_acc(xs, tt, tr, te)
        ml = mlp_probe(Xs, tt, tr, te)

        print(f"  {os.path.basename(path):24} {cls:>3} | {raw:>7.1f}% {std:>7.1f}% "
              f"{sweep:>8.1f}% {rg:>5.1f}% {ml:>5.1f}%", flush=True)
        rows.append((os.path.basename(path), n, cls) +
                    tuple(f"{v:.4f}" for v in (raw, std, sweep, rg, ml)))

    save("residual_probe_robustness",
         "run,n,cls,raw_adam,std_adam,std_sweep,ridge,mlp", rows)

    print(f"\n  {'cls':>4} {'k':>3} {'raw_adam':>18} {'std_adam':>18} "
          f"{'std_sweep':>18} {'ridge':>18} {'mlp':>18}")
    arr = np.array([[float(v) for v in r[3:]] for r in rows])
    cl = np.array([r[2] for r in rows])
    for c in ("A", "B", "I"):
        v = arr[cl == c]
        if not len(v):
            continue
        cells = "".join(f"{v[:, j].mean():>8.1f} [{v[:, j].min():.0f},{v[:, j].max():.0f}]"
                        for j in range(v.shape[1]))
        print(f"  {c:>4} {len(v):>3} {cells}")


if __name__ == "__main__":
    main()
