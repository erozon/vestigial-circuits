"""Are class A models generically more fragile, or is the difference an artifact?
(2026-08-05)

dc_surgical_ablation.py found that removing class A's type bit costs ~12%
accuracy -- but so does a perturbation of the same magnitude that leaves the type
bit perfectly intact (noise-only, 0.869 vs surgical 0.884). So the damage is not
attributable to the type bit.

That leaves one confound. Class A has a large offset along t to remove; class B's
is ~0. "Class B is robust" may only mean that nothing was done to it. This applies
a MATCHED perturbation to both classes and compares.

Perturbation: balanced +/- offsets along a random unit direction v, applied to the
output-layer rows, scaled so that the induced change in logits is a target
fraction f of that model's own logit spread:

    delta = f * sd(logits) / mean|x . v|

so f is comparable across models regardless of weight or logit scale. Random v
rather than t, because t is exactly the direction the classes differ in, and using
it would re-introduce the confound.

Reading, stated before running:
  - Both classes degrade alike => class A's apparent fragility was an artifact of
    perturbation size, and robustness is not part of the fork.
  - Class A degrades faster at matched f => class A really is the more brittle
    solution, which is a genuine difference and worth reporting.

Run: python -m instruments.perturbation_sensitivity
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import evaluate, truth
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_margin_decomposition import resid_and_logits
from instruments.type_xor_probe import all_pairs_typed

FRACS = [0.1, 0.2, 0.4]
DRAWS = 3


def main():
    print("  Matched-magnitude perturbation sensitivity, random directions.")
    print("  f = induced logit change as a fraction of the model's own logit sd.\n")
    print(f"  {'run':24} {'cls':>3} | {'base':>6} " +
          " ".join(f"{'f=' + str(f):>7}" for f in FRACS))
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)
        m0, tok = build(p, n)
        x, logits = resid_and_logits(m0, tok, pairs, n)
        a0 = evaluate(m0, tok, pairs, n, ti, tt)[0]
        lsd = float(logits.std(1).mean())

        row = []
        for f in FRACS:
            accs = []
            for s in range(DRAWS):
                g = torch.Generator().manual_seed(300 + s)
                v = torch.randn(x.shape[1], generator=g)
                v = v / v.norm()
                xv = float(np.abs(x @ v.numpy()).mean())
                delta = f * lsd / (xv + 1e-12)

                mp, _ = build(p, n)
                with torch.no_grad():
                    U = mp.fc_out.weight
                    o = torch.zeros(U.shape[0])
                    for lo, hi in ((0, n), (n, 2 * n)):
                        k = hi - lo
                        sg = torch.ones(k)
                        sg[torch.randperm(k, generator=g)[:k // 2]] = -1.0
                        sg -= sg.mean()
                        o[lo:hi] = delta * sg
                    U -= torch.outer(o, v)
                accs.append(evaluate(mp, tok, pairs, n, ti, tt)[0])
            row.append(float(np.mean(accs)))

        print(f"  {os.path.basename(path):24} {cls:>3} | {a0:>6.3f} " +
              " ".join(f"{r:>7.3f}" for r in row))
        agg.setdefault(cls, []).append([a0] + row)
        rows.append((os.path.basename(path), n, cls) +
                    tuple(f"{v:.6f}" for v in [a0] + row))
    print()
    save("perturbation_sensitivity",
         "run,n,cls,base," + ",".join(f"f{f}" for f in FRACS), rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): base {a[:,0].mean():.3f} | " +
              " ".join(f"f={f}: {a[:, i + 1].mean():.3f}" for i, f in enumerate(FRACS)))


if __name__ == "__main__":
    main()
