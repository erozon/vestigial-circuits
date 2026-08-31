"""Is the index circuit the same in all 35 generalized models? (2026-08-05)

The paper's answer-key claim -- every generalized model computes the product's
index the same way -- was until now supported only by d59_basis_test.py, which
hardcodes N=59 and SEEDS=[0,1,3]. Three runs at one group order cannot carry a
claim about 35 runs at seven, and in particular cannot answer the question the
whole argument leans on: whether class A and class B compute the INDEX alike.
If they differed there too, "both classes reach the answer by the same means"
does not hold and the paper changes shape.

This generalizes the d59 measurement over the population of record. Same metric
(rung0-certified `conc`), same four-quadrant reduction at fixed g1, same a0
robustness sweep -- only the model loop is new.

Reported per run, and then aggregated BY CLASS, which is the comparison that
matters:
  medConc   median top-1 folded spectral concentration over neurons (chance for
            a random length-n vector is ~0.14; a clean single tone is ~1.0)
  %>=0.5    fraction of neurons that are cleanly single-frequency
  #freq     how many distinct dominant frequencies the clean neurons use
  instab    mean sd of conc across the a0 offsets; conc is phase-blind, so a
            clean tone should be near-invariant and instability flags non-tones

Run: python -m instruments.index_conc_population
"""
import os

import numpy as np

from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.rung0_synthetic import conc
from instruments.rung3_dihedral import load_model, sweep, QUADRANTS

A0S = [1, 3, 5, 7, 11]
HI = 0.5


def mode_int(xs):
    u, c = np.unique(xs, return_counts=True)
    return int(u[np.argmax(c)])


def measure(model, n):
    """Returns (med_conc, frac_clean, n_active_freq, instab, model_acc), pooled
    over the four quadrants -- the per-quadrant split is in d59_basis_test and
    was flat there, so pooling is what the population table needs."""
    d_ff = model.transformer_blocks[0].feed_forward.fc1.out_features
    C = {q: np.empty((len(A0S), d_ff)) for q in QUADRANTS}
    K = {q: np.empty((len(A0S), d_ff), dtype=int) for q in QUADRANTS}
    accs = []
    for q in QUADRANTS:
        for ai, a0 in enumerate(A0S):
            pre, acc = sweep(model, n, q, a0 % n)
            accs.append(acc)
            for j in range(d_ff):
                C[q][ai, j], K[q][ai, j] = conc(pre[:, j])
    med, clean, freqs, instab = [], [], set(), []
    for q in QUADRANTS:
        cm, cs = C[q].mean(0), C[q].std(0)
        med.append(np.median(cm))
        clean.append((cm >= HI).mean())
        instab.append(cs.mean())
        for j in np.where(cm >= HI)[0]:
            freqs.add(int(mode_int(K[q][:, j])))
    return (float(np.mean(med)), float(np.mean(clean)), len(freqs),
            float(np.mean(instab)), float(np.mean(accs)))


def main():
    rng = np.random.default_rng(0)
    print("  INDEX CONCENTRATION ACROSS THE POPULATION OF RECORD\n")
    print(f"  {'run':>34} {'n':>3} {'cls':>3} {'acc':>6} {'medConc':>8} "
          f"{'%>=0.5':>7} {'#freq':>6} {'instab':>7}")
    rows = []
    for path, n, cls in RUNS:
        c = ckpt(path)
        if not os.path.exists(c):
            print(f"  {os.path.basename(path):>34} {n:>3} {cls:>3}   MISSING {c}")
            continue
        m = load_model(c, n)
        med, frac, nf, ins, acc = measure(m, n)
        rows.append((path, n, cls, acc, med, frac, nf, ins))
        print(f"  {os.path.basename(path):>34} {n:>3} {cls:>3} {acc:>6.3f} "
              f"{med:>8.3f} {100*frac:>6.0f}% {nf:>6} {ins:>7.3f}", flush=True)

    print()
    save("index_conc_population", "run,n,cls,acc,med_conc,frac_clean,n_freq,instab",
         [(os.path.basename(r[0]),) + r[1:] for r in rows])

    print(f"\n  chance reference (random length-n vectors):")
    for n in sorted({n for _, n, _ in RUNS}):
        ch = [conc(rng.standard_normal(n))[0] for _ in range(2000)]
        print(f"    n={n}: mean {np.mean(ch):.3f}  95th pct {np.percentile(ch, 95):.3f}")

    print("\n  BY CLASS -- the comparison the paper's answer key rests on\n")
    print(f"  {'cls':>4} {'k':>3} {'medConc':>18} {'%>=0.5':>16} {'#freq':>14}")
    arr = np.array([(r[4], r[5], r[6]) for r in rows])
    cl = np.array([r[2] for r in rows])
    for c in ("A", "B", "I"):
        v = arr[cl == c]
        if not len(v):
            continue
        print(f"  {c:>4} {len(v):>3} "
              f"{v[:,0].mean():>10.3f} +- {v[:,0].std():>4.3f} "
              f"{100*v[:,1].mean():>12.0f}% {v[:,2].mean():>13.1f}")
    a, b = arr[cl == "A"], arr[cl == "B"]
    if len(a) and len(b):
        print(f"\n  A - B difference in median concentration: "
              f"{a[:,0].mean() - b[:,0].mean():+.3f}")
        print(f"  worst single run in the population: "
              f"{arr[:,0].min():.3f} (floor for the claim)")


if __name__ == "__main__":
    main()
