"""D_59 binary basis test (conc arm) -- workstream 2, frozen Experiment 1.

PRIMARY metric = conc (Fourier energy concentration over the rotation index),
per the 2026-07-05 resolution (Amendment 7). C_H is the required SECONDARY and
is not built yet -- this run is the conc arm only.

Reduction: Eric's four-quadrant (fix g1, vary g2 over rotation index, condition
on fixed g1 with a0 robustness sweep), validated on Z_113 + D_29.

Pre-registered outcomes (frozen, do not adjust post hoc):
  (a) single-frequency dominates (high conc, few active freqs) -> GCR-like
  (b) delta-localization on reflection cosets -> coset-like (needs C_H to confirm;
      conc alone shows LOW conc / spread as the coset signature)
  (c) mixed/ambiguous -> descriptive finding, no post-hoc verdict

Baselines: random-init model (here) + chance ref (random length-59 vector).
Label-shuffle control is the remaining planned baseline -> needs a short
memorization training run; FLAGGED as pending, not run here.

Run:  python -m instruments.d59_basis_test
"""

import numpy as np

from instruments.population import run_dir
from instruments.rung0_synthetic import conc
from instruments.rung3_dihedral import load_model, sweep, QUADRANTS
from src.model import DihedralModel

N = 59
SEEDS = [0, 1, 3]
A0S = [1, 3, 5, 7, 11]
HI = 0.5


def mode_int(xs):
    u, c = np.unique(xs, return_counts=True)
    return int(u[np.argmax(c)])


def analyze(model, label):
    d_ff = model.transformer_blocks[0].feed_forward.fc1.out_features
    C = {q: np.empty((len(A0S), d_ff)) for q in QUADRANTS}
    K = {q: np.empty((len(A0S), d_ff), dtype=int) for q in QUADRANTS}
    accs = []
    for q in QUADRANTS:
        for ai, a0 in enumerate(A0S):
            pre, acc = sweep(model, N, q, a0 % N)
            accs.append(acc)
            for j in range(d_ff):
                C[q][ai, j], K[q][ai, j] = conc(pre[:, j])
    cm = {q: C[q].mean(0) for q in QUADRANTS}
    cs = {q: C[q].std(0) for q in QUADRANTS}
    kk = {q: np.array([mode_int(K[q][:, j]) for j in range(d_ff)]) for q in QUADRANTS}
    print(f"\n=== {label}  (model_acc={np.mean(accs):.3f}) ===")
    print(f"{'quad':>5} {'medConc':>8} {'%>=0.5':>7} {'#activeFreq':>12} {'a0-instab':>10}")
    for q in QUADRANTS:
        clean = cm[q] >= HI
        active = sorted(set(int(kk[q][j]) for j in np.where(clean)[0]))
        print(f"{q:>5} {np.median(cm[q]):>8.3f} {100*clean.mean():>6.0f}% "
              f"{len(active):>12} {cs[q].mean():>10.3f}")
    # union active freqs across quadrants among clean-in-rr neurons
    clean_rr = cm['rr'] >= HI
    act = sorted(set(int(kk['rr'][j]) for j in np.where(clean_rr)[0]))
    print(f"  active freqs (rr, clean): {act}  ({len(act)} of {N//2})")
    return {q: cm[q] for q in QUADRANTS}


def main():
    print(f"D_59 BINARY BASIS TEST (conc arm) -- {len(SEEDS)} grokked seeds + random-init")
    print(f"conc floor 1/{N//2} = {1/(N//2):.3f}")
    # chance reference
    rng = np.random.default_rng(0)
    ch = [conc(rng.standard_normal(N))[0] for _ in range(3000)]
    print(f"chance (random length-{N} vec): mean={np.mean(ch):.3f} 95pct={np.percentile(ch,95):.3f}")

    for s in SEEDS:
        m = load_model(run_dir(f"seed_sweep/d59_30pct_seed{s}/checkpoints/best_model.pt"), N)
        analyze(m, f"seed {s} (grokked)")

    # random-init baseline (same architecture, untrained)
    rand_model = DihedralModel(vocab_size=2*N+4, d_model=128, num_layers=1, num_heads=4,
                               d_ff=512, max_seq_length=8, dropout=0.0,
                               use_layernorm=False, learned_pos_emb=True, nanda_init=True)
    rand_model.eval()
    analyze(rand_model, "RANDOM-INIT baseline")

    print("\nPENDING baseline: label-shuffle control (needs a memorization training run).")


if __name__ == "__main__":
    main()
