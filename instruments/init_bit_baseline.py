"""Is the type bit really absent at initialization? (2026-08-13)

Section 3.4 says the bit is absent at initialization and built during training --
the claim that rules out a lottery-ticket reading, in which the direction is
present in the random weights and training merely selects models that already had
it. The evidence so far is the first available checkpoint, at epoch 1000. That is
1000 epochs of full-batch training, not initialization, so as measured the claim
was an extrapolation.

Nothing needs training to fix this: the initialization scheme is deterministic
given a seed, so fresh models can be built and measured directly. For each group
order in the population, SEEDS models are constructed exactly as src/train.py
constructs them (Nanda init, same shape) and measured with the same weights-only
statistics used at convergence:

  dc_frac  : ||mean_c D_c||^2 / mean_c||D_c||^2, D_c = U[r_c] - U[s_c]
  mean_cos : mean_c cos(D_c, mean_c D_c)
  med_conc : the index measurement of index_conc_population.py, run on the same
             fresh models. Added 2026-08-28 so that untrained models can be
             plotted beside the trained ones on both axes at once, which is what
             the fork figure does instead of drawing a shaded reference band.

Both are reported against the two things they have to be compared with: the
epoch-1000 values from the trajectory, and the converged class A and class B
values. This also supplies the chance reference the bimodality figure should be
drawn against, which otherwise has to be asserted as ~1/n.

Reading, stated before running:
  - init dc_frac at the ~1/n unstructured level in every seed => the direction is
    built, and no model starts with it.
  - a heavy right tail at init, overlapping class A's converged range => the
    lottery-ticket reading is live and Section 3.4 has a problem.

Note on what this can and cannot settle: agreement between init and chance rules
out the direction being PRESENT at init. It does not rule out some other property
of the initial weights predisposing a run toward building it -- that is
init_geometry_predictor.py, and it found no predictor.

Run: python -m instruments.init_bit_baseline
"""
import numpy as np
import torch

from instruments.index_conc_population import measure
from instruments.population import RUNS
from instruments.results_io import load, save
from instruments.unembed_type_geometry import geometry
from src.model import DihedralModel

SEEDS = 30
D_MODEL, D_FF, HEADS, LAYERS = 128, 512, 4, 1


def fresh(n, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DihedralModel(vocab_size=2 * n + 4, d_model=D_MODEL, num_layers=LAYERS,
                      num_heads=HEADS, d_ff=D_FF, max_seq_length=8, dropout=0.0,
                      use_layernorm=False, learned_pos_emb=True, nanda_init=True)
    m.eval()
    U = m.fc_out.weight.detach().numpy()
    b = m.fc_out.bias.detach().numpy() if m.fc_out.bias is not None else None
    return m, U, b


def main():
    orders = sorted({n for _, n, _ in RUNS})
    print("  Type-direction statistics at INITIALIZATION, no training.")
    print(f"  {SEEDS} fresh models per group order, built as src/train.py builds "
          f"them.\n")
    print(f"  {'group':>7} {'k':>3} | {'dc_frac mean':>13} {'max':>7} "
          f"{'95th pct':>9} | {'1/n':>6} | {'mean_cos':>9} {'max':>7}")

    rows, all_dc = [], []
    for n in orders:
        dc, mc = [], []
        for s in range(SEEDS):
            m, U, b = fresh(n, s)
            common, mean_cos, _, _, _, _ = geometry(U, b, n)
            med_conc = measure(m, n)[0]
            dc.append(common)
            mc.append(mean_cos)
            rows.append((n, s, f"{common:.6f}", f"{mean_cos:.6f}",
                         f"{med_conc:.6f}"))
        dc, mc = np.array(dc), np.array(mc)
        all_dc.append(dc)
        print(f"  {'D_' + str(n):>7} {SEEDS:>3} | {dc.mean():>13.4f} {dc.max():>7.4f} "
              f"{np.percentile(dc, 95):>9.4f} | {1 / n:>6.4f} | {mc.mean():>9.4f} "
              f"{mc.max():>7.4f}")

    save("init_bit_baseline", "n,seed,dc_frac,mean_cos,med_conc", rows)

    pooled = np.concatenate(all_dc)
    print(f"\n  pooled over {len(pooled)} initializations: dc_frac mean "
          f"{pooled.mean():.4f}, max {pooled.max():.4f}, "
          f"99th pct {np.percentile(pooled, 99):.4f}")

    # The comparison the claim actually needs: init vs the first checkpoint vs the
    # converged classes.
    try:
        traj = load("bit_emergence_trajectory")
        e1 = np.array([r["dc_frac"] for r in traj if r["epoch"] == 1000])
        print(f"  epoch 1000 across the dense sweep (k={len(e1)}): mean "
              f"{e1.mean():.4f}, range [{e1.min():.4f}, {e1.max():.4f}]")
    except FileNotFoundError:
        pass
    try:
        spec = load("type_structure_spectrum")
        for c in ("A", "B"):
            v = np.array([r["dc_frac"] for r in spec if r["cls"] == c])
            print(f"  converged class {c} (k={len(v)}): mean {v.mean():.4f}, "
                  f"range [{v.min():.4f}, {v.max():.4f}]")
    except FileNotFoundError:
        pass

    print("\n  READ: if the converged class A range sits far above every one of the "
          f"{len(pooled)}\n  initializations, the direction is built, not drawn.")


if __name__ == "__main__":
    main()
