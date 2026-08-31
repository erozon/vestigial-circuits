"""D_45 graded lattice-tracking test -- the DISCRIMINATING experiment.

Pre-registered in PLAN.md "Pre-registered experiment 3" (frozen 2026-07-06 BEFORE
any D_45 frequency was inspected). n=45=3^2*5: 22 folded rotation-index DFT bins
k=1..22, each living on the quotient D_{45/gcd(k,45)}. The proper quotients the
model could compute in are D_{45/d} for d in {3,5,9,15}, footprint = multiples of
d (M_d).

Question: do a grokked model's ACTIVE frequencies track this divisor lattice
(coset-structured), or are they lattice-blind (GCR/Fourier)?

PRIMARY statistic B (Eric's choice): min-p hypergeometric surprise.
  x_d = |A cap M_d|;  p_d = P(X >= x_d) under Hypergeom(N=22, K=|M_d|, draws=m);
  T_B = -log min_d p_d;  argmin_d = tracked quotient level (the GRADED readout).
  Significance via uniform-m-subset null (operationalizes "GCR = gcd-blind").
SECONDARY statistic A (coarse omnibus): T_A = |A cap (M3 U M5)|, exact hypergeom.

Active set A (per seed, frozen): union over four quadrants of dominant k* of
neurons with conc >= 0.5 (certified conc, a0 robustness sweep). m = |A|.

Instrument is SELF-TESTED on synthetic active sets (calibrate()) before any
checkpoint is touched. Run:  python -m instruments.d45_lattice_test
"""

from math import comb, log

import numpy as np

from instruments.population import run_dir
from instruments.results_io import save
from instruments.rung0_synthetic import conc
from instruments.rung3_dihedral import load_model, sweep, QUADRANTS
from src.model import DihedralModel

N = 45
NFREQ = N // 2                      # 22 folded bins
A0S = [1, 3, 5, 7, 11]
HI = 0.5
LEVELS = [3, 5, 9, 15]             # proper divisors d: model could compute in D_{45/d}
M = {d: [k for k in range(1, NFREQ + 1) if k % d == 0] for d in LEVELS}
NONPRIM = sorted(set(M[3]) | set(M[5]))   # 10 of 22

SEEDS = [
    ("seed1", run_dir("d45_sweep/d45_30pct_seed1/checkpoints/best_model.pt")),
    ("seed4", run_dir("d45_sweep/d45_30pct_seed4/checkpoints/best_model.pt")),
    ("seed0", run_dir("d45_pilot/seed0/checkpoints/best_model.pt")),
    ("seed2_resumed", run_dir("d45_sweep/d45_30pct_seed2_resumed/checkpoints/best_model.pt")),
]


# ---------------------------------------------------------------- statistic
def hyper_sf(x, K, m, Ntot=NFREQ):
    """P(X >= x) for Hypergeom(Ntot, K successes, m draws), exact via comb."""
    lo, hi = max(0, m - (Ntot - K)), min(m, K)
    x = max(x, lo)
    denom = comb(Ntot, m)
    return sum(comb(K, i) * comb(Ntot - K, m - i) for i in range(x, hi + 1)) / denom


def stat_B(A):
    """min-p surprise. Return (T_B, argmin_d, {d: (x_d, p_d)})."""
    A = set(A)
    m = len(A)
    detail = {}
    for d in LEVELS:
        x = len(A & set(M[d]))
        detail[d] = (x, hyper_sf(x, len(M[d]), m))
    dstar = min(LEVELS, key=lambda d: detail[d][1])
    pmin = detail[dstar][1]
    return -log(pmin), dstar, detail


def stat_A(A):
    """coarse omnibus: (count non-primitive, exact hypergeom p)."""
    A = set(A)
    x = len(A & set(NONPRIM))
    return x, hyper_sf(x, len(NONPRIM), len(A))


def null_pvalue(A, n_draws=300_000, seed=0):
    """One-sided p = P(T'_B >= T_obs) under uniform m-subsets of {1..NFREQ}."""
    m = len(A)
    t_obs = stat_B(A)[0]
    if m == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pool = np.arange(1, NFREQ + 1)
    ge = 0
    for _ in range(n_draws):
        draw = rng.choice(pool, size=m, replace=False)
        if stat_B(draw)[0] >= t_obs - 1e-12:
            ge += 1
    return ge / n_draws


# ---------------------------------------------------------------- self-test
def calibrate():
    print("STATISTIC SELF-TEST (synthetic active sets; no checkpoint touched):")
    print(f"  levels M_d sizes: " + ", ".join(f"M{d}={len(M[d])}" for d in LEVELS)
          + f"; non-primitive base {len(NONPRIM)}/{NFREQ}")
    cases = {
        "pure mult-of-3 {3,6,9,12,15}": [3, 6, 9, 12, 15],
        "pure mult-of-5 {5,10,15,20}":  [5, 10, 15, 20],
        "all-primitive {1,2,4,7,8}":    [1, 2, 4, 7, 8],
        "mixed {1,3,5,7,13}":           [1, 3, 5, 7, 13],
    }
    for name, A in cases.items():
        tb, dstar, det = stat_B(A)
        p = null_pvalue(A, n_draws=100_000, seed=1)
        xa, pa = stat_A(A)
        print(f"  {name:32} T_B={tb:5.2f} track=d{dstar} nullp={p:6.4f}"
              f"  |  A: {xa}/{len(A)} nonprim p={pa:.4f}")
    # null calibration: p-values of random m-subsets must be ~uniform (mean ~0.5)
    rng = np.random.default_rng(2)
    ps = [null_pvalue(rng.choice(np.arange(1, NFREQ + 1), size=6, replace=False),
                       n_draws=20_000, seed=int(s)) for s in range(40)]
    print(f"  null calibration (40 random m=6 sets): mean p={np.mean(ps):.3f} "
          f"(expect ~0.5), frac<0.05={np.mean(np.array(ps) < 0.05):.3f} (expect ~0.05)")
    print("  Gate: pure-sublattice sets => tiny null p at the right level;"
          " random sets => uniform p.\n")


# ---------------------------------------------------------------- active set
def mode_int(xs):
    u, c = np.unique(xs, return_counts=True)
    return int(u[np.argmax(c)])


def active_set(model, label):
    """Union over four quadrants of dominant k* of clean (conc>=HI) neurons."""
    d_ff = model.transformer_blocks[0].feed_forward.fc1.out_features
    accs, active, per_q = [], set(), {}
    for q in QUADRANTS:
        C = np.empty((len(A0S), d_ff))
        K = np.empty((len(A0S), d_ff), dtype=int)
        for ai, a0 in enumerate(A0S):
            pre, acc = sweep(model, N, q, a0 % N)
            accs.append(acc)
            for j in range(d_ff):
                C[ai, j], K[ai, j] = conc(pre[:, j])
        cm = C.mean(0)
        clean = np.where(cm >= HI)[0]
        ks = sorted({mode_int(K[:, j]) for j in clean})
        per_q[q] = (int((cm >= HI).sum()), ks)
        active |= set(ks)
    A = sorted(a for a in active if 1 <= a <= NFREQ)   # guard: drop DC/out-of-range
    return A, float(np.mean(accs)), per_q


def report(label, A, acc, per_q):
    """Prints as before, and returns one CSV row so main() can persist the run."""
    print(f"\n=== {label}  (model_acc={acc:.3f}, m=|A|={len(A)}) ===")
    for q in QUADRANTS:
        nclean, ks = per_q[q]
        print(f"  {q}: {nclean:3d} clean neurons, k*={ks}")
    print(f"  ACTIVE SET A = {A}")
    if not A:
        print("  (empty active set -> statistic n/a)")
        return (label, f"{acc:.6f}", 0, "", "", "", "", "", "")
    tb, dstar, det = stat_B(A)
    p = null_pvalue(A)
    xa, pa = stat_A(A)
    print("  B (min-p surprise): " + ", ".join(
        f"d{d}: {det[d][0]}/{len(M[d])} in M{d} (p={det[d][1]:.4f})" for d in LEVELS))
    print(f"  => T_B={tb:.3f}, tracked level d={dstar} (D_{{{N//dstar}}}), "
          f"null p={p:.5f}")
    print(f"  A (omnibus): {xa}/{len(A)} non-primitive, hypergeom p={pa:.4f}")
    return (label, f"{acc:.6f}", len(A), " ".join(str(k) for k in A),
            f"{tb:.6f}", dstar, f"{p:.6f}", xa, f"{pa:.6f}")


def main():
    calibrate()
    print("D_45 GRADED LATTICE-TRACKING TEST -- active-set lattice statistic per seed")
    print(f"(GCR null: frequency choice blind to gcd-level. LEVELS d={LEVELS}.)")

    rows = []
    for label, path in SEEDS:
        m = load_model(path, N)
        A, acc, per_q = active_set(m, label)
        rows.append(report(label, A, acc, per_q))

    rand = DihedralModel(vocab_size=2 * N + 4, d_model=128, num_layers=1, num_heads=4,
                         d_ff=512, max_seq_length=8, dropout=0.0, use_layernorm=False,
                         learned_pos_emb=True, nanda_init=True)
    rand.eval()
    A, acc, per_q = active_set(rand, "RANDOM-INIT")
    rows.append(report("RANDOM-INIT baseline", A, acc, per_q))

    print()
    save("d45_lattice_test",
         "run,acc,m,active_set,T_B,tracked_level,null_p,x_A,p_A", rows)


if __name__ == "__main__":
    main()
