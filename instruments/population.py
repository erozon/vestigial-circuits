"""The population of record (fixed 2026-08-04).

Every run that reached the >= 0.99 validation-accuracy generalization rule, with
its group order and its measured type-variable class. 35 runs of 47 trained,
spanning seven group orders (D_49 trained 3, generalized 0, so it is absent).

Class labels come from two instruments:
  lin_out : linear decode of output type from the MLP hidden layer
  auc_t   : zero-parameter readout along t = mean_c(U[r_c] - U[s_c])
"A" = lin_out 100 and auc_t 1.000. "B" = auc_t ~0.50. "I" = the two runs that sit
between the modes (d47_seed6: 79 / 0.683; d61_seed4: 85 / 0.663). They are NOT
silently binned into A or B.

CORRECTION 2026-08-13. This file used to say the two instruments "agree on every
run". They do not, and the earlier claim was an artifact of scope: lin_out had
only ever been measured on 23 runs, all of which happened to be clean. Measured on
all 35 (results/lin_out_all_seeds.csv):
  auc_t   is sharply split -- A exactly 1.000 (16 runs), B 0.505 [0.500, 0.548]
  lin_out is not -- A exactly 100% (16 runs), B 56.1% [45%, 70%], with five B runs
          above 60%: d63_seed4 70, d63_seed5 69, d63_seed3 69, d63_seed7 62,
          d61_seed6 60. All five are dense-sweep runs, i.e. runs the old 23-run
          scope never touched.
The labels themselves are unchanged, because auc_t (and the fitted final-residual
probe in adversarial_checks, class B 50.0% [49, 51]) is what defines them. What is
no longer true is that the MLP hidden layer carries no type information in class B:
in some B runs it carries a decodable amount, well above the 50% chance level,
while the same models' final residual sits at chance. Any sentence in the paper of
the form "the probe is at chance in class B" must name the site it refers to.

Provenance note (corrected 2026-08-28): SIXTEEN (order, seed) pairs exist in
BOTH the prime/composite and dense sweeps -- d47 seeds 0-4, d61 seeds 0-7,
d63 seeds 0-2 -- and every pair is bit-identical at every shared checkpoint
epoch (verify_twin_runs.py), i.e. one deterministic trajectory recorded twice
at different checkpoint densities and horizons. Each pair appears in RUNS at
most once, under the longer-horizon path (d61_seed3 generalized in neither
sweep, so it is absent from RUNS). Any analysis that
also uses the dense D_63 sweep as a replication set must exclude seeds 0-2, which
are already in this population; the genuinely out-of-sample D_63 seeds are 3-9.

Import RUNS (or the helpers) rather than re-deriving a run list per script.
"""
import os


# (path, n, class)
RUNS = [
    ("runs/prime_composite_sweep/d47_seed0", 47, "A"),
    ("runs/prime_composite_sweep/d47_seed1", 47, "A"),
    ("runs/prime_composite_sweep/d47_seed3", 47, "A"),
    ("runs/prime_composite_sweep/d47_seed4", 47, "B"),
    ("runs/dense_checkpoint_sweep/d47_seed6", 47, "I"),
    ("runs/dense_checkpoint_sweep/d47_seed7", 47, "A"),
    ("runs/dense_checkpoint_sweep/d47_seed8", 47, "A"),
    ("runs/dense_checkpoint_sweep/d47_seed9", 47, "A"),

    ("runs/prime_composite_sweep/d51_seed0", 51, "B"),
    ("runs/prime_composite_sweep/d51_seed2", 51, "A"),

    ("runs/prime_composite_sweep/d53_seed1", 53, "A"),
    ("runs/prime_composite_sweep/d53_seed2", 53, "B"),

    ("runs/prime_composite_sweep/d55_seed0", 55, "B"),
    ("runs/prime_composite_sweep/d55_seed2", 55, "B"),

    ("runs/seed_sweep/d59_30pct_seed0", 59, "A"),
    ("runs/seed_sweep/d59_30pct_seed1", 59, "A"),
    ("runs/seed_sweep/d59_30pct_seed3", 59, "A"),
    ("runs/seed_sweep/d59_30pct_seed4_resumed", 59, "A"),

    ("runs/prime_composite_sweep/d61_seed0", 61, "B"),
    ("runs/prime_composite_sweep/d61_seed1", 61, "B"),
    ("runs/prime_composite_sweep/d61_seed2", 61, "A"),
    ("runs/prime_composite_sweep/d61_seed4", 61, "I"),
    ("runs/prime_composite_sweep/d61_seed6", 61, "B"),
    ("runs/dense_checkpoint_sweep/d61_seed8", 61, "A"),
    ("runs/dense_checkpoint_sweep/d61_seed9", 61, "B"),

    ("runs/prime_composite_sweep/d63_seed0", 63, "B"),
    ("runs/prime_composite_sweep/d63_seed1", 63, "B"),
    ("runs/prime_composite_sweep/d63_seed2", 63, "A"),
    ("runs/dense_checkpoint_sweep/d63_seed3", 63, "B"),
    ("runs/dense_checkpoint_sweep/d63_seed4", 63, "B"),
    ("runs/dense_checkpoint_sweep/d63_seed5", 63, "B"),
    ("runs/dense_checkpoint_sweep/d63_seed6", 63, "B"),
    ("runs/dense_checkpoint_sweep/d63_seed7", 63, "B"),
    ("runs/dense_checkpoint_sweep/d63_seed8", 63, "A"),
    ("runs/dense_checkpoint_sweep/d63_seed9", 63, "B"),
]

# Trained but did not generalize, by group order. Kept here so the denominator in
# the paper is derived from this file rather than re-counted by hand.
NOT_GENERALIZED = {
    47: ["seed2 (0.9895)", "seed5 (0.7732)"],
    49: ["seed0 (0.9221)", "seed1 (0.9769)", "seed2 (0.7803)"],
    51: ["seed1 (0.9883)"],
    53: ["seed0 (0.9626)"],
    55: ["seed1 (0.9353)"],
    59: ["seed2 (0.9645 resumed)"],
    61: ["seed3 (0.9826)", "seed5 (0.9329)", "seed7 (0.9880)"],
}

TRAINED_PER_ORDER = {47: 10, 49: 3, 51: 3, 53: 3, 55: 3, 59: 5, 61: 10, 63: 10}


# Where the trained runs live. The paths below are written relative to a "runs"
# directory because that is where they sat while the work was done; a reader who
# unpacks the released checkpoints somewhere else sets DIHEDRAL_RUNS and every
# instrument follows, since they all reach the filesystem through run_dir().
ROOT = os.environ.get("DIHEDRAL_RUNS", "runs")


def run_dir(path):
    """Re-root a path written as 'runs/...' onto ROOT. Also accepts a bare
    relative path, so glob patterns can be written without the prefix."""
    return os.path.join(ROOT, path[5:] if path.startswith("runs/") else path)


def ckpt(path):
    return f"{run_dir(path)}/checkpoints/best_model.pt"


def epoch_checkpoints(rd, pattern="checkpoint_epoch_*.pt"):
    """Every saved epoch of a run directory, sorted by epoch.

    Raises when the run has none. Skipping silently is worse than it sounds: a
    trajectory instrument that finds no epoch checkpoints still runs, still
    writes its CSV, and writes one row per run where there should be one row per
    checkpoint -- and the instruments downstream read that CSV. The released
    final-checkpoint archive has exactly this shape, so a reader who skips the
    trajectory download hits it on the default path.
    """
    import glob as _glob
    cks = _glob.glob(os.path.join(rd, "checkpoints", pattern))
    if not cks:
        raise FileNotFoundError(
            f"{rd} has no {pattern}. This measurement needs the per-epoch "
            "checkpoints, not just the final weights: unpack the trajectory "
            "archive and point DIHEDRAL_RUNS at it.")
    return sorted(cks, key=lambda c: int(c.split("_")[-1].split(".")[0]))


def by_class(cls):
    return [(p, n) for p, n, c in RUNS if c == cls]


def counts():
    out = {}
    for _, n, c in RUNS:
        out.setdefault(n, {"A": 0, "B": 0, "I": 0})[c] += 1
    return out


if __name__ == "__main__":
    c = counts()
    print(f"  {'n':>4} {'trained':>8} {'gen':>5} {'A':>3} {'B':>3} {'I':>3}")
    for n in sorted(TRAINED_PER_ORDER):
        d = c.get(n, {"A": 0, "B": 0, "I": 0})
        print(f"  {n:>4} {TRAINED_PER_ORDER[n]:>8} {sum(d.values()):>5} "
              f"{d['A']:>3} {d['B']:>3} {d['I']:>3}")
    tot = {k: sum(d[k] for d in c.values()) for k in "ABI"}
    print(f"\n  {sum(TRAINED_PER_ORDER.values())} trained, {len(RUNS)} generalized; "
          f"A={tot['A']} B={tot['B']} between={tot['I']}")
