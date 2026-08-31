"""Analyze the label-shuffle control models (conc + lattice).

Falsifier for the whole instrument: a pure memorizer (trained on shuffled
labels) must NOT show clean single-tones or lattice structure. If it does, the
conc/GCR and lattice-tracking results are pipeline artifacts.

Persisted 2026-08-13: the result was previously readable only in a stray
runs/shuffle_control_RESULTS.txt, while the appendix promises it as a control.
Both arms now write to results/shuffle_control.csv.

Run after training:  python -m instruments.shuffle_analysis
"""
import numpy as np

from instruments.population import run_dir
from instruments.results_io import save
from instruments.rung3_dihedral import load_model, QUADRANTS
from instruments import d59_basis_test as d59
from instruments import d45_lattice_test as d45


def main():
    print("=" * 72)
    print("LABEL-SHUFFLE CONTROL ANALYSIS  (memorizers: train~1, val~chance)")
    print("model_acc below = accuracy on REAL group products -> must be ~chance")
    print("=" * 72)

    rows = []

    # --- D_59: conc arm (the label-shuffle control) ---
    m59 = load_model(run_dir("d59_shuffle/checkpoints/final_model.pt"), 59)
    cm = d59.analyze(m59, "D_59 LABEL-SHUFFLE (memorized)")
    for q in QUADRANTS:
        rows.append(("d59_shuffle", 59, q, f"{np.median(cm[q]):.6f}",
                     f"{(cm[q] >= d59.HI).mean():.6f}", ""))

    # --- D_45: conc (via active-set) + lattice statistic ---
    m45 = load_model(run_dir("d45_shuffle/checkpoints/final_model.pt"), 45)
    A, acc, per_q = d45.active_set(m45, "D_45 LABEL-SHUFFLE")
    d45.report("D_45 LABEL-SHUFFLE (memorized)", A, acc, per_q)
    for q in QUADRANTS:
        nclean, ks = per_q[q]
        rows.append(("d45_shuffle", 45, q, "", "",
                     " ".join(str(k) for k in ks)))

    print()
    save("shuffle_control", "run,n,quadrant,med_conc,frac_clean,active_freqs", rows)

    print("\nEXPECTED: both near random-init (no clean tones, ~empty active set),")
    print("model_acc ~chance. Any clean single-tone structure here impeaches the tool.")


if __name__ == "__main__":
    main()
