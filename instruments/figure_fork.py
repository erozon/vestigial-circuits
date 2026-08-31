"""What splits the population, and what does not. (2026-08-28)

Two measurements over the same 35 runs, on the two axes of one panel.

  y  median spectral concentration of the index computation
     (index_conc_population.csv). Every generalized run is high here. This is
     the axis universality survives on.
  x  shared-direction strength, the fraction of the D_c vectors' energy carried
     by their common part (type_structure_spectrum.csv). This is the axis the population
     splits on.

The 210 freshly initialized models are plotted too, on both axes
(init_bit_baseline.csv). They are a control, not a reference band: they occupy
the region a model sits in before it has learned anything, which is what makes
the two claims visible at once. Class B sits directly above them -- same
strength, an index circuit built -- rather than merely "low".

Nothing is coloured by class: the classes are defined by the sign readout in
the text, and this figure's role is to show that the geometry divides on its
own. The gap does that work or it does not. The two runs the readout leaves
unclassified (d47_seed6, d61_seed4) are drawn as triangles; on this axis they
sit one on each side of the gap.

Run:  python -m instruments.figure_fork
Writes paper/v2/figures/fork.pdf.
"""

import matplotlib.pyplot as plt
import numpy as np

from instruments.results_io import figure_path, load


plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
})


def main():
    out = figure_path("fork")
    share = {r["run"]: r["dc_frac"] for r in load("type_structure_spectrum")}
    conc = {r["run"]: r["med_conc"] for r in load("index_conc_population")}
    runs = sorted(set(share) & set(conc))
    init = load("init_bit_baseline")

    fig, ax = plt.subplots(figsize=(5.2, 2.9))

    ax.scatter([r["dc_frac"] for r in init], [r["med_conc"] for r in init],
               s=6, color="0.66", lw=0, zorder=2)
    UNCLASSIFIED = {"d47_seed6", "d61_seed4"}
    circ = [r for r in runs if r not in UNCLASSIFIED]
    tri = [r for r in runs if r in UNCLASSIFIED]
    ax.scatter([share[r] for r in circ], [conc[r] for r in circ],
               s=32, facecolor="white", edgecolor="0.15", lw=0.9, zorder=3)
    ax.scatter([share[r] for r in tri], [conc[r] for r in tri], marker="^",
               s=34, facecolor="white", edgecolor="0.15", lw=0.9, zorder=3)

    ax.text(0.036, 0.135, "210 untrained models", fontsize=7, color="0.45",
            ha="left", va="center")

    ax.set_xlim(-0.006, 0.225)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("shared-direction strength")
    ax.set_ylabel("spectral concentration of the index")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out)
    print(f"  wrote {out}")
    lo = max(share[r] for r in runs if share[r] < 0.09)
    hi = min(share[r] for r in runs if share[r] > 0.09)
    ic = np.array([r["med_conc"] for r in init])
    idc = np.array([r["dc_frac"] for r in init])
    tc = np.array([conc[r] for r in runs])
    print(f"  trained concentration {tc.min():.3f}-{tc.max():.3f}; "
          f"untrained {ic.min():.3f}-{ic.max():.3f}")
    print(f"  untrained strength {idc.min():.4f}-{idc.max():.4f}; "
          f"gap in the trained population {lo:.3f} to {hi:.3f}")


if __name__ == "__main__":
    main()
