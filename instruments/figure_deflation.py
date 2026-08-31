"""Equal cost, different damage. (2026-08-28)

Ablating the type contrast and displacing the rows by the same amount without
touching it cost the same, break different things, and neither breaks the type
decision. Table 2 reports overall accuracy, which cannot separate those: an
ablation can cost eleven points by confusing the type, by confusing the index,
or by knocking the prediction off both.

So both accuracies, on the two axes of one panel, one point per run per
ablation. Distance from the corner is how much the ablation cost; direction from
the corner is what it cost.

  contrast ablated    on the diagonal -- index and type fall together
  whole projection    on top of it, which is itself a result
  matched control     against the top edge -- index falls, type does not

The bottom right stays empty. Right index and wrong type is what a broken type
decision looks like, and no ablation produces it.

Colour encodes the ablation, which is something we did to the model, not a class
inferred from the same numbers -- so unlike the fork figure it is not the
figure restating its own labelling.

Reads edit_error_profile.csv. Computes nothing new.

Run:  python -m instruments.figure_deflation
Writes paper/v2/figures/deflation.pdf.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from instruments.results_io import figure_path, load


# The two ablations that take the contrast out share a colour family; the one
# that leaves it intact does not.
COND = [("base", "none", "0.15", "o", 26),
        ("crude", "whole projection", "#6baed6", "^", 46),
        ("surgical", "type contrast", "#08519c", "o", 28),
        ("noise_only", "matched control", "#e6550d", "s", 28)]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
})


def main():
    out = figure_path("deflation")
    prof = defaultdict(list)
    for r in load("edit_error_profile"):
        if r["cls"] == "A":
            prof[r["edit"]].append(r)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0.6, 1.02], [0.6, 1.02], color="0.85", lw=0.8, zorder=1)

    for key, lab, col, mk, sz in COND:
        xs = [r["index"] for r in prof[key]]
        ys = [r["type"] for r in prof[key]]
        face = "none" if key == "crude" else col
        ax.scatter(xs, ys, s=sz, marker=mk, facecolor=face,
                   edgecolor=col if key == "crude" else "white",
                   lw=1.0 if key == "crude" else 0.6, alpha=0.9, zorder=3,
                   label=lab)

    ax.text(1.015, 0.655, "a broken type decision\nwould land here",
            fontsize=6.8, color="0.55", ha="right", va="bottom")

    ax.set_xlim(0.63, 1.02)
    ax.set_ylim(0.63, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("index accuracy after the ablation")
    ax.set_ylabel("type accuracy after the ablation")
    leg = ax.legend(loc="upper left", frameon=False, handletextpad=0.2,
                    labelspacing=0.3, borderaxespad=0.1,
                    title="ablation applied")
    leg.get_title().set_fontsize(7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out)
    print(f"  wrote {out}")
    for key, lab, _, _, _ in COND:
        x = np.array([r["index"] for r in prof[key]])
        y = np.array([r["type"] for r in prof[key]])
        print(f"  {lab:<34} index {x.mean():.3f} ({x.min():.3f}-{x.max():.3f})  "
              f"type {y.mean():.3f} ({y.min():.3f}-{y.max():.3f})")


if __name__ == "__main__":
    main()
