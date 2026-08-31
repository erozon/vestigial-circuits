"""Built before it could multiply, still load-bearing at the end. (2026-08-28)

  (a) shared-direction strength against training time, with each run's clock
      normalized by its own grokking epoch. A run's x = 1 is the moment it
      generalizes, so "the direction is built before the model can multiply"
      is one vertical line rather than 25 scattered markers, and runs that
      grok at wildly different epochs become comparable. The axis is linear:
      nothing happens in the first tenth, so a log axis gave a third of the
      panel to a flat stretch and squeezed the plateau after grokking, which
      is where the direction is kept rather than removed.
  (b) type accuracy lost to each ablation, against validation accuracy rather
      than epoch, for the same reason: runs are compared at equal competence.
      The four early points come from matched_acc_bit_ablation.csv, the final
      one from converged_type_ablation.csv, which measures the same ablations
      the same way at each run's last checkpoint.

Both panels use type accuracy or the geometry that feeds it, never overall
accuracy, so nothing here is compared across units.

Class B is drawn faintly in (b): it has no shared direction to ablate, so both
of its series sit at zero. That is the control, not a finding.

The two runs the sign readout leaves unclassified (d47_seed6, d61_seed4) are in
no class statistic anywhere in the paper. Panel (a) draws them in grey; panel
(b) excludes them from both class means.

Run:  python -m instruments.figure_lifecycle
Writes paper/v2/figures/lifecycle.pdf.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from instruments.results_io import figure_path, load


GROK = 0.99
COL = {"A": "tab:blue", "B": "tab:orange", "I": "0.45"}
UNCLASSIFIED = {"d47_seed6", "d61_seed4"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
})


def main():
    out = figure_path("lifecycle")
    cls = {r["run"]: ("I" if r["run"] in UNCLASSIFIED else r["cls"])
           for r in load("early_bit_ablation")}
    traj = defaultdict(list)
    for r in load("bit_emergence_trajectory"):
        if r["run"] in cls and r["epoch"] > 0:
            traj[r["run"]].append(r)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                 gridspec_kw={"width_ratios": [1.1, 1],
                                              "wspace": 0.30})

    # (a) each run on its own clock: x = 1 is the epoch it generalizes
    seen = set()
    for run, rows in sorted(traj.items()):
        rows.sort(key=lambda r: r["epoch"])
        gk = [r for r in rows if r["val"] and r["val"] >= GROK]
        if not gk:
            continue
        g = gk[0]["epoch"]
        ep = np.array([r["epoch"] for r in rows]) / g
        sh = np.array([r["dc_frac"] for r in rows])
        k = cls[run]
        lbl = {"A": "class A", "B": "class B", "I": "between the modes"}[k]
        ax.plot(ep, sh, color=COL[k], lw=0.9, alpha=0.6, zorder=2,
                ls="--" if k == "I" else "-",
                label=lbl if k not in seen else None)
        seen.add(k)

    ax.axvline(1.0, color="0.35", lw=0.9, zorder=3)
    ax.text(1.0, 1.01, "generalizes", fontsize=7, color="0.35", ha="center",
            va="bottom", transform=ax.get_xaxis_transform())
    # Linear, not log. The first tenth of training is flat, so a log axis spent
    # a third of the panel on nothing and compressed the part that matters: the
    # rise before 1, and the plateau after it that is the "never removed" claim.
    ax.set_xlim(0, 3.2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("training time / this run's grokking epoch")
    ax.set_ylabel("shared-direction strength")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.5,
              borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) what ablating it costs, at matched competence
    early = defaultdict(lambda: defaultdict(list))
    for r in load("matched_acc_bit_ablation"):
        if r["run"] in UNCLASSIFIED:
            continue
        early[r["cls"]][r["target_val"]].append(
            (r["base"] - r["surgical"], r["base"] - r["noise_only"]))
    conv = defaultdict(list)
    for r in load("converged_type_ablation"):
        if r["run"] in UNCLASSIFIED:
            continue
        conv[r["cls"]].append((r["base"] - r["surgical"],
                               r["base"] - r["noise_only"]))

    for c in ("A", "B"):
        xs = sorted(early[c]) + [1.06]
        rem = [np.mean([v[0] for v in early[c][x]]) for x in sorted(early[c])]
        ctl = [np.mean([v[1] for v in early[c][x]]) for x in sorted(early[c])]
        rem.append(np.mean([v[0] for v in conv[c]]))
        ctl.append(np.mean([v[1] for v in conv[c]]))
        if c == "A":
            bx.plot(xs, rem, "-o", color=COL[c], ms=4, lw=1.6, zorder=3,
                    label="shared direction ablated")
            bx.plot(xs, ctl, "--o", color=COL[c], ms=4, lw=1.0, mfc="white",
                    zorder=3, label="matched control")
        else:
            bx.plot(xs, rem, "-", color="0.65", lw=1.0, zorder=2,
                    label="class B, either ablation")
            bx.plot(xs, ctl, "-", color="0.65", lw=1.0, zorder=2)

    bx.axhline(0, color="0.85", lw=0.8, zorder=1)
    bx.set_xlim(0.28, 1.16)
    bx.set_xticks([0.35, 0.5, 0.7, 0.9, 1.06])
    bx.set_xticklabels(["0.35", "0.50", "0.70", "0.90", "final"])
    bx.set_ylim(-0.02, 0.40)
    bx.set_xlabel("validation accuracy when measured")
    bx.set_ylabel("type accuracy lost to the ablation")
    bx.legend(loc="upper right", frameon=False, handletextpad=0.5,
              labelspacing=0.3)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.savefig(out)
    print(f"  wrote {out}")
    for c in ("A", "B"):
        xs = sorted(early[c])
        print(f"  class {c}  " + "  ".join(
            f"{x:.2f}:{np.mean([v[0] for v in early[c][x]]):.3f}" for x in xs)
            + f"  final:{np.mean([v[0] for v in conv[c]]):.3f}")


if __name__ == "__main__":
    main()
