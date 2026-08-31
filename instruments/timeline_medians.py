"""The Timeline paragraph's numbers, from a stated rule. (2026-08-29)

app:lifecycle's Timeline paragraph quoted three kinds of median epoch
(readout saturation, orthogonal margin at 90%, grokking) whose generating
rule was never written down; the grok medians were verified against the
trajectory CSV, the other two were not, and an ad-hoc reconstruction gave
different values (8K and 21K/13K against the quoted 7K and 25K/17K). This
instrument defines the rules and becomes the paragraph's sole source.

All statistics read bit_emergence_trajectory.csv, per run:
  grok epoch     : first checkpoint with val >= 0.99.
  best-val epoch : the checkpoint with maximal val (earliest on ties) ---
                   the CSV-grid stand-in for the paper's measurement site.
  readout sat.   : first checkpoint from which auc_t stays >= 0.99 at every
                   later checkpoint in the file (class A only; class B's
                   readout never leaves chance).
  perp-90        : first checkpoint from which m_perp stays >= 0.9x its
                   best-val value at every checkpoint up to best-val.
Sustained crossings, not first touches, so a spike cannot set the date.
Classes from early_bit_ablation.csv; the two unclassified runs are excluded
from medians and reported per run.

Run: python -m instruments.timeline_medians
"""
from collections import defaultdict

import numpy as np

from instruments.results_io import load, save

UNCLASSIFIED = {"d47_seed6", "d61_seed4"}


def sustained(epochs, ok, upto=None):
    """First epoch from which ok holds at every later index (up to `upto`)."""
    last = len(ok) if upto is None else upto + 1
    for i in range(last):
        if all(ok[i:last]):
            return epochs[i]
    return None


def main():
    cls = {r["run"]: r["cls"] for r in load("early_bit_ablation")}
    traj = defaultdict(list)
    for r in load("bit_emergence_trajectory"):
        if r["run"] in cls and r["epoch"] > 0:
            traj[r["run"]].append(r)

    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for run, rs in sorted(traj.items()):
        rs.sort(key=lambda r: r["epoch"])
        ep = [r["epoch"] for r in rs]
        val = np.array([r["val"] for r in rs])
        auc = np.array([r["auc_t"] for r in rs])
        mp = np.array([r["m_perp"] for r in rs])
        grok = next((e for e, v in zip(ep, val) if v >= 0.99), None)
        bv = int(np.argmax(val))
        c = "I" if run in UNCLASSIFIED else cls[run]
        sat = (sustained(ep, auc >= 0.99) if c == "A" or run in UNCLASSIFIED
               else None)
        p90 = sustained(ep, mp >= 0.9 * mp[bv], upto=bv)
        rows.append((run, rs[0]["n"], c, grok, sat, p90, ep[bv]))
        if c in "AB":
            for k, v in (("grok", grok), ("sat", sat), ("p90", p90)):
                if v is not None:
                    agg[c][k].append(v)

    save("timeline_medians",
         "run,n,cls,grok_epoch,readout_sat_epoch,perp90_epoch,bestval_epoch",
         rows)
    for c in ("A", "B"):
        print(f"  class {c}: " + "  ".join(
            f"median {k} = {int(np.median(v))} (k={len(v)})"
            for k, v in agg[c].items()))
    for r in rows:
        if r[2] == "I":
            print(f"  unclassified {r[0]}: grok {r[3]}, sat {r[4]}, "
                  f"perp90 {r[5]}")


if __name__ == "__main__":
    main()
