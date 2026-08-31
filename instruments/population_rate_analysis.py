"""How should the per-order rate variation be reported? (2026-08-05)

The class A rate varies a lot by group order -- D_59 4/4, D_47 6/8, D_63 2/10 --
but several orders contribute only two runs, so eyeballing the table invites
over-reading. This decides what can honestly be said, rather than leaving the
question to prose.

Three things, in the order a reviewer would ask them:

1. RATES WITH UNCERTAINTY. Wilson score intervals per order. With two runs the
   interval is nearly the whole unit interval, which is the point.

2. IS THE VARIATION REAL? Permutation test. Statistic = Pearson chi-square of the
   order-by-class table; the null shuffles class labels across all runs, which
   preserves both the per-order run counts and the overall class balance. Reported
   two ways: excluding the two intermediate runs, and counting them as non-A, so
   the answer does not hinge on how they are binned.

3. IS IT REALLY ABOUT GROUP ORDER, OR ABOUT TIME? Class A runs grok later, and the
   orders differ in how long they take to grok, so "order" and "slow grokking" are
   confounded. Within each order (which holds group size fixed), report the AUC of
   grok epoch predicting class. Consistent within-order AUC above 0.5 would say the
   association is with training time, not with the group itself. Uses the dense
   sweep, the only runs with per-epoch checkpoints.

Run: python -m instruments.population_rate_analysis
"""
import csv
import os
from collections import defaultdict

import numpy as np

from instruments.population import RUNS, TRAINED_PER_ORDER
from instruments.results_io import path as result_path, save

TRAJ = result_path("bit_emergence_trajectory")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def chi2(table):
    t = np.asarray(table, dtype=float)
    row, col = t.sum(1, keepdims=True), t.sum(0, keepdims=True)
    exp = row @ col / t.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(exp > 0, (t - exp) ** 2 / exp, 0.0)
    return float(term.sum())


def perm_test(orders, is_a, iters=20000, seed=0):
    rng = np.random.default_rng(seed)
    uo = sorted(set(orders))

    def stat(labels):
        tab = [[sum(1 for o, l in zip(orders, labels) if o == u and l),
                sum(1 for o, l in zip(orders, labels) if o == u and not l)]
               for u in uo]
        return chi2(tab)

    obs = stat(is_a)
    arr = np.array(is_a)
    null = np.array([stat(rng.permutation(arr)) for _ in range(iters)])
    return obs, float((null >= obs).mean())


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    y = y[o]
    r = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    per = defaultdict(lambda: {"A": 0, "B": 0, "I": 0})
    for _, n, c in RUNS:
        per[n][c] += 1

    print("  1. CLASS A RATE BY GROUP ORDER, with 95% Wilson intervals\n")
    print(f"  {'group':>7} {'trained':>8} {'gen':>4} {'A':>3} {'B':>3} {'I':>3} "
          f"{'rate':>6} {'95% interval':>16}")
    rate_rows = []
    for n in sorted(TRAINED_PER_ORDER):
        d = per.get(n, {"A": 0, "B": 0, "I": 0})
        g = sum(d.values())
        if g:
            lo_w, hi_w = wilson(d["A"], g)
            rate_rows.append((n, TRAINED_PER_ORDER[n], g, d["A"], d["B"], d["I"],
                              f"{d['A'] / g:.6f}", f"{lo_w:.6f}", f"{hi_w:.6f}"))
        else:
            rate_rows.append((n, TRAINED_PER_ORDER[n], 0, "", "", "", "", "", ""))
        if g == 0:
            print(f"  {'D_' + str(n):>7} {TRAINED_PER_ORDER[n]:>8} {0:>4} "
                  f"{'-':>3} {'-':>3} {'-':>3} {'-':>6} {'(none generalized)':>16}")
            continue
        lo, hi = wilson(d["A"], g)
        print(f"  {'D_' + str(n):>7} {TRAINED_PER_ORDER[n]:>8} {g:>4} {d['A']:>3} "
              f"{d['B']:>3} {d['I']:>3} {d['A'] / g:>6.2f} "
              f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>16}")

    save("population_rate_by_order",
         "n,trained,generalized,A,B,I,rate_A,wilson_lo,wilson_hi", rate_rows)

    print("\n  2. IS THE VARIATION MORE THAN CHANCE? permutation test, 20000 draws")
    for label, keep_i in (("intermediates excluded", False),
                          ("intermediates counted as non-A", True)):
        orders = [n for _, n, c in RUNS if keep_i or c != "I"]
        is_a = [c == "A" for _, n, c in RUNS if keep_i or c != "I"]
        obs, p = perm_test(orders, is_a)
        print(f"    {label:34} chi2 = {obs:5.2f}, p = {p:.3f}  (k = {len(is_a)})")

    print("\n  3. ORDER, OR TIME TO GROK? within-order AUC of grok epoch -> class")
    if not os.path.exists(TRAJ):
        print("    (trajectory CSV missing; run bit_emergence_trajectory.py first)")
        return
    rows = list(csv.DictReader(open(TRAJ)))
    val = defaultdict(dict)
    fin = {}
    for r in rows:
        e = int(r["epoch"])
        v = float(r["val"]) if r["val"] not in ("nan", "") else np.nan
        if e == -1:
            fin[r["run"]] = (float(r["auc_t"]), v)
        else:
            val[r["run"]][e] = v
    grok = defaultdict(lambda: defaultdict(list))
    for run, (a, v) in fin.items():
        if not (v >= 0.99):
            continue
        n = int(run.split("_")[0][1:])
        eps = sorted(val[run])
        hit = [e for e in eps if val[run][e] >= 0.99]
        if not hit:
            continue
        grok[n]["A" if a >= 0.9 else "B"].append(hit[0])
    print(f"    {'group':>7} {'nA':>3} {'nB':>3} {'median A':>9} {'median B':>9} "
          f"{'AUC':>6}")
    aucs, grok_rows = [], []
    for n in sorted(grok):
        A, B = np.array(grok[n]["A"]), np.array(grok[n]["B"])
        u = auc(A, B)
        aucs.append(u)
        print(f"    {'D_' + str(n):>7} {len(A):>3} {len(B):>3} "
              f"{np.median(A) if len(A) else float('nan'):>9.0f} "
              f"{np.median(B) if len(B) else float('nan'):>9.0f} {u:>6.2f}")
        grok_rows.append((n, len(A), len(B),
                          f"{np.median(A):.0f}" if len(A) else "",
                          f"{np.median(B):.0f}" if len(B) else "",
                          f"{u:.6f}" if u == u else ""))
    print(f"    mean within-order AUC = {np.nanmean(aucs):.2f} "
          f"(0.5 = grok time says nothing about class)")
    print()
    save("population_grok_time_by_order",
         "n,k_A,k_B,median_grok_A,median_grok_B,auc", grok_rows)


if __name__ == "__main__":
    main()
