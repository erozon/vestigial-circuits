"""Task-unit verdict at the stage-confound checkpoints. (2026-08-28)

early_edit_error_profile.py established task-accuracy parity between the
surgical edit and the matched control at epochs 3/5/7/9K, and Table 2
establishes it at convergence. Between those sites --- the accuracy-matched
checkpoints of matched_acc_bit_ablation.py, val 0.35 to 0.90, median class-A
epochs 16K-29K --- only type units were ever scored, and there the removal
costs up to 0.35. Whether "the task never depends on the contrast" is true
hinges on this window: at val 0.90 the index is mostly right, so a 0.23
type-accuracy cost could surface in task units.

Prediction, stated before running: parity should WEAKEN as validation accuracy
rises, because the pairs whose type the contrast carries increasingly have
correct indices. If surgical task cost exceeds the control's at high val, the
contrast has a genuinely task-load-bearing window and the paper must say
"needed transiently", not "never needed".

Same edits, same stats as early_edit_error_profile.py; checkpoints taken from
matched_acc_bit_ablation.csv (a run enters a cell only if a checkpoint landed
within tolerance of the target, so cell counts vary as in that instrument).

Run: python -m instruments.matched_edit_error_profile
"""
import csv
import os
from collections import defaultdict

import numpy as np

from instruments.dc_component_ablation import truth
from instruments.early_edit_error_profile import (NOISE_DRAWS,
                                                              UNCLASSIFIED,
                                                              edit_model,
                                                              predict, stats)
from instruments.population import run_dir
from instruments.results_io import path, save
from instruments.type_xor_probe import all_pairs_typed

def main():
    print("  The Section-3.3 edit pair at the accuracy-matched checkpoints, "
          "scored on the full product.\n")
    with open(path("matched_acc_bit_ablation")) as f:
        cells = [(r["run"], r["cls"], float(r["target_val"]), int(r["epoch"]))
                 for r in csv.DictReader(f)]
    rows = []
    agg = defaultdict(lambda: defaultdict(list))
    pair_cache = {}
    for run, cls, tv, ep in cells:
        if run in UNCLASSIFIED:
            cls = "I"
        n = int(run.split("_")[0][1:])
        if n not in pair_cache:
            pairs, _, _, _ = all_pairs_typed(n)
            pair_cache[n] = (pairs,) + truth(pairs, n)
        pairs, ti, tt = pair_cache[n]
        ck = os.path.join(run_dir(f"dense_checkpoint_sweep/{run}"),
                          "checkpoints", f"checkpoint_epoch_{ep}.pt")
        m, tok = edit_model(ck, n, "base")
        bp = predict(m, tok, pairs, n)
        out = {"base": stats(bp, None, ti, tt, n)}
        m, _ = edit_model(ck, n, "surgical")
        out["surgical"] = stats(predict(m, tok, pairs, n), bp, ti, tt, n)
        draws = []
        for s in range(NOISE_DRAWS):
            m, _ = edit_model(ck, n, "noise_only", seed=11 + s)
            draws.append(stats(predict(m, tok, pairs, n), bp, ti, tt, n))
        out["noise_only"] = tuple(
            (None if any(d[i] is None for d in draws)
             else float(np.mean([d[i] for d in draws])))
            for i in range(9))
        for kind in ("base", "surgical", "noise_only"):
            agg[cls, kind][tv].append(out[kind])
            rows.append((run, n, cls, tv, ep, kind) +
                        tuple(None if x is None else f"{x:.6f}"
                              for x in out[kind]))
        print(f"  measured {run} ({cls}) val {tv} epoch {ep}", flush=True)

    save("matched_edit_error_profile",
         "run,n,cls,target_val,epoch,edit,overall,index,type,err_type_only,"
         "err_index_only,err_both,typ_break,typ_fix,partner_frac", rows)

    for cls in ("A", "B", "I"):
        if not agg[cls, "base"]:
            continue
        print(f"\n  CLASS {cls} --- task-accuracy COST vs base (type cost in "
              "parens)")
        print(f"  {'val':>5} {'k':>3} {'base':>6} {'surgical':>15} "
              f"{'control':>15}")
        for tv in sorted({t for (c, k), d in agg.items() if c == cls
                          for t in d}):
            b = agg[cls, "base"][tv]
            s = agg[cls, "surgical"][tv]
            c = agg[cls, "noise_only"][tv]
            bo = np.mean([v[0] for v in b])
            so = np.mean([bi[0] - si[0] for bi, si in zip(b, s)])
            co = np.mean([bi[0] - ci[0] for bi, ci in zip(b, c)])
            st = np.mean([bi[2] - si[2] for bi, si in zip(b, s)])
            ct = np.mean([bi[2] - ci[2] for bi, ci in zip(b, c)])
            print(f"  {tv:>5.2f} {len(b):>3} {bo:>6.3f} "
                  f"{so:>7.3f} ({st:>5.3f}) {co:>7.3f} ({ct:>5.3f})")


if __name__ == "__main__":
    main()
