"""What do the Section-3.3 edits actually break? (2026-08-28)

Table 2 compares the four output-layer edits on OVERALL accuracy alone, where
three of them land within two points of each other. Equal cost is not equal
effect: an edit can cost eleven points by confusing the type, by confusing the
index, or by knocking the prediction off both. Overall accuracy cannot tell
those apart, and the argument in that subsection turns on which it is.

This measures all three accuracies under each edit, plus the composition of the
errors -- right index and wrong type, wrong index and right type, or both wrong
-- on the sixteen class-A runs of the population of record, and on class B for
the comparison the appendix makes.

  base       : untouched
  crude      : the whole projection onto t_hat removed
  surgical   : the type-constant contrast removed, per-row deviations kept
  noise_only : balanced +/- offsets of the same magnitude, contrast left intact

The type-partner distinction is the one to watch. An edit that makes the model
choose the correct index with the wrong type is an edit that broke the type
decision. An edit whose errors are wrong in both coordinates broke something
else, and its cost in type accuracy is a byproduct.

Run: python -m instruments.edit_error_profile
"""
from collections import defaultdict

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import (error_profile,
                                                           evaluate, truth)
from instruments.early_bit_ablation import unit_t
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

DRAWS = 3
EDITS = ("base", "crude", "surgical", "noise_only")


def apply_edit(p, n, kind, seed):
    """A model with one edit applied. Magnitudes follow dc_surgical_ablation."""
    m, tok = build(p, n)
    if kind == "base":
        return m, tok
    with torch.no_grad():
        U = m.fc_out.weight
        th = unit_t(U, n)
        pr = U @ th
        if kind == "crude":
            U -= torch.outer(pr, th)           # the whole projection
            return m, tok
        d = (pr[:n].mean() - pr[n:2 * n].mean()) / 2
        o = torch.zeros_like(pr)
        if kind == "surgical":
            o[:n], o[n:2 * n] = d, -d
        else:
            g = torch.Generator().manual_seed(seed)
            for lo, hi in ((0, n), (n, 2 * n)):
                k = hi - lo
                sg = torch.ones(k)
                sg[torch.randperm(k, generator=g)[:k // 2]] = -1.0
                sg -= sg.mean()                # exactly balanced within type
                o[lo:hi] = d * sg
        U -= torch.outer(o, th)
    return m, tok


def measure(p, n, kind):
    """(overall, index, type, type-only, index-only, both), averaged over draws
    for the randomized edit and taken once for the deterministic ones."""
    draws = DRAWS if kind == "noise_only" else 1
    got = []
    for s in range(draws):
        m, tok = apply_edit(p, n, kind, 900 + s)
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)
        a, ai, at, (ok_i, ok_t) = evaluate(m, tok, pairs, n, ti, tt)
        got.append((a, ai, at) + error_profile(ok_i, ok_t))
    return tuple(np.mean(got, axis=0))


def main():
    print("  What each edit breaks. Errors split into right-index/wrong-type, "
          "wrong-index/right-type, and both wrong.\n")
    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for path, n, cls in RUNS:
        if cls not in ("A", "B"):
            continue
        for kind in EDITS:
            v = measure(ckpt(path), n, kind)
            agg[cls][kind].append(v)
            rows.append((path.split("/")[-1], n, cls, kind) +
                        tuple(f"{x:.6f}" for x in v))
        print(f"  measured {path.split('/')[-1]} ({cls})", flush=True)

    save("edit_error_profile",
         "run,n,cls,edit,overall,index,type,err_type_only,err_index_only,err_both",
         rows)

    for cls in ("A", "B"):
        print(f"\n  CLASS {cls}  (k = {len(agg[cls]['base'])})")
        print(f"  {'edit':<11} {'overall':>8} {'index':>7} {'type':>7} |"
              f" {'type-only':>10} {'idx-only':>9} {'both':>7}")
        for kind in EDITS:
            v = np.array(agg[cls][kind]).mean(0)
            print(f"  {kind:<11} {v[0]:>8.3f} {v[1]:>7.3f} {v[2]:>7.3f} |"
                  f" {v[3]:>10.3f} {v[4]:>9.3f} {v[5]:>7.3f}")


if __name__ == "__main__":
    main()
