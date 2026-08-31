"""Native D_63 base rate under the SHARP metrics (2026-08-04).

Control group for the unembedding-transplant experiment (PLAN.md Experiment 4).
The transplants are 40K-horizon D_63 runs scored by lin_out + the zero-parameter
readout AUC. Their baseline must be measured the SAME way on the SAME kind of run,
or the comparison is unit-mismatched.

The dense checkpoint sweep's ten D_63 seeds are the right control: same group
order, same frozen recipe, same 40K horizon, all ten grokked. They were previously
labelled only by the COARSE override metric (which gave 1 localized / 9 not) --
that number is NOT in the same units as the transplant readout and must not be
used as its baseline.

Per seed:
  best_val  : best val accuracy (grokked iff >= 0.99)
  lin_out   : linear decode of output type from the MLP hidden layer
  auc_t     : zero-parameter readout along t = mean_c(U[r_c]-U[s_c]) of that run's
              OWN unembedding (nothing fitted)
  common    : ||mean_c D_c||^2 / mean_c||D_c||^2 of its own unembedding
  override  : the coarse metric, for cross-reference with the older labelling

Run: python -m instruments.native_d63_base_rate
"""
import glob
import json
import os

import numpy as np
import torch

from instruments.population import run_dir
from instruments.das_type_subspace import act_resid, build
from instruments.prime_composite_analyze import override_score
from instruments.type_xor_probe import all_pairs_typed, lin_probe, split
from instruments.unembed_type_direction_readout import auc
from instruments.unembed_type_geometry import geometry, load_unembed

N = 63
GROK_ACC = 0.99


def best_val(rd):
    path = os.path.join(rd, "metrics.jsonl")
    best = 0.0
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("val_acc") is not None:
                best = max(best, r["val_acc"])
    return best


def main():
    runs = sorted(glob.glob(run_dir("dense_checkpoint_sweep/d63_seed*")))
    print("  Native D_63 (dense sweep, 40K horizon) under the transplant metrics.\n")
    print(f"  {'seed':6} {'best_val':>9} {'lin_out':>8} {'auc_t':>7} {'common':>8} "
          f"{'override':>9}")
    rows = []
    for rd in runs:
        ck = os.path.join(rd, "checkpoints", "best_model.pt")
        if not os.path.exists(ck):
            continue
        seed = os.path.basename(rd).replace("d63_seed", "")
        bv = best_val(rd)
        m, tok = build(ck, N)
        pairs, _, _, out_type = all_pairs_typed(N)
        act, _ = act_resid(m, tok, pairs)
        tr, te = split(len(pairs))
        lin = lin_probe(act, out_type, tr, te)

        U, _ = load_unembed(ck)
        common = geometry(U, None, N)[0]
        t = torch.tensor(U[:N] - U[N:2 * N], dtype=torch.float32).mean(0)
        grabbed = {}
        bos = tok.token_to_id["<BOS>"]
        ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                            for a, b in pairs])
        h = m.fc_out.register_forward_pre_hook(
            lambda mod, inp: grabbed.__setitem__("x", inp[0].detach()))
        with torch.no_grad():
            m(ids)
        h.remove()
        score = (grabbed["x"][:, 2, :] @ t).numpy()
        is_rot = out_type == 0
        a = auc(score[is_rot], score[~is_rot])
        a = max(a, 1 - a)
        _base, ovr, _accq = override_score(ck, N)   # returns (base, score, per-quadrant)

        rows.append((seed, bv, lin, a, common, ovr))
        print(f"  {seed:6} {bv:>9.4f} {lin:>7.0f}% {a:>7.3f} {common:>8.3f} "
              f"{ovr:>9.1f}")

    grokked = [r for r in rows if r[1] >= GROK_ACC]
    mat = [r for r in grokked if r[3] >= 0.9]
    print(f"\n  {len(grokked)}/{len(rows)} grokked; "
          f"{len(mat)}/{len(grokked)} materialized a type variable (auc_t >= 0.9)")
    print(f"  => NATIVE D_63 BASE RATE under the transplant metric: "
          f"{len(mat)}/{len(grokked)}")
    print("  (the older '1 localized / 9' figure was the COARSE override metric; "
          "not comparable to transplant auc_t.)")


if __name__ == "__main__":
    main()
