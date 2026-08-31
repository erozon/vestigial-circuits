"""When does the redundant type bit appear, relative to generalization? (2026-08-05)

Established: class A models carry a near-constant type bit along one global
direction that contributes 19% of their type margin and is functionally
unnecessary -- the index-bound frequency mechanism, identical in both classes,
already decides type 99.9% of the time on its own.

Open: why class A builds it at all, and whether it matters DURING training even
though it does nothing at inference. Timing is the cheapest evidence available.

Over the dense checkpoint sweep (every 1000 epochs), per checkpoint:
  val      : validation accuracy at that epoch, from metrics.jsonl
  dc_frac  : ||mean_c D_c||^2 / mean_c||D_c||^2 of the output layer (weights only)
  auc_t    : zero-parameter readout along t, on a fixed subsample of products
  m_dc     : mean type margin contributed by the shared direction (normalized)
  m_perp   : mean type margin contributed by everything else (normalized)

Readings, stated before running:
  - bit appears AFTER generalization => a byproduct that accumulates once the
    task is already solved; nothing to do with reaching the solution.
  - bit appears BEFORE or WITH the frequency margin => it may be scaffolding the
    model used to get there and then kept.
  - m_perp reaching full strength before the bit exists in class A would show the
    frequency mechanism does not need the bit developmentally either.

Subsampling: 3000 products, fixed across checkpoints and runs of the same order so
trajectories are comparable.

SAMPLING FIX (2026-08-13): this took every 2nd checkpoint throughout, which meant
the interval where the bit forms was sampled at epochs 1000, 3000, 5000 -- two
points supporting the "forms at ~1-3K" claim. Checkpoints exist every 1000 epochs,
so the first EARLY_DENSE_TO epochs are now taken at full resolution and the stride
applies only after that, where the curves are flat and the extra points buy
nothing.

Run: python -m instruments.bit_emergence_trajectory
"""
import glob
import json
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.population import epoch_checkpoints, run_dir
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed
from instruments.unembed_type_direction_readout import auc

SUB = 3000
EARLY_DENSE_TO = 10000          # every checkpoint up to here, every 2nd after


def val_curve(rd):
    pts = []
    p = os.path.join(rd, "metrics.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("val_acc") is not None and r.get("epoch") is not None:
                pts.append((r["epoch"], r["val_acc"]))
    return sorted(pts)


def val_at(curve, ep):
    best = None
    for e, v in curve:
        if e <= ep:
            best = v
        else:
            break
    return best if best is not None else float("nan")


def measure(ckpt, n, sub_idx, pairs, ti, tt):
    m, tok = build(ckpt, n)
    U = m.fc_out.weight.detach().numpy()
    D = U[:n] - U[n:2 * n]
    t = D.mean(0)
    dc_frac = float(t @ t) / float((D * D).sum(1).mean())
    th = t / (np.linalg.norm(t) + 1e-12)

    sp = [pairs[i] for i in sub_idx]
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]] for a, b in sp])
    grabbed = {}
    h = m.fc_out.register_forward_pre_hook(
        lambda mod, inp: grabbed.__setitem__("x", inp[0].detach()))
    with torch.no_grad():
        logits = m(ids)[:, 2, :2 * n].numpy()
    h.remove()
    x = grabbed["x"][:, 2, :].numpy()

    sti, stt = ti[sub_idx], tt[sub_idx]
    score = x @ th
    a = auc(score[stt == 0], score[stt == 1])
    a = max(a, 1 - a)

    sign = np.where(stt == 0, 1.0, -1.0)
    Dc = D[sti]
    margin = sign * (x * Dc).sum(1)
    dc = sign * (x @ th) * (Dc @ th)
    scale = logits.std(1) + 1e-12
    return dc_frac, float(a), float((dc / scale).mean()), \
        float(((margin - dc) / scale).mean())


def main():
    runs = sorted(r for r in glob.glob(run_dir("dense_checkpoint_sweep/d*"))
                  if os.path.isdir(r))
    rows = []
    cache = {}
    for rd in runs:
        n = int(os.path.basename(rd).split("_")[0][1:])
        if n not in cache:
            pairs, _, _, _ = all_pairs_typed(n)
            ti, tt = truth(pairs, n)
            rng = np.random.default_rng(0)
            cache[n] = (pairs, ti, tt, rng.choice(len(pairs), SUB, replace=False))
        pairs, ti, tt, sub_idx = cache[n]
        curve = val_curve(rd)

        def ep_of(s):
            return int(s.split("_")[-1].split(".")[0])

        cks = epoch_checkpoints(rd)
        early = [c for c in cks if ep_of(c) <= EARLY_DENSE_TO]
        late = [c for c in cks if ep_of(c) > EARLY_DENSE_TO][::2]
        cks = early + late
        final = os.path.join(rd, "checkpoints", "best_model.pt")
        for ck in cks + ([final] if os.path.exists(final) else []):
            ep = (int(ck.split("_")[-1].split(".")[0])
                  if "checkpoint_epoch_" in ck else -1)
            try:
                dcf, a, mdc, mpe = measure(ck, n, sub_idx, pairs, ti, tt)
            except Exception as e:                       # noqa: BLE001
                print(f"  skip {ck}: {e}")
                continue
            rows.append((os.path.basename(rd), n, ep, val_at(curve, ep) if ep > 0
                         else (curve[-1][1] if curve else float("nan")),
                         dcf, a, mdc, mpe))
        print(f"  done {os.path.basename(rd)} ({len(cks)} checkpoints)", flush=True)

    print()
    save("bit_emergence_trajectory", "run,n,epoch,val,dc_frac,auc_t,m_dc,m_perp",
         rows)


if __name__ == "__main__":
    main()
