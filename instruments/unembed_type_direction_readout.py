"""Zero-parameter test of the shared type direction (2026-08-04).

unembed_type_geometry showed the unembedding's per-index type-difference vectors
D_c = U[r_c] - U[s_c] carry a large SHARED component in runs where output type is
linearly decodable from the MLP hidden layer, and none (at/below the ~1/n chance
level) in runs where it is not.

That was weights-only. This closes the loop to actual computation WITHOUT fitting
anything: take the shared direction

    t = mean_c ( U[r_c] - U[s_c] )        [read straight off the weights]

project each input's final residual (the exact vector fc_out consumes) onto t, and
ask whether that scalar alone tracks the answer's type. No probe, no training, no
free parameters -- AUC needs no threshold at all.

  class A (lin_out ~100) : AUC ~1.0  => the weight-derived direction IS the type
                           feature; the earlier probe was not doing the work.
  class B (lin_out ~50)  : AUC ~0.5  => nothing to read along any shared direction.

Reported alongside:
  acc@thr : accuracy of sign(<resid,t> - thr) at the best threshold (ONE fitted
            scalar, for interpretability only; AUC is the headline).
  model_acc : full-model accuracy on the same inputs (sanity).
  ffn_share : fraction of <resid,t> variance contributed by the FFN's write
            (vs the attention+embedding path) -- says WHICH component drives the
            projection in runs where it works.

SCOPE FIX (2026-08-13): same change as unembed_type_geometry -- this ran over a
23-run glob gated by that module's hardcoded LIN_OUT dict, which no longer exists.
It now iterates population.RUNS (35 runs) and reads lin_out from
results/lin_out_all_seeds.csv when available. auc() and best_acc() are imported by
several other instruments and are unchanged.

Run: python -m instruments.unembed_type_direction_readout
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.population import RUNS, ckpt as ckpt_path
from instruments.results_io import column, save
from instruments.unembed_type_geometry import load_unembed
from src.dihedral import multiply


def auc(pos, neg):
    """P(score higher in pos) via rank statistic; 0.5 = chance."""
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_acc(pos, neg):
    allv = np.sort(np.concatenate([pos, neg]))
    thr = (allv[:-1] + allv[1:]) / 2
    best = 0.0
    for t in thr[:: max(1, len(thr) // 400)]:
        a = ((pos > t).sum() + (neg <= t).sum()) / (len(pos) + len(neg))
        best = max(best, a, 1 - a)
    return float(best)


def main():
    try:
        lin_out = column("lin_out_all_seeds", "lin_out")
    except FileNotFoundError:
        lin_out = {}

    print("  Zero-parameter readout along t = mean_c(U[r_c] - U[s_c]).")
    print(f"  {'run':24} {'n':>3} {'cls':>3} {'lin_out':>7} | {'AUC':>6} "
          f"{'acc@thr':>8} {'ffn_share':>9} {'model_acc':>9}")
    rows = []
    for path, n, cls in RUNS:
        ck = ckpt_path(path)
        run = os.path.basename(path)
        if not os.path.exists(ck):
            continue
        m, tok = build(ck, n)
        U, _ = load_unembed(ck)
        t = torch.tensor(U[:n] - U[n:2 * n], dtype=torch.float32).mean(0)

        toks = [f"r{i}" for i in range(n)] + [f"s{i}" for i in range(n)]
        pairs = [(a, b) for a in toks for b in toks]
        prods = [multiply(a, b, n) for a, b in pairs]
        is_rot = np.array([p.startswith("r") for p in prods])
        target = torch.tensor([tok.token_to_id[p] for p in prods])

        bos = tok.token_to_id["<BOS>"]
        ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                            for a, b in pairs])

        grabbed = {}
        h = m.fc_out.register_forward_pre_hook(
            lambda mod, inp: grabbed.__setitem__("x", inp[0].detach()))
        with torch.no_grad():
            logits = m(ids)
        h.remove()
        resid = grabbed["x"][:, 2, :]
        model_acc = float((logits[:, 2, :].argmax(-1) == target).float().mean())

        score = (resid @ t).numpy()
        a = auc(score[is_rot], score[~is_rot])
        a = max(a, 1 - a)                      # direction-agnostic
        acc = best_acc(score[is_rot], score[~is_rot])

        # how much of the projection is written by the FFN vs the attn/embed path
        blk = m.transformer_blocks[0]
        with torch.no_grad():
            pre = blk.feed_forward(resid)       # FFN's write into the residual
        s_ffn = (pre @ t).numpy()
        ffn_share = float(np.var(s_ffn) / (np.var(score) + 1e-12))

        lo_v = lin_out.get(run)
        rows.append((run, n, cls, lo_v, a, acc, ffn_share, model_acc))
        lo_s = f"{lo_v:>6.0f}%" if lo_v is not None else f"{'--':>7}"
        print(f"  {run:24} {n:>3} {cls:>3} {lo_s} | {a:>6.3f} {acc:>8.3f} "
              f"{ffn_share:>9.2f} {model_acc:>9.3f}")

    save("unembed_type_direction_readout",
         "run,n,cls,lin_out,auc_t,acc_thr,ffn_share,model_acc",
         [(r[0], r[1], r[2], "" if r[3] is None else f"{r[3]:.4f}") +
          tuple(f"{v:.6f}" for v in r[4:]) for r in rows])

    print(f"\n  {'cls':>4} {'k':>3} {'AUC':>18} {'acc@thr':>18}")
    for c in ("A", "B", "I"):
        v = np.array([[r[4], r[5]] for r in rows if r[2] == c])
        if len(v):
            print(f"  {c:>4} {len(v):>3} {v[:,0].mean():>8.3f} "
                  f"[{v[:,0].min():.3f},{v[:,0].max():.3f}] "
                  f"{v[:,1].mean():>8.3f} [{v[:,1].min():.3f},{v[:,1].max():.3f}]")
    print("  READ: AUC~1 in class A only => the weight-derived shared direction IS "
          "the type feature there (no probe needed).")


if __name__ == "__main__":
    main()
