"""How is the type margin split between the shared direction and the frequencies?
(2026-08-05)

Chain so far. The output layer's type-discriminating vectors D_c = U[r_c] - U[s_c]
are strongly structured in BOTH classes and sit on the same frequencies as the
index circuit. Class A additionally has an index-independent component t =
mean_c D_c (~15% of energy vs class B's 1.5%, which is chance). Removing t costs
class A ~11% accuracy, but the errors are near-ties -- the true token is still
ranked 2nd in 86% of them -- so t supplies MARGIN rather than the distinction
itself. Class B does the same job with no such direction at all.

Open question this addresses: does class A's frequency-based type discrimination
carry less margin in absolute terms than class B's, with t making up the
difference? If so the fork is about how models SPLIT one job, not about one
having a variable the other lacks.

There is no bias on the output layer, so logits are exactly x @ U^T and the
decomposition is exact. For each product, with c the true index and the partner
being the same index of the other type:

  margin = logit(true) - logit(partner) = s * (x . D_c),   s = +1 if true is a
                                                           rotation else -1
  dc_part   = s * (x . t_hat)(D_c . t_hat)
  perp_part = margin - dc_part

Skeptical controls, all reported:
  - Margins are normalized per example by the standard deviation of that example's
    2n element logits, so models with different logit scales are comparable. Raw
    means are printed too.
  - rand_share: the same decomposition along a RANDOM unit direction instead of t.
    Any direction captures some margin by chance; this is that floor.
  - dc_cv: coefficient of variation of (x . t_hat) within an output type. If t's
    contribution barely varies with the input, it is acting as a per-type bias
    rather than as a computed quantity.
  - Everything split by output type, because the ablation errors were concentrated
    on rotation outputs (16% vs 6%).

Run: python -m instruments.type_margin_decomposition
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed


def resid_and_logits(m, tok, pairs, n):
    grabbed = {}
    h = m.fc_out.register_forward_pre_hook(
        lambda mod, inp: grabbed.__setitem__("x", inp[0].detach()))
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                        for a, b in pairs])
    with torch.no_grad():
        logits = m(ids)[:, 2, :2 * n]
    h.remove()
    return grabbed["x"][:, 2, :].numpy(), logits.numpy()


def main():
    print("  Type margin split between the index-independent direction and the rest.")
    print("  Margins normalized per example by the sd of that example's element "
          "logits.\n")
    print(f"  {'run':24} {'cls':>3} | {'margin':>7} {'dc':>6} {'perp':>6} "
          f"{'dc_sh':>6} {'rnd_sh':>6} {'dc_cv':>6} | {'perp_rot':>8} {'perp_ref':>8} | {'neg':>6} {'p05':>6}")
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)
        m, tok = build(p, n)
        x, logits = resid_and_logits(m, tok, pairs, n)

        U = m.fc_out.weight.detach().numpy()
        D = U[:n] - U[n:2 * n]
        t = D.mean(0)
        th = t / (np.linalg.norm(t) + 1e-12)

        sign = np.where(tt == 0, 1.0, -1.0)          # +1 when the true token is r_c
        Dc = D[ti]                                   # (N, d)
        margin = sign * (x * Dc).sum(1)
        xt = x @ th
        dc = sign * xt * (Dc @ th)
        perp = margin - dc

        scale = logits.std(1) + 1e-12
        mg, dcn, pen = margin / scale, dc / scale, perp / scale
        dc_share = dc.sum() / (margin.sum() + 1e-12)

        rng = np.random.default_rng(7)
        shares = []
        for _ in range(3):
            v = rng.normal(size=x.shape[1])
            v /= np.linalg.norm(v)
            shares.append((sign * (x @ v) * (Dc @ v)).sum() / (margin.sum() + 1e-12))
        rnd_share = float(np.mean(shares))

        rot, ref = tt == 0, tt == 1
        cv = float(np.mean([np.std(xt[rot]) / (abs(np.mean(xt[rot])) + 1e-12),
                            np.std(xt[ref]) / (abs(np.mean(xt[ref])) + 1e-12)]))

        # Equal MEAN margins would not explain unequal ablation damage; the tail is
        # what decides argmax outcomes. neg = share of products where the frequency
        # structure alone puts the type partner AHEAD of the true token.
        neg = float((pen < 0).mean())
        p05 = float(np.percentile(pen, 5))

        print(f"  {os.path.basename(path):24} {cls:>3} | {mg.mean():>7.2f} "
              f"{dcn.mean():>6.2f} {pen.mean():>6.2f} {dc_share:>6.2f} "
              f"{rnd_share:>6.3f} {cv:>6.2f} | {pen[rot].mean():>8.2f} "
              f"{pen[ref].mean():>8.2f} | {neg:>6.3f} {p05:>6.2f}")
        agg.setdefault(cls, []).append(
            [mg.mean(), dcn.mean(), pen.mean(), dc_share, rnd_share, cv,
             pen[rot].mean(), pen[ref].mean(), neg, p05])
        rows.append((os.path.basename(path), n, cls) + tuple(
            f"{v:.6f}" for v in (mg.mean(), dcn.mean(), pen.mean(), dc_share,
                                 rnd_share, cv, pen[rot].mean(), pen[ref].mean(),
                                 neg, p05)))
    print()
    save("type_margin_decomposition",
         "run,n,cls,margin,dc,perp,dc_share,rand_share,dc_cv,perp_rot,perp_ref,"
         "neg_frac,perp_p05", rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): total margin {a[:,0].mean():.2f}, "
              f"of which dc {a[:,1].mean():.2f} and perp {a[:,2].mean():.2f}")
        print(f"           dc share {a[:,3].mean():.2f} (random direction "
              f"{a[:,4].mean():.3f}); dc_cv {a[:,5].mean():.2f}; "
              f"perp margin rot {a[:,6].mean():.2f} vs refl {a[:,7].mean():.2f}")
        print(f"           frequency-only margin NEGATIVE on {a[:,8].mean():.3f} of "
              f"products [{a[:,8].min():.3f}, {a[:,8].max():.3f}]; 5th pct {a[:,9].mean():.2f}")


if __name__ == "__main__":
    main()
