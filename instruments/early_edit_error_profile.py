"""Does the EARLY ablation cost look like a severed type decision? (2026-08-28)

Section 3.4 calls the contrast load-bearing at epoch 7,000 because removing it
costs 0.24 of type accuracy while the matched control costs 0.006. But that is
the same statistic that reads 0.115 vs 0.014 at convergence, where the paper
calls the contrast inert -- rescued there by edit_error_profile.py, which shows
the converged surgical errors are 99.4% wrong on BOTH axes (generic wreckage)
and only 0.5% right-index-wrong-type (the signature a broken type decision must
produce). The early claim has no such second measurement. This is it.

Same checkpoints as early_bit_ablation.py (epochs 3/5/7/9K, dense sweep), same
edits, but recording what the errors are made of. Because the base model is
still memorizing at these epochs (index accuracy ~ the train fraction), the
vs-truth composition is dominated by errors the base model already makes, so
the load-bearing question lives in the DELTA: over pairs whose type the base
model got right and the edit got wrong (typ_break), did the prediction move to
the type-partner of the base prediction -- same predicted index, flipped type
(partner_frac)? That is what cutting a type wire looks like. The same columns
are computed for the matched control.

Predictions, stated before running:
  - Load-bearing claim SURVIVES if early surgical breakage is large and lands
    on the type-partner (partner_frac high), while the control breaks little.
  - Load-bearing claim DIES if early surgical breakage looks like convergence:
    predictions scattered to unrelated tokens at rates a matched displacement
    also produces. Then "genuinely load-bearing before generalization" was the
    same shape artifact the paper warns about, and Section 3.4 must be recut.

Differences from early_bit_ablation.py, for anyone reconciling numbers: this
measures the full pair table rather than its 3,000-pair subsample, so base and
surgical type accuracies differ from early_bit_ablation.csv in the third
decimal.

Run: python -m instruments.early_edit_error_profile
"""
import glob
import os
from collections import defaultdict

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.early_bit_ablation import final_class, unit_t
from instruments.population import epoch_checkpoints, run_dir
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

EPOCHS = [3000, 5000, 7000, 9000]
NOISE_DRAWS = 3
UNCLASSIFIED = {"d47_seed6", "d61_seed4"}


def predict(m, tok, pairs, n):
    """Argmax over element tokens at the result position, one int per pair."""
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                        for a, b in pairs])
    with torch.no_grad():
        return m(ids)[:, 2, :2 * n].argmax(1).numpy()


def edit_model(ck, n, kind, seed=0):
    """A checkpoint with one Section-3.3 edit applied, magnitudes recomputed
    from that checkpoint's own output layer, exactly as early_bit_ablation."""
    m, tok = build(ck, n)
    if kind == "base":
        return m, tok
    with torch.no_grad():
        U = m.fc_out.weight
        th = unit_t(U, n)
        pr = U @ th
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
                sg -= sg.mean()
                o[lo:hi] = d * sg
        U -= torch.outer(o, th)
    return m, tok


def stats(pred, base_pred, ti, tt, n):
    """vs-truth accuracies and error composition, plus the vs-base delta:
    typ_break, typ_fix (fractions of all pairs) and partner_frac (of the broken
    pairs, how many moved to the type-partner of the base prediction)."""
    p_idx, p_typ = pred % n, (pred >= n).astype(int)
    ok_i, ok_t = p_idx == ti, p_typ == tt
    overall, index, typ = (ok_i & ok_t).mean(), ok_i.mean(), ok_t.mean()
    bad = ~(ok_i & ok_t)
    if bad.sum():
        typ_only = (ok_i & ~ok_t)[bad].mean()
        idx_only = (~ok_i & ok_t)[bad].mean()
        both = (~ok_i & ~ok_t)[bad].mean()
    else:
        typ_only = idx_only = both = 0.0
    if base_pred is None:
        return (overall, index, typ, typ_only, idx_only, both,
                None, None, None)
    b_typ_ok = ((base_pred >= n).astype(int)) == tt
    broke = b_typ_ok & ~ok_t
    fixed = ~b_typ_ok & ok_t
    partner = (float((p_idx[broke] == (base_pred[broke] % n)).mean())
               if broke.sum() else None)
    return (overall, index, typ, typ_only, idx_only, both,
            broke.mean(), fixed.mean(), partner)


def main():
    print("  The Section-3.3 edit pair at epochs 3/5/7/9K, errors decomposed.")
    print("  typ_break: base type right -> edited type wrong, over all pairs.")
    print("  partner_frac: of those, prediction = type-partner of the base "
          "prediction.\n")
    rows = []
    agg = defaultdict(lambda: defaultdict(list))
    for rd in sorted(d for d in glob.glob(run_dir("dense_checkpoint_sweep/d*"))
                     if os.path.isdir(d)):
        run = os.path.basename(rd)
        n = int(run.split("_")[0][1:])
        cls = final_class(rd, n)
        if cls is None:
            continue
        if run in UNCLASSIFIED:
            cls = "I"
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)

        epoch_checkpoints(rd)   # fails loudly if only final weights are present
        for ep in EPOCHS:
            ck = os.path.join(rd, "checkpoints", f"checkpoint_epoch_{ep}.pt")
            if not os.path.exists(ck):
                continue
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
                v = out[kind]
                agg[cls, kind][ep].append(v)
                rows.append((run, n, cls, ep, kind) +
                            tuple(None if x is None else f"{x:.6f}" for x in v))
        print(f"  measured {run} ({cls})", flush=True)

    save("early_edit_error_profile",
         "run,n,cls,epoch,edit,overall,index,type,err_type_only,"
         "err_index_only,err_both,typ_break,typ_fix,partner_frac", rows)

    for cls in ("A", "B", "I"):
        for kind in ("surgical", "noise_only"):
            if not agg[cls, kind]:
                continue
            print(f"\n  CLASS {cls}, {kind}")
            print(f"  {'epoch':>6} {'k':>3} {'type':>7} {'typ_break':>10} "
                  f"{'typ_fix':>8} {'partner_frac':>13}")
            for ep in EPOCHS:
                vs = agg[cls, kind].get(ep, [])
                if not vs:
                    continue
                col = lambda i: [v[i] for v in vs if v[i] is not None]
                pf = col(8)
                print(f"  {ep:>6} {len(vs):>3} {np.mean(col(2)):>7.3f} "
                      f"{np.mean(col(6)):>10.3f} {np.mean(col(7)):>8.3f} "
                      + (f"{np.mean(pf):>13.3f}" if pf else f"{'--':>13}"))


if __name__ == "__main__":
    main()
