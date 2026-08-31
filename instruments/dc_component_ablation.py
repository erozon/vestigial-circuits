"""Is the index-independent type direction load-bearing, or redundant? (2026-08-05)

Established so far (weights only, all 35 runs):
  - The output layer's type-discriminating vectors D_c = U[r_c] - U[s_c] carry
    ~72% of their energy in three nonzero frequencies of the index, in BOTH
    classes, and those are the same frequencies the index circuit uses
    (overlap 2.7/3 vs chance 0.3; spectrum correlation 0.92-0.95).
  - The classes differ at k=0 and essentially nowhere else: class A 0.155 of
    total energy, class B 0.015 against a chance level of ~0.017.

So class A has an index-independent type direction and class B has none. This
instrument asks what class A's costs it to lose.

Ablation: t = mean_c D_c, normalized. Remove that direction from EVERY row of the
output layer, U' = U - (U t_hat) t_hat^T. This destroys the index-independent
type discriminant while leaving all nonzero-frequency structure intact. Nothing is
retrained; the rest of the network is untouched.

Reported before and after, over all n^2 products:
  acc   : exact token accuracy
  idx   : predicted index == true index (ignoring type)
  typ   : predicted type == true type (ignoring index)

Reading, stated before running:
  - class A accuracy survives => the index-independent direction is a redundant
    copy of a distinction the frequency structure already makes.
  - class A type accuracy collapses while index accuracy survives => that
    direction is what class A actually uses to decide type, and class B must do
    something else.
Class B runs are the control: removing a direction that carries ~nothing should
change ~nothing.

Run: python -m instruments.dc_component_ablation
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

from src.dihedral import multiply


def truth(pairs, n):
    """True product token index in the 0..2n-1 element ordering (r0..r{n-1}, s0..)."""
    idx, typ = [], []
    for a, b in pairs:
        p = multiply(a, b, n)
        typ.append(0 if p[0] == "r" else 1)
        idx.append(int(p[1:]))
    return np.array(idx), np.array(typ)


def evaluate(m, tok, pairs, n, true_idx, true_typ):
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]]
                        for a, b in pairs])
    with torch.no_grad():
        logits = m(ids)[:, 2, :2 * n]          # element tokens only
        pred = logits.argmax(1).numpy()
    p_idx, p_typ = pred % n, (pred >= n).astype(int)
    ok_i, ok_t = p_idx == true_idx, p_typ == true_typ
    return (ok_i & ok_t).mean(), ok_i.mean(), ok_t.mean(), (ok_i, ok_t)


def error_profile(ok_i, ok_t):
    """Of the wrong predictions, how are they wrong?"""
    bad = ~(ok_i & ok_t)
    if bad.sum() == 0:
        return 0.0, 0.0, 0.0
    typ_only = (ok_i & ~ok_t)[bad].mean()     # right index, wrong type
    idx_only = (~ok_i & ok_t)[bad].mean()     # wrong index, right type
    both = (~ok_i & ~ok_t)[bad].mean()
    return typ_only, idx_only, both


def ablate_dir(m, v):
    """Remove unit direction v from every row of the output layer, in place."""
    with torch.no_grad():
        U = m.fc_out.weight
        U -= torch.outer(U @ v, v)


def main():
    print("  Removing the index-independent type direction from the output layer.")
    print("  rand = same ablation along a random unit direction, mean of 3 draws.")
    print("  Error profile is for the t-ablation: of the wrong predictions, the "
          "fraction that are\n  right-index/wrong-type, wrong-index/right-type, "
          "and both-wrong.\n")
    print(f"  {'run':26} {'cls':>3} | {'acc':>6} | {'acc_t':>6} {'idx_t':>6} "
          f"{'typ_t':>6} | {'acc_rnd':>7} | {'typ!':>5} {'idx!':>5} {'both!':>5}")
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)

        m, tok = build(p, n)
        a0, i0, y0, _ = evaluate(m, tok, pairs, n, ti, tt)

        with torch.no_grad():
            U = m.fc_out.weight
            t = (U[:n] - U[n:2 * n]).mean(0)
            t = t / (t.norm() + 1e-12)
        ablate_dir(m, t)
        a1, i1, y1, masks = evaluate(m, tok, pairs, n, ti, tt)
        ty, ix, bo = error_profile(*masks)

        rnd = []
        for s in range(3):
            mr, _ = build(p, n)                      # fresh copy each draw
            g = torch.Generator().manual_seed(1000 + s)
            v = torch.randn(mr.fc_out.weight.shape[1], generator=g)
            ablate_dir(mr, v / v.norm())
            rnd.append(evaluate(mr, tok, pairs, n, ti, tt)[0])
        ar = float(np.mean(rnd))

        print(f"  {os.path.basename(path):26} {cls:>3} | {a0:>6.3f} | {a1:>6.3f} "
              f"{i1:>6.3f} {y1:>6.3f} | {ar:>7.3f} | {ty:>5.2f} {ix:>5.2f} {bo:>5.2f}")
        agg.setdefault(cls, []).append((a0, a1, i1, y1, ar, ty, ix, bo))
        rows.append((os.path.basename(path), n, cls) +
                    tuple(f"{v:.6f}" for v in (a0, a1, i1, y1, ar, ty, ix, bo)))
    print()
    save("dc_component_ablation",
         "run,n,cls,base,acc_t,idx_t,typ_t,acc_rand,err_typ_only,err_idx_only,"
         "err_both", rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): acc {a[:,0].mean():.3f} -> {a[:,1].mean():.3f} "
              f"under t-ablation, -> {a[:,4].mean():.3f} under a random direction; "
              f"after t-ablation idx {a[:,2].mean():.3f} typ {a[:,3].mean():.3f}")
        print(f"           error profile: wrong-type-only {a[:,5].mean():.2f}, "
              f"wrong-index-only {a[:,6].mean():.2f}, both {a[:,7].mean():.2f}")


if __name__ == "__main__":
    main()
