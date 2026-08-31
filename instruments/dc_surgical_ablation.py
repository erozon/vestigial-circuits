"""Surgical removal of the type bit, correcting a flaw in the crude ablation.
(2026-08-05)

dc_component_ablation.py removed the ENTIRE direction t = mean_c(U[r_c] - U[s_c])
from every row of the output layer. That cost class A ~11% accuracy, but the
margin decomposition then showed the frequency-only margin against the true
token's type partner is negative on just 0.1% of products. So the 11% was not
type confusion.

The reason is that the crude ablation removed too much. Writing each row's
projection onto t as

    U[r_c] . t_hat = m_r + delta_c        U[s_c] . t_hat = m_s + eps_c

the TYPE BIT is the constant contrast (m_r - m_s). The per-row deviations
delta_c, eps_c vary with the index and carry index information; the crude
ablation destroyed those too, reshuffling rankings among tokens of the SAME type.
That is why 99% of its errors had both index and type wrong.

This removes only the constant contrast, leaving every per-row deviation intact:

    U'[r_c] = U[r_c] - (m_r - m) t_hat      U'[s_c] = U[s_c] - (m_s - m) t_hat
    where m = (m_r + m_s) / 2

Conditions reported:
  base     : untouched
  surgical : type-constant contrast removed, per-row deviations kept
  crude    : the whole direction removed (the earlier, flawed ablation)
  shuffled : the SAME per-row offsets as surgical, but assigned to rows in a
             random permutation that ignores type. Same magnitude of perturbation,
             no type structure -- if surgical hurts and shuffled does not, the
             damage is specifically about the type contrast.

Reading, stated before running: if class A survives the surgical ablation, its
explicit type bit is redundant -- the frequency mechanism it shares with class B
already decides type, and the bit is a second, unnecessary copy.

Run: python -m instruments.dc_surgical_ablation
"""
import os

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.dc_component_ablation import evaluate, truth
from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed


def unit_t(U, n):
    t = (U[:n] - U[n:2 * n]).mean(0)
    return t / (t.norm() + 1e-12)


def main():
    print("  Surgical vs crude removal of the type bit from the output layer.\n")
    print(f"  {'run':24} {'cls':>3} | {'base':>6} | {'surg':>6} {'s_idx':>6} "
          f"{'s_typ':>6} | {'crude':>6} | {'shuf':>6} {'noise':>6}")
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        pairs, _, _, _ = all_pairs_typed(n)
        ti, tt = truth(pairs, n)

        m0, tok = build(p, n)
        a0 = evaluate(m0, tok, pairs, n, ti, tt)[0]

        # surgical
        ms, _ = build(p, n)
        with torch.no_grad():
            U = ms.fc_out.weight
            th = unit_t(U, n)
            proj = U @ th
            mr, msf = proj[:n].mean(), proj[n:2 * n].mean()
            mid = (mr + msf) / 2
            off = torch.zeros_like(proj)
            off[:n] = mr - mid
            off[n:2 * n] = msf - mid
            U -= torch.outer(off, th)
        a1, i1, y1, _ = evaluate(ms, tok, pairs, n, ti, tt)

        # crude (whole direction)
        mc, _ = build(p, n)
        with torch.no_grad():
            U = mc.fc_out.weight
            thc = unit_t(U, n)
            U -= torch.outer(U @ thc, thc)
        a2 = evaluate(mc, tok, pairs, n, ti, tt)[0]

        # shuffled control: same offsets, type-blind assignment
        shuf = []
        for s in range(3):
            mh, _ = build(p, n)
            with torch.no_grad():
                U = mh.fc_out.weight
                thh = unit_t(U, n)
                pr = U @ thh
                mr2, ms2 = pr[:n].mean(), pr[n:2 * n].mean()
                mid2 = (mr2 + ms2) / 2
                o = torch.zeros_like(pr)
                o[:n] = mr2 - mid2
                o[n:2 * n] = ms2 - mid2
                g = torch.Generator().manual_seed(500 + s)
                perm = torch.randperm(2 * n, generator=g)
                o[:2 * n] = o[:2 * n][perm]
                U -= torch.outer(o, thh)
            shuf.append(evaluate(mh, tok, pairs, n, ti, tt)[0])
        a3 = float(np.mean(shuf))

        # noise_only: SAME per-row magnitude along t_hat, but balanced within each
        # type so the type contrast is left exactly intact. If this hurts as much
        # as surgical, the damage is generic sensitivity to perturbation along
        # t_hat and says nothing about the type bit being load-bearing.
        noi = []
        for s in range(3):
            mn, _ = build(p, n)
            with torch.no_grad():
                U = mn.fc_out.weight
                thn = unit_t(U, n)
                pr = U @ thn
                d = (pr[:n].mean() - pr[n:2 * n].mean()) / 2
                g = torch.Generator().manual_seed(900 + s)
                o = torch.zeros_like(pr)
                for lo, hi in ((0, n), (n, 2 * n)):
                    k = hi - lo
                    signs = torch.ones(k)
                    signs[torch.randperm(k, generator=g)[:k // 2]] = -1.0
                    signs -= signs.mean()          # exactly balanced within type
                    o[lo:hi] = d * signs
                U -= torch.outer(o, thn)
            noi.append(evaluate(mn, tok, pairs, n, ti, tt)[0])
        a4 = float(np.mean(noi))

        print(f"  {os.path.basename(path):24} {cls:>3} | {a0:>6.3f} | {a1:>6.3f} "
              f"{i1:>6.3f} {y1:>6.3f} | {a2:>6.3f} | {a3:>6.3f} {a4:>6.3f}")
        agg.setdefault(cls, []).append([a0, a1, i1, y1, a2, a3, a4])
        rows.append((os.path.basename(path), n, cls) +
                    tuple(f"{v:.6f}" for v in (a0, a1, i1, y1, a2, a3, a4)))
    print()
    save("dc_surgical_ablation",
         "run,n,cls,base,surgical,surg_idx,surg_typ,crude,shuffled,noise_only", rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): base {a[:,0].mean():.3f} | surgical "
              f"{a[:,1].mean():.3f} (idx {a[:,2].mean():.3f}, typ {a[:,3].mean():.3f}) "
              f"| crude {a[:,4].mean():.3f} | shuffled {a[:,5].mean():.3f} "
              f"| noise-only {a[:,6].mean():.3f}")
        print(f"           surgical worst run {a[:,1].min():.3f}")


if __name__ == "__main__":
    main()
