"""Was the bit load-bearing early, or are the classes just at different stages?
(2026-08-13)

early_bit_ablation.py compares the classes at matched EPOCH (3000, 5000, 7000,
9000) and finds that surgical removal of the type bit costs class A up to 0.289 of
type accuracy while the magnitude-matched control costs 0.008, with class B flat
in every condition. The trouble is that class A groks later -- the within-order
AUC of grok epoch predicting class is 0.74 -- so at epoch 7000 the two classes are
not at the same point in their own training. The finding has a ready alternative
reading: the bit is not doing work, class A is simply earlier in a trajectory that
every model passes through, and we are comparing an early class A against a
late-ish class B.

This removes the confound by indexing checkpoints on VALIDATION ACCURACY instead
of on epoch. For each run, the checkpoint closest to each target accuracy is
selected (subject to TOL, else that cell is empty), and the identical
surgical-versus-noise contrast is applied there.

  surgical : type-constant contrast removed along t = mean_c(U[r_c] - U[s_c])
  noise    : same per-row magnitude along t, balanced within type, so the type
             contrast survives intact. The control that matters -- it is what
             showed the convergence-time effect to be perturbation size.

Predictions, stated before running:
  - If the gap survives at matched accuracy, the stage confound is dead and the
    bit really was doing work at a point where class B, at the SAME competence,
    was doing that work some other way.
  - If the gap collapses once accuracy is matched, then the epoch-indexed result
    was a training-stage artifact and Section 3.4 needs rewriting around it.

The matched epochs themselves are reported per class, since the size of the
confound being corrected is worth seeing rather than asserting.

Run: python -m instruments.matched_acc_bit_ablation
"""
import glob

from instruments.population import epoch_checkpoints, run_dir
import os
from collections import defaultdict

import numpy as np
import torch

from instruments.bit_emergence_trajectory import val_at, val_curve
from instruments.das_type_subspace import build
from instruments.dc_component_ablation import truth
from instruments.early_bit_ablation import final_class, type_acc, unit_t
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

TARGETS = [0.35, 0.50, 0.70, 0.90]
TOL = 0.05                      # a checkpoint must land this close to the target
SUB = 3000
DRAWS = 3


def ep_of(p):
    return int(p.split("_")[-1].split(".")[0])


def pick(cks, curve, target):
    """Checkpoint whose validation accuracy is nearest `target`, or None.

    Ties and plateaus: training passes each accuracy once on the way up, so
    nearest-in-accuracy is well defined in practice; where several checkpoints sit
    equally close (a plateau), the EARLIEST is taken, which is the conservative
    choice for a claim about early training.
    """
    best, best_d = None, 1e9
    for c in cks:
        v = val_at(curve, ep_of(c))
        if v != v:                          # nan
            continue
        d = abs(v - target)
        if d < best_d - 1e-12:
            best, best_d = (c, v), d
    if best is None or best_d > TOL:
        return None
    return best


def ablate(ck, n, sp, tt):
    m0, tok = build(ck, n)
    base = type_acc(m0, tok, sp, n, tt)

    ms, _ = build(ck, n)
    with torch.no_grad():
        U = ms.fc_out.weight
        th = unit_t(U, n)
        pr = U @ th
        d = (pr[:n].mean() - pr[n:2 * n].mean()) / 2
        o = torch.zeros_like(pr)
        o[:n], o[n:2 * n] = d, -d
        U -= torch.outer(o, th)
    surg = type_acc(ms, tok, sp, n, tt)

    noi = []
    for s in range(DRAWS):
        mn, _ = build(ck, n)
        g = torch.Generator().manual_seed(11 + s)
        with torch.no_grad():
            U = mn.fc_out.weight
            th = unit_t(U, n)
            pr = U @ th
            d = (pr[:n].mean() - pr[n:2 * n].mean()) / 2
            o = torch.zeros_like(pr)
            for lo, hi in ((0, n), (n, 2 * n)):
                k = hi - lo
                sg = torch.ones(k)
                sg[torch.randperm(k, generator=g)[:k // 2]] = -1.0
                sg -= sg.mean()
                o[lo:hi] = d * sg
            U -= torch.outer(o, th)
        noi.append(type_acc(mn, tok, sp, n, tt))
    return base, surg, float(np.mean(noi))


def main():
    print("  Type-bit ablation at matched VALIDATION ACCURACY, not matched epoch.")
    print(f"  Target accuracies {TARGETS}, tolerance +-{TOL}. Chance = 0.50.\n")
    cache = {}
    rows = []
    agg = defaultdict(lambda: defaultdict(list))
    for rd in sorted(d for d in glob.glob(run_dir("dense_checkpoint_sweep/d*"))
                     if os.path.isdir(d)):
        n = int(os.path.basename(rd).split("_")[0][1:])
        cls = final_class(rd, n)
        if cls is None:
            continue
        if n not in cache:
            pairs, _, _, _ = all_pairs_typed(n)
            ti, tt = truth(pairs, n)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pairs), SUB, replace=False)
            cache[n] = ([pairs[i] for i in idx], tt[idx])
        sp, tt = cache[n]

        curve = val_curve(rd)
        cks = epoch_checkpoints(rd)
        for target in TARGETS:
            got = pick(cks, curve, target)
            if got is None:
                continue
            ck, v = got
            base, surg, noi = ablate(ck, n, sp, tt)
            agg[cls][target].append((base, surg, noi, ep_of(ck), v))
            rows.append((os.path.basename(rd), n, cls, target, ep_of(ck),
                         f"{v:.6f}", f"{base:.6f}", f"{surg:.6f}", f"{noi:.6f}"))
        print(f"  measured {os.path.basename(rd)} ({cls})", flush=True)

    save("matched_acc_bit_ablation",
         "run,n,cls,target_val,epoch,val,base,surgical,noise_only", rows)

    print(f"\n  {'target':>7} {'cls':>3} {'k':>3} | {'med epoch':>9} {'val':>6} | "
          f"{'base':>6} {'surgical':>9} {'noise':>7} | {'surg drop':>10} "
          f"{'noise drop':>11}")
    for target in TARGETS:
        for cls in ("A", "B"):
            v = np.array(agg[cls][target])
            if not len(v):
                continue
            print(f"  {target:>7.2f} {cls:>3} {len(v):>3} | "
                  f"{np.median(v[:,3]):>9.0f} {v[:,4].mean():>6.3f} | "
                  f"{v[:,0].mean():>6.3f} {v[:,1].mean():>9.3f} "
                  f"{v[:,2].mean():>7.3f} | {v[:,0].mean() - v[:,1].mean():>10.3f} "
                  f"{v[:,0].mean() - v[:,2].mean():>11.3f}")

    print("\n  The epoch column is the confound this instrument removes: at matched")
    print("  accuracy the classes sit at different epochs. If the surgical drop")
    print("  still separates the classes here, it is not a stage effect.")


if __name__ == "__main__":
    main()
