"""Was the bit doing work EARLY, or is the fossil story correlational? (2026-08-05)

At convergence the bit is inert: removing it costs no more than an equal-magnitude
perturbation that leaves it intact. Early in training class A gets output type
right far more often than class B (0.87 vs 0.67 at epoch 7000) while both sit at
index accuracy 0.30, the training fraction. The story we want to tell is that the
bit EARNS its place early and is superseded later.

That story is currently correlational, and this is the test that can kill it. Take
early checkpoints and remove the type-constant contrast surgically, exactly as at
convergence. Compare against the noise-only control (same magnitude, type contrast
intact) so the comparison is not confounded by perturbation size -- the mistake
that produced a false positive at convergence.

Predictions, stated before running:
  - Fossil story SURVIVES if, early, surgical removal drops type accuracy toward
    class B's level while noise-only does not.
  - Fossil story DIES if surgical and noise-only hurt equally early, exactly as
    they do at convergence. Then the bit never did any work and its correlation
    with early type accuracy has some other cause.

Run: python -m instruments.early_bit_ablation
"""
import glob

from instruments.population import epoch_checkpoints, run_dir
import os
from collections import defaultdict

import numpy as np
import torch

from instruments.das_type_subspace import build
from instruments.native_d63_base_rate import best_val
from instruments.dc_component_ablation import truth
from instruments.results_io import save
from instruments.type_xor_probe import all_pairs_typed

EPOCHS = [3000, 5000, 7000, 9000]
SUB = 3000

# The two runs the sign readout leaves unclassified. final_class stamps them
# "I" so that raw readers of any CSV built on it cannot re-bin them into a
# class by the kappa threshold (which puts one on each side of the gap).
UNCLASSIFIED = {"d47_seed6", "d61_seed4"}


def unit_t(U, n):
    t = (U[:n] - U[n:2 * n]).mean(0)
    return t / (t.norm() + 1e-12)


def type_acc(m, tok, sp, n, tt):
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]] for a, b in sp])
    with torch.no_grad():
        pred = m(ids)[:, 2, :2 * n].argmax(1).numpy()
    return float(((pred >= n).astype(int) == tt).mean())


def final_class(rd, n):
    """Class of a run's endpoint, or None if it never generalized.

    The generalization filter matters: an earlier version of this script classified
    all 30 dense runs by final geometry without it, giving 15 A / 15 B instead of
    the 10 A / 13 B / 2 unclassified that actually reached 0.99. The paper's
    numbers must come from generalized runs only, since every other claim is
    conditioned on them. The two runs the sign readout leaves unclassified are
    stamped "I", never A or B.
    """
    ck = os.path.join(rd, "checkpoints", "best_model.pt")
    if not os.path.exists(ck):
        return None
    if best_val(rd) < 0.99:
        return None
    if os.path.basename(rd) in UNCLASSIFIED:
        return "I"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    U = sd["fc_out.weight"].numpy()
    D = U[:n] - U[n:2 * n]
    t = D.mean(0)
    dcf = float(t @ t) / float((D * D).sum(1).mean())
    return "A" if dcf >= 0.09 else "B"


def main():
    print("  Removing the type bit at EARLY checkpoints, before the models can "
          "multiply.")
    print("  base / surgical (bit removed) / noise (same magnitude, bit intact). "
          "Chance = 0.50.\n")
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

        epoch_checkpoints(rd)   # fails loudly if only final weights are present
        for ep in EPOCHS:
            ck = os.path.join(rd, "checkpoints", f"checkpoint_epoch_{ep}.pt")
            if not os.path.exists(ck):
                continue
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
            for s in range(3):
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
            agg[cls][ep].append((base, surg, float(np.mean(noi))))
            rows.append((os.path.basename(rd), n, cls, ep, f"{base:.6f}",
                         f"{surg:.6f}", f"{np.mean(noi):.6f}"))
        print(f"  measured {os.path.basename(rd)} ({cls})", flush=True)

    save("early_bit_ablation", "run,n,cls,epoch,base,surgical,noise_only", rows)

    print(f"\n  {'epoch':>7} {'cls':>3} {'k':>3} | {'base':>6} {'surgical':>9} "
          f"{'noise':>7} | {'surg drop':>10} {'noise drop':>11}")
    for ep in EPOCHS:
        for cls in ("A", "B"):
            v = np.array(agg[cls][ep])
            if not len(v):
                continue
            print(f"  {ep:>7} {cls:>3} {len(v):>3} | {v[:,0].mean():>6.3f} "
                  f"{v[:,1].mean():>9.3f} {v[:,2].mean():>7.3f} | "
                  f"{v[:,0].mean() - v[:,1].mean():>10.3f} "
                  f"{v[:,0].mean() - v[:,2].mean():>11.3f}")


if __name__ == "__main__":
    main()
