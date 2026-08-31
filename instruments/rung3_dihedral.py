"""Rung 3 of the calibration ladder: smallest non-abelian (D_n), per-quadrant.

First appearance of a 2-D irrep + reflection/type structure, at a scale where
you can inspect neurons by hand. This is the BRIDGE to the real D_59 basis test:
the reduction implemented here is the one Eric resolved on 2026-07-05.

G x G -> one-argument REDUCTION (Eric's decision, 2026-07-05):
  Four sweeps, one per quadrant (rr, rs, sr, ss). In every sweep:
    - FIX g1 (condition on a single value; do NOT marginalize -- averaging a
      pure tone over the fixed operand nullifies it, (1/n) sum_a cos(k(a+b))=0).
    - VARY g2 over the rotation index b = 0..n-1.
  Quadrant = (type of g1, type of g2):
    rr: g1=r_a0, g2=r_b  -> result r_{(a0+b)%n}
    rs: g1=r_a0, g2=s_b  -> result s_{(a0+b)%n}
    sr: g1=s_a0, g2=r_b  -> result s_{(a0-b)%n}
    ss: g1=s_a0, g2=s_b  -> result r_{(a0-b)%n}
  Within a quadrant g1's type is CONSTANT, so the sign-gate is held in a known
  fixed state (off for rr/rs, on for sr/ss) -- we measure the rotation-index
  structure with the gate pinned, per the refocused spine.

a0 ROBUSTNESS SWEEP: conc is phase-blind (Rung 0d), so for a clean single-freq
neuron conc should be ~invariant to a0. We condition on several a0 (avoiding the
identity) and report conc STABILITY across them. Instability is itself a
diagnostic that a neuron is not a clean tone (candidate coset structure).

METRIC: imports `conc` from rung0_synthetic -- the SAME Rung-0-certified metric.

NOTE: requires a GROKKED small-D checkpoint. As of 2026-07-05 none exists
(runs/d8 memorized, best val_acc 0.017; experiments/d29 plateaued at 0.48). Run
this only once a grokked checkpoint is available; the built-in accuracy guard
(--check) prints the model's accuracy on the swept inputs so you never analyze
an ungrokked model by accident.

Run:  python -m instruments.rung3_dihedral --ckpt <path> --n <n>
"""

import argparse

import numpy as np
import torch

from src.dihedral import multiply
from src.model import DihedralModel
from instruments.rung0_synthetic import conc  # Rung-0-certified metric

QUADRANTS = ("rr", "rs", "sr", "ss")


def load_model(ckpt_path, n):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model = DihedralModel(
        vocab_size=sd["embedding.weight"].shape[0],                       # 2n+4
        d_model=128, num_layers=1, num_heads=4,
        d_ff=sd["transformer_blocks.0.feed_forward.fc1.weight"].shape[0],
        max_seq_length=sd["pos_encoding.embedding.weight"].shape[0],
        dropout=0.0, use_layernorm=False, learned_pos_emb=True, nanda_init=True,
    )
    model.load_state_dict(sd)
    model.eval()
    assert model.vocab_size == 2 * n + 4, (model.vocab_size, n)
    return model


def tok(typ, idx, n):
    """token id: r_a -> a, s_a -> n+a  (vocab = [r0..r{n-1}, s0..s{n-1}, ...])."""
    return (idx % n) + (0 if typ == "r" else n)


def bos_id(n):
    return 2 * n + 2  # vocab = [...2n elements..., PAD, UNK, BOS, EOS]


def sweep(model, n, quad, a0):
    """One quadrant sweep at fixed g1: returns (preact (n, d_ff), model_acc).

    preact[b, j] = pre-activation (fc1 output, before ReLU) of neuron j at the
    result position, for g2 index b. model_acc = fraction of the n inputs whose
    argmax prediction equals the true product (guards against ungrokked models).
    """
    t1, t2 = quad[0], quad[1]
    b = np.arange(n)
    g1 = np.full(n, tok(t1, a0, n))
    g2 = np.array([tok(t2, bb, n) for bb in b])
    inp = torch.tensor(np.stack([np.full(n, bos_id(n)), g1, g2], axis=1), dtype=torch.long)

    captured = {}
    h = model.transformer_blocks[0].feed_forward.fc1.register_forward_hook(
        lambda m, i, o: captured.__setitem__("p", o.detach())
    )
    try:
        with torch.no_grad():
            logits = model(inp)
    finally:
        h.remove()

    preact = captured["p"][:, -1, :].numpy()                 # (n, d_ff) at result pos
    pred = logits[:, -1, :].argmax(-1).numpy()               # predicted result token
    true = np.array([_true_tok(t1, a0, t2, bb, n) for bb in b])
    acc = float((pred == true).mean())
    return preact, acc


def _true_tok(t1, a0, t2, bb, n):
    res = multiply(f"{t1}{a0}", f"{t2}{bb}", n)   # e.g. "s4"
    return tok(res[0], int(res[1:]), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a GROKKED D_n checkpoint")
    ap.add_argument("--n", type=int, required=True, help="dihedral n")
    ap.add_argument("--a0", type=int, nargs="+", default=[1, 3, 5],
                    help="fixed-g1 values for the robustness sweep (avoid 0=identity)")
    ap.add_argument("--hi", type=float, default=0.5, help="high-conc threshold")
    args = ap.parse_args()

    n = args.n
    a0s = [a % n for a in args.a0]
    model = load_model(args.ckpt, n)
    d_ff = model.transformer_blocks[0].feed_forward.fc1.out_features
    floor = 1.0 / (n // 2)
    print(f"Rung 3 -- D_{n}  ({d_ff} neurons, conc floor 1/{n // 2}={floor:.3f}, "
          f"a0={a0s})\n")

    for quad in QUADRANTS:
        # conc[a0_index, neuron]
        C = np.empty((len(a0s), d_ff))
        accs = []
        for ai, a0 in enumerate(a0s):
            preact, acc = sweep(model, n, quad, a0)
            accs.append(acc)
            for j in range(d_ff):
                C[ai, j] = conc(preact[:, j])[0]
        mean_acc = float(np.mean(accs))
        guard = "" if mean_acc > 0.99 else "  <-- WARNING: model not grokked on these inputs!"
        conc_mean = C.mean(0)                      # per-neuron conc, averaged over a0
        conc_std = C.std(0)                        # per-neuron a0-instability
        n_hi = int((conc_mean >= args.hi).sum())
        print(f"[{quad}] model_acc={mean_acc:.3f}{guard}")
        print(f"      conc: median={np.median(conc_mean):.3f} max={conc_mean.max():.3f}"
              f"  |  neurons conc>={args.hi}: {n_hi}/{d_ff}"
              f"  |  a0-instability (mean std): {conc_std.mean():.3f}")
    print("\n(For a grokked model expect model_acc~1.0, a pile of high-conc neurons,"
          " and LOW a0-instability where neurons are clean single tones.)")


if __name__ == "__main__":
    main()
