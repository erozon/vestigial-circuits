"""Distributed Alignment Search (DAS) for the output-TYPE variable (2026-07-11).

The definitive test of the localized-vs-non-localized fork. We LEARN an orthonormal
FF subspace R (d_ff x r) and, via interchange, replace B's projection onto R with
A's, then decode. Because the minimal pair shares the index (A=(r_a,r_b)->r_{a+b},
B=(r_a,s_b)->s_{a+b}), the clean counterfactual "B but output-type=rotation" is
exactly r_{a+b} = A's token. So we optimize R to make the patched output = A's
token: a subspace that flips TYPE while PRESERVING index. Sweep r; the smallest r
that achieves high (held-out) clean-flip is the effective dimensionality of the
type variable.

  localized run   : saturates at small r (compact type variable) -- positive control.
  non-localized   : needs large r / never saturates below full => genuinely spread.
Full-rank patch (fact) = 100% by construction; that is the trivial upper anchor.

Train/test split over minimal pairs (70/30) so a high rank cannot just memorize.
Run: python -m instruments.das_type_subspace
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
STEPS = 300
THRESH = 50.0

from instruments.population import run_dir  # noqa: E402
from src.dihedral import elements  # noqa: E402
from src.model import DihedralModel  # noqa: E402
from src.tokenizer import create_tokenizer  # noqa: E402

MODELS = [
    ("D59_s0  loc", run_dir("seed_sweep/d59_30pct_seed0/checkpoints/best_model.pt"), 59),
    ("D47_s1  loc", run_dir("prime_composite_sweep/d47_seed1/checkpoints/best_model.pt"), 47),
    ("D63_s2  loc", run_dir("prime_composite_sweep/d63_seed2/checkpoints/best_model.pt"), 63),
    ("D63_s0  qsi", run_dir("prime_composite_sweep/d63_seed0/checkpoints/best_model.pt"), 63),
    ("D53_s2  NON", run_dir("prime_composite_sweep/d53_seed2/checkpoints/best_model.pt"), 53),
    ("D63_s1  NON", run_dir("prime_composite_sweep/d63_seed1/checkpoints/best_model.pt"), 63),
]


def build(ck_path, n):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    m = DihedralModel(vocab_size=2 * n + 4, d_model=128, num_layers=1, num_heads=4,
                      d_ff=sd["transformer_blocks.0.feed_forward.fc1.weight"].shape[0],
                      max_seq_length=8, dropout=0.0, use_layernorm=False,
                      learned_pos_emb=True, nanda_init=True)
    m.load_state_dict(sd); m.eval()
    return m, create_tokenizer(n)


def act_resid(m, tok, toks):
    bos = tok.token_to_id["<BOS>"]
    ids = torch.tensor([[bos, tok.token_to_id[a], tok.token_to_id[b]] for a, b in toks])
    blk = m.transformer_blocks[0]; ff = blk.feed_forward
    with torch.no_grad():
        x = m.pos_encoding(m.embedding(ids) * m.embed_scale)
        Q = blk.attention.query(x).view(-1, 3, 4, 32).transpose(1, 2)
        Kk = blk.attention.key(x).view(-1, 3, 4, 32).transpose(1, 2)
        Vv = blk.attention.value(x).view(-1, 3, 4, 32).transpose(1, 2)
        aw = torch.softmax(torch.matmul(Q, Kk.transpose(-2, -1)) / math.sqrt(32), -1)
        ctx = torch.matmul(aw, Vv).transpose(1, 2).contiguous().view(-1, 3, 128)
        resid = (x + blk.attention.fc_out(ctx))[:, 2, :]
        act = torch.relu(ff.fc1(resid))
    return act, resid


def minimal_pairs(n):
    A, B, idxfam = [], [], []
    for fam in (1, 2):
        for a in range(n):
            for b in range(n):
                if fam == 1:
                    A.append((f"r{a}", f"r{b}")); B.append((f"r{a}", f"s{b}")); idx = (a + b) % n
                else:
                    A.append((f"s{a}", f"s{b}")); B.append((f"s{a}", f"r{b}")); idx = (a - b) % n
                idxfam.append(idx)
    return A, B, idxfam


def main():
    torch.manual_seed(0)
    print(f"  DAS: learn rank-r FF subspace; patch A->B; target = A's exact token (type-flip")
    print(f"  + index preserved). Cells = held-out clean-flip %. r_loc = smallest r > {THRESH:.0f}%.")
    print(f"  {'model':12} | " + " ".join(f"r{r:>3}" for r in RANKS) + f" | {'full':>4} {'r_loc':>5}")
    for label, ck, n in MODELS:
        if not os.path.exists(ck):
            print(f"  {label:12}  MISSING"); continue
        m, tok = build(ck, n)
        A, B, idxfam = minimal_pairs(n)
        idA = torch.tensor([tok.token_to_id[f"r{i}"] for i in idxfam])
        actA, _ = act_resid(m, tok, A)
        actB, residB = act_resid(m, tok, B)
        ff = m.transformer_blocks[0].feed_forward
        W2, b2 = ff.fc2.weight.detach(), ff.fc2.bias.detach()
        Wout = m.fc_out.weight.detach()
        bout = m.fc_out.bias.detach() if m.fc_out.bias is not None else torch.zeros(Wout.shape[0])
        d_ff = actA.shape[1]

        rng = np.random.default_rng(0)
        perm = rng.permutation(len(idA)); cut = int(0.7 * len(idA))
        tr, te = perm[:cut], perm[cut:]

        def readout(act, resid):
            return (resid + act @ W2.t() + b2) @ Wout.t() + bout

        # full-rank anchor (patch all of act) = fact_cln
        with torch.no_grad():
            full = 100 * (readout(actA[te], residB[te]).argmax(1) == idA[te]).float().mean().item()

        cells, r_loc = [], None
        for r in RANKS:
            if r >= d_ff:
                cells.append(full); continue
            Wp = nn.Parameter(torch.randn(d_ff, r) * 0.02)
            opt = torch.optim.Adam([Wp], lr=0.05)
            for _ in range(STEPS):
                R = torch.linalg.qr(Wp, mode="reduced")[0]          # (d_ff, r) orthonormal
                patch = actB[tr] + (actA[tr] @ R - actB[tr] @ R) @ R.t()
                loss = nn.functional.cross_entropy(readout(patch, residB[tr]), idA[tr])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                R = torch.linalg.qr(Wp, mode="reduced")[0]
                patch = actB[te] + (actA[te] @ R - actB[te] @ R) @ R.t()
                acc = 100 * (readout(patch, residB[te]).argmax(1) == idA[te]).float().mean().item()
            cells.append(acc)
            if r_loc is None and acc > THRESH:
                r_loc = r
        print(f"  {label:12} | " + " ".join(f"{c:>4.0f}" for c in cells) +
              f" | {full:>4.0f} {str(r_loc) if r_loc else 'none':>5}")
    print("\n  small r_loc = compact type variable (localized, incl. rotated basis).")
    print("  large/none while full~100 = type variable is genuinely spread across the FF.")


if __name__ == "__main__":
    main()
