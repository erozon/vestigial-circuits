"""Analyze the prime-vs-composite sweep: does the one-sided output-type override
recur in primes and not composites? (2026-07-08)

Runnable anytime -- it analyzes whatever grokked checkpoints exist so far
(sweep anchors D_59/D_45 + runs/prime_composite_sweep/*). For each grokked model:
  override_score = mean(rs,sr acc) - mean(rr,ss acc) after ablating the top-50
                   neurons ranked by |corr(output-type, preact)|.
  ~99 => clean rotation-output override (D_59 signature); ~0 => no override.

Run:  python -m instruments.prime_composite_analyze
"""
import glob
import math
import os
import numpy as np
import torch

from instruments.population import run_dir  # noqa: E402
from src.model import DihedralModel
from src.tokenizer import create_tokenizer
from src.dihedral import elements, multiply


def is_prime(n):
    return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))


# (label, checkpoint, n) -- anchors first, then the sweep (discovered below)
ANCHORS = [
    ("D59_s0", run_dir("seed_sweep/d59_30pct_seed0/checkpoints/best_model.pt"), 59),
    ("D59_s1", run_dir("seed_sweep/d59_30pct_seed1/checkpoints/best_model.pt"), 59),
    ("D59_s4", run_dir("seed_sweep/d59_30pct_seed4/checkpoints/best_model.pt"), 59),
    ("D45_s1", run_dir("d45_sweep/d45_30pct_seed1/checkpoints/best_model.pt"), 45),
    ("D45_s4", run_dir("d45_sweep/d45_30pct_seed4/checkpoints/best_model.pt"), 45),
    ("D45_s0", run_dir("d45_pilot/seed0/checkpoints/best_model.pt"), 45),
    ("D45_s2r", run_dir("d45_sweep/d45_30pct_seed2_resumed/checkpoints/best_model.pt"), 45),
]


def discover_sweep():
    out = []
    for d in sorted(glob.glob(run_dir("prime_composite_sweep/d*_seed*"))):
        ck = os.path.join(d, "checkpoints", "best_model.pt")
        base = os.path.basename(d)          # d47_seed0
        n = int(base.split("_")[0][1:])
        seed = base.split("seed")[-1]
        if os.path.exists(ck):
            out.append((f"D{n}_s{seed}", ck, n))
    return out


def override_score(ck_path, n):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    m = DihedralModel(vocab_size=2 * n + 4, d_model=128, num_layers=1, num_heads=4,
                      d_ff=sd["transformer_blocks.0.feed_forward.fc1.weight"].shape[0],
                      max_seq_length=8, dropout=0.0, use_layernorm=False,
                      learned_pos_emb=True, nanda_init=True)
    m.load_state_dict(sd); m.eval()
    tok = create_tokenizer(n)
    elems = elements(n)
    pairs = [(g, h) for g in elems for h in elems]
    a_rot = np.array([g[0] == "r" for g, _ in pairs])
    b_rot = np.array([h[0] == "r" for _, h in pairs])
    out_rot = (a_rot == b_rot).astype(float)
    quad = np.array(["rr" if ar and br else "rs" if ar and not br
                     else "sr" if not ar and br else "ss" for ar, br in zip(a_rot, b_rot)])
    bos = tok.token_to_id["<BOS>"]
    g_ids = torch.tensor([tok.token_to_id[g] for g, _ in pairs])
    h_ids = torch.tensor([tok.token_to_id[h] for _, h in pairs])
    exp = torch.tensor([tok.token_to_id[multiply(g, h, n)] for g, h in pairs])
    inp = torch.stack([torch.full((len(g_ids),), bos, dtype=torch.long), g_ids, h_ids], 1)
    blk = m.transformer_blocks[0]
    d_ff = blk.feed_forward.fc1.out_features
    with torch.no_grad():
        x = m.pos_encoding(m.embedding(inp) * m.embed_scale)
        Q = blk.attention.query(x).view(-1, 3, 4, 32).transpose(1, 2)
        K = blk.attention.key(x).view(-1, 3, 4, 32).transpose(1, 2)
        Vv = blk.attention.value(x).view(-1, 3, 4, 32).transpose(1, 2)
        aw = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(32), -1)
        ctx = torch.matmul(aw, Vv).transpose(1, 2).contiguous().view(-1, 3, 128)
        xpa = x + blk.attention.fc_out(ctx)
        pre = blk.feed_forward.fc1(xpa[:, 2, :])
        act = torch.relu(pre)
        base = (m.fc_out(xpa[:, 2, :] + blk.feed_forward.fc2(act)).argmax(-1) == exp).float().mean().item()
    pre_np = pre.numpy()
    tc = np.array([abs(np.corrcoef(out_rot, pre_np[:, j])[0, 1]) if pre_np[:, j].std() > 1e-8 else 0.0
                   for j in range(d_ff)])
    top = np.argsort(-tc)[:50]
    mask = torch.ones(d_ff); mask[list(top)] = 0.0
    with torch.no_grad():
        preds = m.fc_out(xpa[:, 2, :] + blk.feed_forward.fc2(act * mask.view(1, -1))).argmax(-1).numpy()
    ok = preds == exp.numpy()
    accq = {q: 100 * ok[quad == q].mean() for q in ("rr", "rs", "sr", "ss")}
    score = (accq["rs"] + accq["sr"]) / 2 - (accq["rr"] + accq["ss"]) / 2
    return base, score, accq


def main():
    models = ANCHORS + discover_sweep()
    rows = []
    print(f"  {'model':10} {'n':>3} {'type':>9} {'base':>5} {'ovr_score':>9}  "
          f"{'rr':>5} {'rs':>5} {'sr':>5} {'ss':>5}")
    for label, ck, n in models:
        if not os.path.exists(ck):
            continue
        try:
            base, score, aq = override_score(ck, n)
        except Exception as e:
            print(f"  {label:10} {n:>3}  ERROR {e}"); continue
        grok = "" if base >= 0.99 else "  (NOT GROKKED)"
        typ = "prime" if is_prime(n) else "composite"
        print(f"  {label:10} {n:>3} {typ:>9} {base:>5.2f} {score:>9.1f}  "
              f"{aq['rr']:>5.1f} {aq['rs']:>5.1f} {aq['sr']:>5.1f} {aq['ss']:>5.1f}{grok}")
        if base >= 0.99:
            rows.append((typ, score))
    print()
    for typ in ("prime", "composite"):
        s = [sc for t, sc in rows if t == typ]
        if s:
            print(f"  {typ:9}: n={len(s)}  override_score mean={np.mean(s):5.1f} "
                  f"min={min(s):5.1f} max={max(s):5.1f}")
    print("\n  HYPOTHESIS: primes -> high override_score (~99), composites -> ~0.")


if __name__ == "__main__":
    main()
