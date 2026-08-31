"""Is output type an UNLINEARIZED XOR of input types? (2026-07-13)

Robust fork (3 instruments): output type is globally linearly decodable from the
MLP hidden layer in localized runs (~100%) but at CHANCE in non-localized runs
(~50%) -- yet non-localized models output type at 100% accuracy. Output type in
D_n is exactly XOR of the two input types (rr,ss->rotation; rs,sr->reflection),
the canonical not-linearly-decodable function.

This probes the same MLP activations (post-ReLU, result position, full dataset) for:
  lin_op1 : linear decode of operand-1's type   (input feature)
  lin_op2 : linear decode of operand-2's type   (input feature)
  lin_out : linear decode of OUTPUT type (=op1 XOR op2)   [anchor; ~100 loc / ~50 non]
  nl_out  : 1-hidden-layer decode of output type          (nonlinear)
  nl_op1  : nonlinear decode of op1 type (capacity sanity; should be ~100)

Reading:
  non-localized: lin_op1 & lin_op2 HIGH, lin_out ~chance, nl_out HIGH
                 => input types linearly present, output type only as their XOR
                    (unlinearized) => the MLP has NOT computed an explicit type bit.
  localized    : lin_out HIGH (explicit, linearized type bit). What happens to
                 lin_op1/lin_op2 tells us whether it also retains the input types
                 or collapses them into the output bit.

Run: python -m instruments.type_xor_probe
"""
import os

import numpy as np
import torch
import torch.nn as nn

from instruments.das_type_subspace import MODELS, act_resid, build


def all_pairs_typed(n):
    toks = [f"r{i}" for i in range(n)] + [f"s{i}" for i in range(n)]
    pairs, o1, o2 = [], [], []
    for t1 in toks:
        for t2 in toks:
            pairs.append((t1, t2))
            o1.append(0 if t1[0] == "r" else 1)
            o2.append(0 if t2[0] == "r" else 1)
    o1, o2 = np.array(o1), np.array(o2)
    return pairs, o1, o2, (o1 ^ o2)


def split(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows); cut = int(0.7 * n_rows)
    return perm[:cut], perm[cut:]


def lin_probe(X, y, tr, te, steps=1000):
    yt = torch.tensor(y, dtype=torch.float32)
    w = nn.Parameter(torch.zeros(X.shape[1])); b = nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(X[tr] @ w + b, yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return 100 * (((X[te] @ w + b) > 0).float() == yt[te]).float().mean().item()


def mlp_probe(X, y, tr, te, hidden=64, steps=1500):
    yt = torch.tensor(y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(net(X[tr]).squeeze(1), yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return 100 * ((net(X[te]).squeeze(1) > 0).float() == yt[te]).float().mean().item()


def main():
    torch.manual_seed(0)
    print("  XOR probe on MLP hidden. lin_* = linear decode; nl_* = 1-hidden-layer decode.")
    print("  non-loc hypo: lin_op1/op2 high, lin_out ~chance, nl_out high (unlinearized XOR).\n")
    print(f"  {'model':12} | {'lin_op1':>7} {'lin_op2':>7} {'lin_out':>7} | {'nl_out':>6} {'nl_op1':>6}")
    for label, ck, n in MODELS:
        if not os.path.exists(ck):
            print(f"  {label:12}  MISSING"); continue
        m, tok = build(ck, n)
        pairs, o1, o2, out = all_pairs_typed(n)
        actFull, _ = act_resid(m, tok, pairs)
        X = actFull
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        tr, te = split(X.shape[0])
        lin_op1 = lin_probe(X, o1, tr, te)
        lin_op2 = lin_probe(X, o2, tr, te)
        lin_out = lin_probe(X, out, tr, te)
        nl_out = mlp_probe(X, out, tr, te)
        nl_op1 = mlp_probe(X, o1, tr, te)
        print(f"  {label:12} | {lin_op1:>6.0f}% {lin_op2:>6.0f}% {lin_out:>6.0f}% "
              f"| {nl_out:>5.0f}% {nl_op1:>5.0f}%")
    print("\n  non-loc: lin_op1/op2 high + lin_out chance + nl_out high => output type kept")
    print("           as unlinearized XOR of (linearly present) input types.")
    print("  loc: lin_out high => explicit linearized type bit.")


if __name__ == "__main__":
    main()
