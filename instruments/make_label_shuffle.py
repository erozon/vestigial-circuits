"""Label-shuffle control data: permute the RESULT token across train lines.

Destroys the group map (input pair -> product) while preserving the label
distribution: each (g1, g2) keeps its inputs but gets a random result drawn from
the shuffled pool. A model can only MEMORIZE this (no generalizable structure),
so any Fourier/single-tone concentration the analysis reports on it would be a
pipeline artifact rather than learned group computation.

Fixed seed for reproducibility. Only train.txt is shuffled (val stays real ->
val_acc measures generalization, which must sit at chance).

Usage: python -m instruments.make_label_shuffle <in_train> <out_train> [seed]
"""
import sys
import numpy as np


def main(inp, outp, seed=0):
    lines = [ln.split() for ln in open(inp).read().splitlines() if ln.strip()]
    results = [w[2] for w in lines]
    perm = np.random.default_rng(int(seed)).permutation(len(results))
    shuffled = [results[i] for i in perm]
    fixed = sum(a == b for a, b in zip(results, shuffled))
    with open(outp, "w") as f:
        for (g1, g2, _), r in zip(lines, shuffled):
            f.write(f"{g1} {g2} {r}\n")
    print(f"{inp} -> {outp}: {len(lines)} lines shuffled (seed {seed}); "
          f"{fixed} coincidental fixed points ({100*fixed/len(lines):.1f}%).")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 0)
