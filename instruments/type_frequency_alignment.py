"""Do the type-carrying frequencies coincide with the index-circuit frequencies?
(2026-08-05)

instruments/type_structure_spectrum.py showed that the output layer's
type-discriminating vectors D_c = U[r_c] - U[s_c] have ~72% of their energy in
three nonzero frequencies of c, in BOTH classes -- the classes differ only in the
index-independent (k=0) term. Concentration at a few frequencies is expected of
any model implementing this group, so on its own it says little.

The question this instrument answers: are those the SAME frequencies the model
uses for the modular-arithmetic circuit, or different ones?

Index-circuit frequencies are read off the input embedding, the standard route:
FFT the rotation-token embedding rows over the index and see which frequencies
carry energy. Type frequencies come from D_c as before. Both are spectra over the
same index variable, so they are directly comparable.

  emb_top3   : the three strongest nonzero frequencies of the rotation embedding
  typ_top3   : the three strongest nonzero frequencies of D_c
  overlap    : |emb_top3 & typ_top3|, 0 to 3
  chance     : expected overlap if the two triples were independent, 9/((n-1)/2)
  corr       : Pearson correlation of the two full folded energy spectra

Reading, stated before running:
  - overlap ~3 and high corr => the type distinction rides on the same frequencies
    as the index computation.
  - overlap ~chance => the type structure occupies its own frequencies, separate
    from the arithmetic.
Whether this differs between classes is the point; report per class.

Run: python -m instruments.type_frequency_alignment
"""
import os

import numpy as np
import torch

from instruments.population import RUNS, ckpt
from instruments.results_io import save


def folded_energy(M, n):
    """Energy per nonzero frequency of an (n, d) matrix, folding k with n-k."""
    F = np.fft.fft(M, axis=0) / n
    e = (np.abs(F) ** 2).sum(1)
    nz = e[1:]
    half = (n - 1) // 2
    return np.array([nz[k] + nz[n - 2 - k] for k in range(half)])   # index k -> freq k+1


def top3(folded):
    return set(int(k) + 1 for k in np.argsort(folded)[::-1][:3])


def main():
    print("  Type-carrying frequencies vs index-circuit frequencies.\n")
    print(f"  {'run':28} {'cls':>3} {'emb_top3':>14} {'typ_top3':>14} "
          f"{'ovl':>4} {'chance':>7} {'corr':>6}")
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
        E = sd["embedding.weight"].numpy()
        U = sd["fc_out.weight"].numpy()

        emb = folded_energy(E[:n], n)                 # rotation-token embeddings
        typ = folded_energy(U[:n] - U[n:2 * n], n)    # type-discriminating vectors
        et, tt = top3(emb), top3(typ)
        ovl = len(et & tt)
        chance = 9.0 / ((n - 1) // 2)
        corr = float(np.corrcoef(emb, typ)[0, 1])

        print(f"  {os.path.basename(path):28} {cls:>3} "
              f"{str(sorted(et)):>14} {str(sorted(tt)):>14} {ovl:>4} "
              f"{chance:>7.2f} {corr:>6.2f}")
        agg.setdefault(cls, []).append((ovl, chance, corr))
        # top-3 sets are written space-separated so the CSV stays one row per run
        rows.append((os.path.basename(path), n, cls,
                     " ".join(str(k) for k in sorted(et)),
                     " ".join(str(k) for k in sorted(tt)),
                     ovl, f"{chance:.6f}", f"{corr:.6f}"))
    print()
    save("type_frequency_alignment",
         "run,n,cls,emb_top3,typ_top3,overlap,chance,corr", rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): mean overlap {a[:,0].mean():.2f} of 3 "
              f"(chance {a[:,1].mean():.2f}), mean corr {a[:,2].mean():.2f}, "
              f"overlap=3 in {int((a[:,0]==3).sum())}/{len(a)} runs")


if __name__ == "__main__":
    main()
