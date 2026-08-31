"""Where does the type-discriminating structure sit as a function of index? (2026-08-05)

Weights only. For each group index c, the output layer's type-discriminating
direction is

    D_c = U[r_c] - U[s_c]        (d-dimensional)

The zero-parameter readout uses t = mean_c D_c. Class A runs have large ||t||
relative to mean_c||D_c||^2 (the "common" fraction); class B runs have it at or
below chance. That says the D_c do not share a direction. It does NOT say the D_c
are unstructured -- they could vary systematically with c and cancel on averaging.

This instrument distinguishes those. Take the discrete Fourier transform of D_c
over c and report where the energy sits:

    F_k = (1/n) sum_c D_c exp(-2 pi i k c / n),   energy_k = ||F_k||^2

  dc_frac   : energy at k=0 as a fraction of total (this is the "common" measure)
  top_k     : the nonzero frequency carrying the most energy
  top_frac  : that frequency's share of total energy
  top3_frac : the three strongest nonzero frequencies' combined share
  chance    : 1/n per frequency, the flat-spectrum reference for unstructured D_c

Reading, stated before running:
  - If class B has top_frac >> 1/n, the type distinction is present in the output
    layer but bound to the index rather than free-standing.
  - If class B's spectrum is flat (top_frac ~ 1/n), the output layer carries no
    index-independent type structure at all.
These are different claims and the paper should not assert either without this.

Run: python -m instruments.type_structure_spectrum
"""
import os

import numpy as np

from instruments.population import RUNS, ckpt
from instruments.results_io import save
from instruments.unembed_type_geometry import load_unembed


def spectrum(U, n):
    D = U[:n] - U[n:2 * n]                      # (n, d)
    F = np.fft.fft(D, axis=0) / n               # (n, d) complex
    energy = (np.abs(F) ** 2).sum(1)            # (n,)
    total = energy.sum()
    dc = energy[0] / total
    nz = energy[1:]
    # real D_c => spectrum is conjugate-symmetric; fold so a frequency and its
    # mirror count once, otherwise every "top" frequency is silently doubled.
    half = (n - 1) // 2
    folded = np.array([nz[k] + nz[n - 2 - k] for k in range(half)])
    freqs = np.arange(1, half + 1)
    order = np.argsort(folded)[::-1]
    top_k = int(freqs[order[0]])
    top_frac = float(folded[order[0]] / total)
    top3 = float(folded[order[:3]].sum() / total)
    return float(dc), top_k, top_frac, top3


def main():
    print("  Type-discriminating structure across the output layer, by frequency.\n")
    print(f"  {'run':44} {'cls':>3} {'dc_frac':>8} {'top_k':>6} {'top_frac':>9} "
          f"{'top3':>7} {'chance':>7}")
    agg, rows = {}, []
    for path, n, cls in RUNS:
        p = ckpt(path)
        if not os.path.exists(p):
            print(f"  {path:44} {cls:>3}  (no checkpoint)")
            continue
        U, _ = load_unembed(p)
        dc, top_k, top_frac, top3 = spectrum(U, n)
        print(f"  {os.path.basename(path):44} {cls:>3} {dc:>8.3f} {top_k:>6} "
              f"{top_frac:>9.3f} {top3:>7.3f} {1.0 / n:>7.3f}")
        agg.setdefault(cls, []).append((dc, top_frac, top3, 1.0 / n))
        rows.append((os.path.basename(path), n, cls, f"{dc:.6f}", top_k,
                     f"{top_frac:.6f}", f"{top3:.6f}", f"{1.0 / n:.6f}"))
    print()
    save("type_structure_spectrum",
         "run,n,cls,dc_frac,top_k,top_frac,top3_frac,chance", rows)
    print()
    for cls in ("A", "B", "I"):
        if cls not in agg:
            continue
        a = np.array(agg[cls])
        print(f"  class {cls} (n={len(a)}): dc_frac {a[:,0].mean():.3f} "
              f"[{a[:,0].min():.3f}, {a[:,0].max():.3f}]  "
              f"top_frac {a[:,1].mean():.3f} [{a[:,1].min():.3f}, {a[:,1].max():.3f}]  "
              f"top3 {a[:,2].mean():.3f}  chance {a[:,3].mean():.3f}")


if __name__ == "__main__":
    main()
