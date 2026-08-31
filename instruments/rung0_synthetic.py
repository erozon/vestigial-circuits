"""Rung 0 of the calibration ladder: DFT / concentration sanity on KNOWN signals.

Purpose: validate the Fourier-concentration measurement on synthetic signals
whose answer is known analytically, BEFORE trusting it on any trained network.
No checkpoint, no group elements, no G x G reduction is involved here -- this is
pure measurement-code validation, decoupled from the science.

Spec: `~/.claude/projects/grok-dihedral-scratch/CALIBRATION_EXPERIMENTS.md`,
Rung 0 (definitions in "Core definitions"). This file implements `conc()`
faithfully to that spec:

    conc(f) = max_k P(k) / sum_k P(k),   P(k) = |f_hat(k)|^2
    - folded: for real f, P(k) == P(N-k); independent freqs are k = 1..floor(N/2)
    - k = 0 (DC) is DROPPED via mean-centering (the stated, fixed choice)
    - bounded in [1/n_freq, 1]

DERIVATION-FIRST DISCIPLINE (LEARNING.md + the calibration doc):
This script prints OBSERVED values only. Do NOT read them until you have written
your PREDICTION for each of 0a-0e in the calibration doc, by hand. The whole
point of the ladder is that a prediction you derived catches a measurement bug;
a number you read off the screen does not. Fill "Observed" + PASS/FAIL after.

Run:  python -m instruments.rung0_synthetic
"""

import numpy as np


def conc(f, mean_center=True):
    """Top-1 folded spectral-energy concentration of a real signal f: Z/N -> R.

    Returns (concentration, dominant_frequency k*). Frequencies are 1..floor(N/2);
    k=0 is removed by mean-centering (the fixed DC choice). See module docstring.
    """
    f = np.asarray(f, dtype=float)
    N = f.shape[0]
    if mean_center:
        f = f - f.mean()
    P = np.abs(np.fft.fft(f)) ** 2
    n_ind = N // 2  # independent nonzero freqs: k = 1 .. floor(N/2)
    Pf = np.empty(n_ind)
    for k in range(1, n_ind + 1):
        # fold k with its conjugate partner N-k; for even N the Nyquist bin
        # (k == N-k) has no distinct partner.
        Pf[k - 1] = P[k] + (P[N - k] if k != (N - k) else 0.0)
    total = Pf.sum()
    if total == 0.0:
        return np.nan, None
    kstar = int(np.argmax(Pf)) + 1
    return float(Pf.max() / total), kstar


def support(f, tol=1e-9):
    """Indices k where |f_hat(k)| > tol (NOT mean-centered -- shows the DC/k=0
    component too). Used for the coset-indicator sublattice check (0e)."""
    fhat = np.fft.fft(np.asarray(f, dtype=float))
    return np.nonzero(np.abs(fhat) > tol)[0]


def main():
    N = 59  # matches the real D_59 case; n_freq = 29
    a = np.arange(N)

    print(f"Rung 0 -- DFT/concentration sanity (N={N}, n_freq={N // 2}, "
          f"DC dropped via mean-centering)\n")
    print("  Write your prediction in the calibration doc BEFORE reading these.\n")

    # (0a) pure tone
    f0a = np.cos(2 * np.pi * 7 * a / N)
    c, k = conc(f0a)
    print(f"(0a) pure tone cos(2pi*7*a/{N}):        conc={c:.4f}  k*={k}")

    # (0b) single-input delta
    f0b = np.zeros(N); f0b[0] = 1.0
    c, k = conc(f0b)
    print(f"(0b) delta at a=0:                     conc={c:.4f}  k*={k}  (floor=1/{N // 2}={1 / (N // 2):.4f})")

    # (0c) two-tone mix (amplitudes 1.0 and 0.5)
    f0c = np.cos(2 * np.pi * 7 * a / N) + 0.5 * np.cos(2 * np.pi * 13 * a / N)
    c, k = conc(f0c)
    print(f"(0c) two-tone (k=7 @1.0, k=13 @0.5):   conc={c:.4f}  k*={k}")

    # (0d) phase-shifted tone -- concentration must be phase-blind
    f0d = np.cos(2 * np.pi * 7 * a / N + 1.0)
    c, k = conc(f0d)
    print(f"(0d) phase-shifted tone (phase=1.0):   conc={c:.4f}  k*={k}")

    # (0e) coset indicator on a COMPOSITE modulus: N=45, H = <9> (order 5).
    # NOTE (correction to the calibration doc as written): the Fourier support of
    # the indicator of a subgroup H is the ANNIHILATOR H^perp = {k : k*a == 0 mod
    # N for all a in H}. For H = <9> in Z/45 that is {k : 5 | k} -- MULTIPLES OF 5
    # (nine frequencies), NOT multiples of 9. |H^perp| = N/|H| = 45/5 = 9. The doc
    # asserted multiples of 9 (it wrote down H itself). Derive H^perp yourself,
    # then confirm below.
    N45 = 45
    f0e = np.zeros(N45); f0e[[0, 9, 18, 27, 36]] = 1.0
    print(f"\n(0e) coset indicator, N={N45}, H=<9>={{0,9,18,27,36}}:")
    print(f"     f_hat support (which sublattice lights up): {support(f0e).tolist()}")


if __name__ == "__main__":
    main()
