"""How equal is "indistinguishable"? Intervals on the class A - class B margin gap.
(2026-08-13)

The paper wants to say that both classes discriminate type by the same
index-bound frequency mechanism, at margins that do not differ. The evidence is
currently a pair of means -- 2.16 against 2.15 -- and that is an equivalence claim
resting on a null result with no interval attached. A reviewer is entitled to ask
what difference the data could have ruled out, and "the means were close" is not
an answer: with 16 and 17 runs, a real gap of moderate size would also produce
close means.

So report the DIFFERENCE with an interval, and let the interval carry the claim.
An honest statement is "A - B lies in [lo, hi], so any difference larger than
|hi| is excluded" -- not "there is no difference".

Inputs are already-measured per-run values, so this loads no models:
  perp_logit, perp_perp, perp_geom   from results/adversarial_checks.csv
    the three normalizations of the frequency-only (perpendicular) type margin;
    they fail differently, which is why all three are carried through.

Three things, each answering a distinct objection:

1. BOOTSTRAP CI on the difference of class means, resampling runs within class.
   Also Hedges' g, since a raw difference in normalized margin units is hard to
   judge without a scale.

2. PERMUTATION TEST of the same difference, shuffling class labels. Reported for
   completeness; a large p here is NOT evidence of equality, which is exactly why
   the interval above is the thing to quote.

3. WITHIN-ORDER paired difference. Class composition varies by group order
   (D_59 is all A, D_63 mostly B), so a pooled difference confounds class with
   order. Take the A-minus-B difference inside each order that has both, then
   average those. Orders contributing a single run per class are listed with
   their k so nobody reads a one-versus-one cell as a measurement.

Run: python -m instruments.margin_equivalence_interval
"""
import numpy as np

from instruments.results_io import load, save

METRICS = [("perp_logit", "logit sd (original)"),
           ("perp_perp", "logit sd, bit removed"),
           ("perp_geom", "pure geometry")]
BOOT = 20000
PERM = 20000


def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / (sp + 1e-12)
    j = 1 - 3 / (4 * (na + nb) - 9)          # small-sample correction
    return float(d * j)


def boot_ci(a, b, rng, iters=BOOT, lo=2.5, hi=97.5):
    d = np.array([rng.choice(a, len(a), replace=True).mean() -
                  rng.choice(b, len(b), replace=True).mean() for _ in range(iters)])
    return float(np.percentile(d, lo)), float(np.percentile(d, hi))


def perm_p(a, b, rng, iters=PERM):
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    null = np.empty(iters)
    for i in range(iters):
        p = rng.permutation(pool)
        null[i] = abs(p[:na].mean() - p[na:].mean())
    return float((null >= obs).mean())


def main():
    rows = load("adversarial_checks")
    rng = np.random.default_rng(0)

    print("  Class A - class B difference in the frequency-only type margin.")
    print("  The INTERVAL is the claim; the permutation p is context, not evidence")
    print("  of equality.\n")
    print(f"  {'normalization':24} {'A mean':>8} {'B mean':>8} {'A-B':>7} "
          f"{'95% CI':>18} {'g':>6} {'perm p':>7}")

    out = []
    for key, label in METRICS:
        a = np.array([r[key] for r in rows if r["cls"] == "A"])
        b = np.array([r[key] for r in rows if r["cls"] == "B"])
        lo, hi = boot_ci(a, b, rng)
        g = hedges_g(a, b)
        p = perm_p(a, b, rng)
        print(f"  {label:24} {a.mean():>8.3f} {b.mean():>8.3f} "
              f"{a.mean() - b.mean():>+7.3f} "
              f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>18} "
              f"{g:>+6.2f} {p:>7.3f}")
        out.append(("pooled", key, len(a), len(b), f"{a.mean():.6f}",
                    f"{b.mean():.6f}", f"{a.mean() - b.mean():.6f}",
                    f"{lo:.6f}", f"{hi:.6f}", f"{g:.6f}", f"{p:.6f}"))

    print("\n  WITHIN GROUP ORDER (only orders holding both classes), A - B")
    orders = sorted({int(r["n"]) for r in rows})
    for key, label in METRICS:
        diffs, detail = [], []
        for n in orders:
            a = np.array([r[key] for r in rows if r["cls"] == "A" and int(r["n"]) == n])
            b = np.array([r[key] for r in rows if r["cls"] == "B" and int(r["n"]) == n])
            if not len(a) or not len(b):
                continue
            diffs.append(a.mean() - b.mean())
            detail.append(f"D_{n} {a.mean() - b.mean():+.2f} (k={len(a)}/{len(b)})")
            out.append((f"D_{n}", key, len(a), len(b), f"{a.mean():.6f}",
                        f"{b.mean():.6f}", f"{a.mean() - b.mean():.6f}",
                        "", "", "", ""))
        d = np.array(diffs)
        # the mean of within-order differences, and how often it even keeps its sign
        print(f"  {label:24} mean {d.mean():+.3f} over {len(d)} orders, "
              f"sign consistent in {int(max((d > 0).sum(), (d < 0).sum()))}/{len(d)}")
        print(f"  {'':24} {'; '.join(detail)}")

    save("margin_equivalence_interval",
         "scope,metric,k_A,k_B,mean_A,mean_B,diff,ci_lo,ci_hi,hedges_g,perm_p", out)

    print("\n  READ: quote the pooled CI. If it excludes differences larger than the")
    print("  effect the paper would care about, the equality claim is supported; if")
    print("  it does not, the honest sentence is that the data cannot resolve it.")


if __name__ == "__main__":
    main()
