# The population of record

**The population of record (fixed 2026-08-04).**

Every run that reached the >= 0.99 validation-accuracy generalization rule, with
its group order and its measured type-variable class. 35 runs of 47 trained,
spanning seven group orders (D_49 trained 3, generalized 0, so it is absent).

Class labels come from two instruments:
  lin_out : linear decode of output type from the MLP hidden layer  
  auc_t   : zero-parameter readout along t = mean_c(U[r_c] - U[s_c])  
"A" = lin_out 100 and auc_t 1.000. "B" = auc_t ~0.50. "I" = the two runs that sit
between the modes (d47_seed6: 79 / 0.683; d61_seed4: 85 / 0.663). They are NOT
silently binned into A or B.

CORRECTION 2026-08-13. This file used to say the two instruments "agree on every
run". They do not, and the earlier claim was an artifact of scope: lin_out had
only ever been measured on 23 runs, all of which happened to be clean. Measured on
all 35 (results/lin_out_all_seeds.csv):
  auc_t   is sharply split -- A exactly 1.000 (16 runs), B 0.505 [0.500, 0.548]  
  lin_out is not -- A exactly 100% (16 runs), B 56.1% [45%, 70%], with five B runs  
          above 60%: d63_seed4 70, d63_seed5 69, d63_seed3 69, d63_seed7 62,  
          d61_seed6 60. All five are dense-sweep runs, i.e. runs the old 23-run  
          scope never touched.  
The labels themselves are unchanged, because auc_t (and the fitted final-residual
probe in adversarial_checks, class B 50.0% [49, 51]) is what defines them. What is
no longer true is that the MLP hidden layer carries no type information in class B:
in some B runs it carries a decodable amount, well above the 50% chance level,
while the same models' final residual sits at chance. Any sentence in the paper of
the form "the probe is at chance in class B" must name the site it refers to.

Provenance note (corrected 2026-08-28): SIXTEEN (order, seed) pairs exist in
BOTH the prime/composite and dense sweeps -- d47 seeds 0-4, d61 seeds 0-7,
d63 seeds 0-2 -- and every pair is bit-identical at every shared checkpoint
epoch (verify_twin_runs.py), i.e. one deterministic trajectory recorded twice
at different checkpoint densities and horizons. Each pair appears in RUNS at
most once, under the longer-horizon path (d61_seed3 generalized in neither
sweep, so it is absent from RUNS). Any analysis that
also uses the dense D_63 sweep as a replication set must exclude seeds 0-2, which
are already in this population; the genuinely out-of-sample D_63 seeds are 3-9.

Import RUNS (or the helpers) rather than re-deriving a run list per script.

This file is generated from `instruments/population.py`. Import `RUNS` from there
rather than re-deriving a run list per script; the table below is that list.

## The 35 runs

| run | order | class | sweep |
|---|---|---|---|
| `d47_seed0` | 47 | A | `prime_composite_sweep` |
| `d47_seed1` | 47 | A | `prime_composite_sweep` |
| `d47_seed3` | 47 | A | `prime_composite_sweep` |
| `d47_seed4` | 47 | B | `prime_composite_sweep` |
| `d47_seed6` | 47 | I | `dense_checkpoint_sweep` |
| `d47_seed7` | 47 | A | `dense_checkpoint_sweep` |
| `d47_seed8` | 47 | A | `dense_checkpoint_sweep` |
| `d47_seed9` | 47 | A | `dense_checkpoint_sweep` |
| `d51_seed0` | 51 | B | `prime_composite_sweep` |
| `d51_seed2` | 51 | A | `prime_composite_sweep` |
| `d53_seed1` | 53 | A | `prime_composite_sweep` |
| `d53_seed2` | 53 | B | `prime_composite_sweep` |
| `d55_seed0` | 55 | B | `prime_composite_sweep` |
| `d55_seed2` | 55 | B | `prime_composite_sweep` |
| `d59_30pct_seed0` | 59 | A | `seed_sweep` |
| `d59_30pct_seed1` | 59 | A | `seed_sweep` |
| `d59_30pct_seed3` | 59 | A | `seed_sweep` |
| `d59_30pct_seed4_resumed` | 59 | A | `seed_sweep` |
| `d61_seed0` | 61 | B | `prime_composite_sweep` |
| `d61_seed1` | 61 | B | `prime_composite_sweep` |
| `d61_seed2` | 61 | A | `prime_composite_sweep` |
| `d61_seed4` | 61 | I | `prime_composite_sweep` |
| `d61_seed6` | 61 | B | `prime_composite_sweep` |
| `d61_seed8` | 61 | A | `dense_checkpoint_sweep` |
| `d61_seed9` | 61 | B | `dense_checkpoint_sweep` |
| `d63_seed0` | 63 | B | `prime_composite_sweep` |
| `d63_seed1` | 63 | B | `prime_composite_sweep` |
| `d63_seed2` | 63 | A | `prime_composite_sweep` |
| `d63_seed3` | 63 | B | `dense_checkpoint_sweep` |
| `d63_seed4` | 63 | B | `dense_checkpoint_sweep` |
| `d63_seed5` | 63 | B | `dense_checkpoint_sweep` |
| `d63_seed6` | 63 | B | `dense_checkpoint_sweep` |
| `d63_seed7` | 63 | B | `dense_checkpoint_sweep` |
| `d63_seed8` | 63 | A | `dense_checkpoint_sweep` |
| `d63_seed9` | 63 | B | `dense_checkpoint_sweep` |

## Trained but did not generalize

Kept so the denominator is derived from the population file rather than re-counted
by hand. Best validation accuracy in parentheses; the rule is 0.99.

| order | trained | generalized | did not |
|---|---|---|---|
| 47 | 10 | 8 | seed2 (0.9895), seed5 (0.7732) |
| 49 | 3 | 0 | seed0 (0.9221), seed1 (0.9769), seed2 (0.7803) |
| 51 | 3 | 2 | seed1 (0.9883) |
| 53 | 3 | 2 | seed0 (0.9626) |
| 55 | 3 | 2 | seed1 (0.9353) |
| 59 | 5 | 4 | seed2 (0.9645 resumed) |
| 61 | 10 | 7 | seed3 (0.9826), seed5 (0.9329), seed7 (0.9880) |
| 63 | 10 | 10 | -- |

Total: 47 trained, 35 generalized.
