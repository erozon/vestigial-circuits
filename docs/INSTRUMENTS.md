# Instruments

One entry per measurement, generated from the module docstrings.
Run any of them with `python -m instruments.<name>`; each writes
`results/<name>.csv` and prints the table it read the numbers off.

Set `DIHEDRAL_RUNS` to wherever the released checkpoints were unpacked.

## Run order

Some instruments read another's CSV. A consumer run too early does not fail: it
leaves the joined column blank, or reads a stale file, and says so on the console.
Run each producer below before its consumers. This list is derived from the
sources, not maintained by hand.

- `figure_deflation` needs `edit_error_profile`
- `figure_fork` needs `index_conc_population`, `init_bit_baseline`, `type_structure_spectrum`
- `figure_lifecycle` needs `bit_emergence_trajectory`, `converged_type_ablation`, `early_bit_ablation`, `matched_acc_bit_ablation`
- `init_bit_baseline` needs `bit_emergence_trajectory`, `type_structure_spectrum`
- `margin_equivalence_interval` needs `adversarial_checks`
- `matched_edit_error_profile` needs `matched_acc_bit_ablation`
- `population_rate_analysis` needs `bit_emergence_trajectory`
- `timeline_medians` needs `bit_emergence_trajectory`, `early_bit_ablation`
- `unembed_type_direction_readout` needs `lin_out_all_seeds`
- `unembed_type_geometry` needs `lin_out_all_seeds`

## Which archive each one needs

The instruments below read per-epoch checkpoints or a whole sweep directory, so they need
the trajectory archive. Every other instrument runs from the final-checkpoint
archive alone. This list is derived from the sources.

- `bit_emergence_trajectory`
- `converged_type_ablation`
- `early_bit_ablation`
- `early_edit_error_profile`
- `matched_acc_bit_ablation`
- `matched_edit_error_profile`
- `verify_twin_runs`

## `population`

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

## `results_io`

**Persist instrument output so figures and prose read numbers, not stdout. (2026-08-13)**

Most instruments in this directory printed their measurements and discarded them.
That was fine while the numbers were being read once, in a terminal, by a person
deciding what to measure next. It is not fine now: every figure in the paper has
to be built from a measurement, and rebuilding a figure should not mean re-running
a 35-model sweep and re-parsing a console dump.

The convention: an instrument keeps its printed table exactly as it was -- that is
still the thing a person reads -- and additionally calls save() with one row per
run. Nothing about what is measured changes.

    from instruments.results_io import save  
    save("dc_surgical_ablation", "run,n,cls,base,surgical", rows)  

writes results/dc_surgical_ablation.csv, where rows is a
list of tuples matching the header. Aggregates (class means and so on) are NOT
written: they are one groupby away from the per-run rows, and storing a derived
number invites it drifting from the rows it came from.

Read back with load(), which returns a list of dicts with numeric fields already
converted -- enough for plotting without a pandas dependency.

## `index_conc_population`

**Is the index circuit the same in all 35 generalized models? (2026-08-05)**

The paper's answer-key claim -- every generalized model computes the product's
index the same way -- was until now supported only by d59_basis_test.py, which
hardcodes N=59 and SEEDS=[0,1,3]. Three runs at one group order cannot carry a
claim about 35 runs at seven, and in particular cannot answer the question the
whole argument leans on: whether class A and class B compute the INDEX alike.
If they differed there too, "both classes reach the answer by the same means"
does not hold and the paper changes shape.

This generalizes the d59 measurement over the population of record. Same metric
(rung0-certified `conc`), same four-quadrant reduction at fixed g1, same a0
robustness sweep -- only the model loop is new.

Reported per run, and then aggregated BY CLASS, which is the comparison that
matters:
  medConc   median top-1 folded spectral concentration over neurons (chance for  
            a random length-n vector is ~0.14; a clean single tone is ~1.0)  
  %>=0.5    fraction of neurons that are cleanly single-frequency  
  #freq     how many distinct dominant frequencies the clean neurons use  
  instab    mean sd of conc across the a0 offsets; conc is phase-blind, so a  
            clean tone should be near-invariant and instability flags non-tones  

Run: python -m instruments.index_conc_population

## `unembed_type_geometry`

**Output-side geometry of the type distinction (2026-08-04).**

Every measurement of the localized/non-localized fork so far lives in the MLP
hidden layer. This one looks at the OTHER end: the unembedding rows, which is
what actually decides the answer.

At the result position the choice between the two tokens sharing index c is
governed entirely by

    D_c = U[r_c] - U[s_c]  

The answer is a rotation at index c iff the residual has a positive component
along D_c. So:

  * if the D_c are near-PARALLEL across c, there is ONE global "rotation-ness"  
    direction in output space. A single hidden feature writing to it raises every  
    rotation logit and lowers every reflection logit at once -- a type bit is  
    worth computing, and a linear probe of the hidden layer will find it.  
  * if the D_c point in DIFFERENT directions per c, no such global direction  
    exists. A feature pushing toward rotation at one index pushes toward  
    reflection at another; nothing is gained by holding a separate type bit, and  
    the network can only emit the right token directly.  

PREDICTION under that account: common_frac (below) is HIGH in runs where output
type is globally linearly decodable from the hidden layer (lin_out ~100) and LOW
in runs where it is at chance (lin_out ~50). If both classes show the same output
geometry, the account is WRONG and the fork lives elsewhere.

Metrics per run (all scale-invariant, weights only -- no forward passes):
  common_frac : ||mean_c D_c||^2 / mean_c ||D_c||^2, in [0,1]. Fraction of the  
                per-index type-difference energy carried by the SHARED component.  
                Chance reference for unstructured rows is ~1/n (see printout).  
  mean_cos    : mean_c cos(D_c, mean D). 1 = every index separates its two tokens  
                along the same direction; ~0 = index-specific.  
  min_cos     : the worst index (does ANY index disagree in sign?).  
  sv1_frac    : top singular value share of D -- how close D is to rank 1  
                (rank-1 = one shared axis, possibly with per-index scaling).  
  sv1_cos_t   : |cos| between the top singular direction and the shared component  
                (separates "one axis, sign flips per index" from "one direction").  
  bias_gap    : if the output layer has a bias, mean_c(b_r_c - b_s_c) normalized  
                by its own std across c -- a constant type offset can also live  
                here rather than in the rows.  

SCOPE FIX (2026-08-13): this ran over a glob of the prime/composite sweep plus
three D_59 seeds, filtered by a hardcoded LIN_OUT dict copied from a 2026-08-04
run of lin_out_all_seeds -- 23 runs, and a table of numbers that could silently
fall out of date. It now iterates population.RUNS (35 runs) and reads lin_out from
results/lin_out_all_seeds.csv if that instrument has been run, leaving it blank
otherwise. The correlation against lin_out is computed on whichever runs have it.

Run: python -m instruments.unembed_type_geometry

## `unembed_type_direction_readout`

**Zero-parameter test of the shared type direction (2026-08-04).**

unembed_type_geometry showed the unembedding's per-index type-difference vectors
D_c = U[r_c] - U[s_c] carry a large SHARED component in runs where output type is
linearly decodable from the MLP hidden layer, and none (at/below the ~1/n chance
level) in runs where it is not.

That was weights-only. This closes the loop to actual computation WITHOUT fitting
anything: take the shared direction

    t = mean_c ( U[r_c] - U[s_c] )        [read straight off the weights]  

project each input's final residual (the exact vector fc_out consumes) onto t, and
ask whether that scalar alone tracks the answer's type. No probe, no training, no
free parameters -- AUC needs no threshold at all.

  class A (lin_out ~100) : AUC ~1.0  => the weight-derived direction IS the type  
                           feature; the earlier probe was not doing the work.  
  class B (lin_out ~50)  : AUC ~0.5  => nothing to read along any shared direction.  

Reported alongside:
  acc@thr : accuracy of sign(<resid,t> - thr) at the best threshold (ONE fitted  
            scalar, for interpretability only; AUC is the headline).  
  model_acc : full-model accuracy on the same inputs (sanity).  
  ffn_share : fraction of <resid,t> variance contributed by the FFN's write  
            (vs the attention+embedding path) -- says WHICH component drives the  
            projection in runs where it works.  

SCOPE FIX (2026-08-13): same change as unembed_type_geometry -- this ran over a
23-run glob gated by that module's hardcoded LIN_OUT dict, which no longer exists.
It now iterates population.RUNS (35 runs) and reads lin_out from
results/lin_out_all_seeds.csv when available. auc() and best_acc() are imported by
several other instruments and are unchanged.

Run: python -m instruments.unembed_type_direction_readout

## `sign_readout`

**Does the SIGN of t_hat . x give the type, with the threshold fixed at zero?**

(2026-08-16)

unembed_type_direction_readout.py reports AUC, which is threshold-free, and
acc_thr, which picks the best threshold by scanning -- one fitted scalar. Neither
licenses the sentence "the sign of t.x gives the type", because AUC = 1.000 says
only that SOME threshold separates the two types perfectly. It could sit far from
zero, in which case the readout needs a fitted offset after all and the paper must
say "a threshold on t.x" instead of "the sign of t.x".

This settles it. Same construction as the AUC instrument --

    t_hat = sum_c (U[r_c] - U[s_c]) / || . ||     [weights only]  
    beta  = t_hat . x                             [x = final residual, fc_out input]  

-- but scored at the fixed threshold zero, with no fitting anywhere. Reported:

  acc_sign0 : accuracy of sign(beta) against the true output type. Orientation is  
              resolved by taking the better of the two global sign conventions,  
              which is a choice of labelling, not a fitted parameter.  
  sep_thr   : midpoint of the separating gap when the classes are linearly  
              separable along beta, else blank. This is where the perfect  
              threshold actually sits.  
  thr_sd    : sep_thr in units of the sd of beta -- how far the ideal threshold is  
              from zero on the scale of the data.  

Result (2026-08-16, all 35 runs): class A is 1.0000 in 15 of 16 runs and 0.9999 in
d59_30pct_seed4_resumed, which misses one input of 13924. The separating threshold
sits within 0.38 sd of zero in every class A run. So the sign alone carries it.
Class B averages 0.517 with one run at 0.616; report class B by AUC (0.505), not
by this statistic, since sign accuracy at a fixed threshold can exceed the AUC
when the two score distributions differ in spread.

Run: python -m instruments.sign_readout

## `lin_out_all_seeds`

**lin_out decodability across ALL grokked seeds -- the bimodality check (2026-07-13).**

The robust "localized vs non-localized" fork (output type globally linearly
decodable from the MLP hidden: 100% vs chance) was established on only 6 hand-picked
models, while the "23-seed bimodality" claim lives on the COARSER override/k_loc
metric. This runs the SHARP metric (lin_out) on the SAME grokked population the
override analysis used, to settle: is lin_out actually BIMODAL across seeds, or a
binarized continuum?

For every grokked checkpoint (base acc >= 0.99), on the MLP hidden layer over the
full dataset:
  lin_out : linear decode of OUTPUT type (=op1 XOR op2)  -- the sharp fork metric  
  lin_op1 : linear decode of operand-1 type              -- input feature (context)  
  lin_op2 : linear decode of operand-2 type  
  override: the coarse sign-gate localization metric      -- for cross-check  

Bimodal => lin_out clusters near 100 and near 50 with a gap => fork is real on the
real population. Spread across 60/70/80 => continuum => reframe to "degree of
type-linearization".

SCOPE FIX (2026-08-13): this originally ran over a glob of the prime/composite
sweep plus three hand-listed D_59 seeds -- 23 runs, missing every run that lives
only in the dense checkpoint sweep. That was the population before population.py
existed, and it is not the population the paper reports on. It now iterates
population.RUNS, so lin_out is measured on all 35 generalized runs rather than on
whichever subset happened to be globbed. The class labels in population.py were
set partly from the old subset; the printed comparison against `cls` is therefore
worth reading as a check on those labels, not as a foregone agreement.

Run: python -m instruments.lin_out_all_seeds

## `dc_component_ablation`

**Is the index-independent type direction load-bearing, or redundant? (2026-08-05)**

Established so far (weights only, all 35 runs):
  - The output layer's type-discriminating vectors D_c = U[r_c] - U[s_c] carry  
    ~72% of their energy in three nonzero frequencies of the index, in BOTH  
    classes, and those are the same frequencies the index circuit uses  
    (overlap 2.7/3 vs chance 0.3; spectrum correlation 0.92-0.95).  
  - The classes differ at k=0 and essentially nowhere else: class A 0.155 of  
    total energy, class B 0.015 against a chance level of ~0.017.  

So class A has an index-independent type direction and class B has none. This
instrument asks what class A's costs it to lose.

Ablation: t = mean_c D_c, normalized. Remove that direction from EVERY row of the
output layer, U' = U - (U t_hat) t_hat^T. This destroys the index-independent
type discriminant while leaving all nonzero-frequency structure intact. Nothing is
retrained; the rest of the network is untouched.

Reported before and after, over all n^2 products:
  acc   : exact token accuracy  
  idx   : predicted index == true index (ignoring type)  
  typ   : predicted type == true type (ignoring index)  

Reading, stated before running:
  - class A accuracy survives => the index-independent direction is a redundant  
    copy of a distinction the frequency structure already makes.  
  - class A type accuracy collapses while index accuracy survives => that  
    direction is what class A actually uses to decide type, and class B must do  
    something else.  
Class B runs are the control: removing a direction that carries ~nothing should
change ~nothing.

Run: python -m instruments.dc_component_ablation

## `dc_surgical_ablation`

**Surgical removal of the type bit, correcting a flaw in the crude ablation.**

(2026-08-05)

dc_component_ablation.py removed the ENTIRE direction t = mean_c(U[r_c] - U[s_c])
from every row of the output layer. That cost class A ~11% accuracy, but the
margin decomposition then showed the frequency-only margin against the true
token's type partner is negative on just 0.1% of products. So the 11% was not
type confusion.

The reason is that the crude ablation removed too much. Writing each row's
projection onto t as

    U[r_c] . t_hat = m_r + delta_c        U[s_c] . t_hat = m_s + eps_c  

the TYPE BIT is the constant contrast (m_r - m_s). The per-row deviations
delta_c, eps_c vary with the index and carry index information; the crude
ablation destroyed those too, reshuffling rankings among tokens of the SAME type.
That is why 99% of its errors had both index and type wrong.

This removes only the constant contrast, leaving every per-row deviation intact:

    U'[r_c] = U[r_c] - (m_r - m) t_hat      U'[s_c] = U[s_c] - (m_s - m) t_hat  
    where m = (m_r + m_s) / 2  

Conditions reported:
  base     : untouched  
  surgical : type-constant contrast removed, per-row deviations kept  
  crude    : the whole direction removed (the earlier, flawed ablation)  
  shuffled : the SAME per-row offsets as surgical, but assigned to rows in a  
             random permutation that ignores type. Same magnitude of perturbation,  
             no type structure -- if surgical hurts and shuffled does not, the  
             damage is specifically about the type contrast.  

Reading, stated before running: if class A survives the surgical ablation, its
explicit type bit is redundant -- the frequency mechanism it shares with class B
already decides type, and the bit is a second, unnecessary copy.

Run: python -m instruments.dc_surgical_ablation

## `type_structure_spectrum`

**Where does the type-discriminating structure sit as a function of index? (2026-08-05)**

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

## `type_frequency_alignment`

**Do the type-carrying frequencies coincide with the index-circuit frequencies?**

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

## `type_margin_decomposition`

**How is the type margin split between the shared direction and the frequencies?**

(2026-08-05)

Chain so far. The output layer's type-discriminating vectors D_c = U[r_c] - U[s_c]
are strongly structured in BOTH classes and sit on the same frequencies as the
index circuit. Class A additionally has an index-independent component t =
mean_c D_c (~15% of energy vs class B's 1.5%, which is chance). Removing t costs
class A ~11% accuracy, but the errors are near-ties -- the true token is still
ranked 2nd in 86% of them -- so t supplies MARGIN rather than the distinction
itself. Class B does the same job with no such direction at all.

Open question this addresses: does class A's frequency-based type discrimination
carry less margin in absolute terms than class B's, with t making up the
difference? If so the fork is about how models SPLIT one job, not about one
having a variable the other lacks.

There is no bias on the output layer, so logits are exactly x @ U^T and the
decomposition is exact. For each product, with c the true index and the partner
being the same index of the other type:

  margin = logit(true) - logit(partner) = s * (x . D_c),   s = +1 if true is a  
                                                           rotation else -1  
  dc_part   = s * (x . t_hat)(D_c . t_hat)  
  perp_part = margin - dc_part  

Skeptical controls, all reported:
  - Margins are normalized per example by the standard deviation of that example's  
    2n element logits, so models with different logit scales are comparable. Raw  
    means are printed too.  
  - rand_share: the same decomposition along a RANDOM unit direction instead of t.  
    Any direction captures some margin by chance; this is that floor.  
  - dc_cv: coefficient of variation of (x . t_hat) within an output type. If t's  
    contribution barely varies with the input, it is acting as a per-type bias  
    rather than as a computed quantity.  
  - Everything split by output type, because the ablation errors were concentrated  
    on rotation outputs (16% vs 6%).  

Run: python -m instruments.type_margin_decomposition

## `margin_equivalence_interval`

**How equal is "indistinguishable"? Intervals on the class A - class B margin gap.**

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

## `bit_emergence_trajectory`

**When does the redundant type bit appear, relative to generalization? (2026-08-05)**

Established: class A models carry a near-constant type bit along one global
direction that contributes 19% of their type margin and is functionally
unnecessary -- the index-bound frequency mechanism, identical in both classes,
already decides type 99.9% of the time on its own.

Open: why class A builds it at all, and whether it matters DURING training even
though it does nothing at inference. Timing is the cheapest evidence available.

Over the dense checkpoint sweep (every 1000 epochs), per checkpoint:
  val      : validation accuracy at that epoch, from metrics.jsonl  
  dc_frac  : ||mean_c D_c||^2 / mean_c||D_c||^2 of the output layer (weights only)  
  auc_t    : zero-parameter readout along t, on a fixed subsample of products  
  m_dc     : mean type margin contributed by the shared direction (normalized)  
  m_perp   : mean type margin contributed by everything else (normalized)  

Readings, stated before running:
  - bit appears AFTER generalization => a byproduct that accumulates once the  
    task is already solved; nothing to do with reaching the solution.  
  - bit appears BEFORE or WITH the frequency margin => it may be scaffolding the  
    model used to get there and then kept.  
  - m_perp reaching full strength before the bit exists in class A would show the  
    frequency mechanism does not need the bit developmentally either.  

Subsampling: 3000 products, fixed across checkpoints and runs of the same order so
trajectories are comparable.

SAMPLING FIX (2026-08-13): this took every 2nd checkpoint throughout, which meant
the interval where the bit forms was sampled at epochs 1000, 3000, 5000 -- two
points supporting the "forms at ~1-3K" claim. Checkpoints exist every 1000 epochs,
so the first EARLY_DENSE_TO epochs are now taken at full resolution and the stride
applies only after that, where the curves are flat and the extra points buy
nothing.

Run: python -m instruments.bit_emergence_trajectory

## `early_bit_ablation`

**Was the bit doing work EARLY, or is the fossil story correlational? (2026-08-05)**

At convergence the bit is inert: removing it costs no more than an equal-magnitude
perturbation that leaves it intact. Early in training class A gets output type
right far more often than class B (0.87 vs 0.67 at epoch 7000) while both sit at
index accuracy 0.30, the training fraction. The story we want to tell is that the
bit EARNS its place early and is superseded later.

That story is currently correlational, and this is the test that can kill it. Take
early checkpoints and remove the type-constant contrast surgically, exactly as at
convergence. Compare against the noise-only control (same magnitude, type contrast
intact) so the comparison is not confounded by perturbation size -- the mistake
that produced a false positive at convergence.

Predictions, stated before running:
  - Fossil story SURVIVES if, early, surgical removal drops type accuracy toward  
    class B's level while noise-only does not.  
  - Fossil story DIES if surgical and noise-only hurt equally early, exactly as  
    they do at convergence. Then the bit never did any work and its correlation  
    with early type accuracy has some other cause.  

Run: python -m instruments.early_bit_ablation

## `matched_acc_bit_ablation`

**Was the bit load-bearing early, or are the classes just at different stages?**

(2026-08-13)

early_bit_ablation.py compares the classes at matched EPOCH (3000, 5000, 7000,
9000) and finds that surgical removal of the type bit costs class A up to 0.289 of
type accuracy while the magnitude-matched control costs 0.008, with class B flat
in every condition. The trouble is that class A groks later -- the within-order
AUC of grok epoch predicting class is 0.74 -- so at epoch 7000 the two classes are
not at the same point in their own training. The finding has a ready alternative
reading: the bit is not doing work, class A is simply earlier in a trajectory that
every model passes through, and we are comparing an early class A against a
late-ish class B.

This removes the confound by indexing checkpoints on VALIDATION ACCURACY instead
of on epoch. For each run, the checkpoint closest to each target accuracy is
selected (subject to TOL, else that cell is empty), and the identical
surgical-versus-noise contrast is applied there.

  surgical : type-constant contrast removed along t = mean_c(U[r_c] - U[s_c])  
  noise    : same per-row magnitude along t, balanced within type, so the type  
             contrast survives intact. The control that matters -- it is what  
             showed the convergence-time effect to be perturbation size.  

Predictions, stated before running:
  - If the gap survives at matched accuracy, the stage confound is dead and the  
    bit really was doing work at a point where class B, at the SAME competence,  
    was doing that work some other way.  
  - If the gap collapses once accuracy is matched, then the epoch-indexed result  
    was a training-stage artifact and Section 3.4 needs rewriting around it.  

The matched epochs themselves are reported per class, since the size of the
confound being corrected is worth seeing rather than asserting.

Run: python -m instruments.matched_acc_bit_ablation

## `init_bit_baseline`

**Is the type bit really absent at initialization? (2026-08-13)**

Section 3.4 says the bit is absent at initialization and built during training --
the claim that rules out a lottery-ticket reading, in which the direction is
present in the random weights and training merely selects models that already had
it. The evidence so far is the first available checkpoint, at epoch 1000. That is
1000 epochs of full-batch training, not initialization, so as measured the claim
was an extrapolation.

Nothing needs training to fix this: the initialization scheme is deterministic
given a seed, so fresh models can be built and measured directly. For each group
order in the population, SEEDS models are constructed exactly as src/train.py
constructs them (Nanda init, same shape) and measured with the same weights-only
statistics used at convergence:

  dc_frac  : ||mean_c D_c||^2 / mean_c||D_c||^2, D_c = U[r_c] - U[s_c]  
  mean_cos : mean_c cos(D_c, mean_c D_c)  
  med_conc : the index measurement of index_conc_population.py, run on the same  
             fresh models. Added 2026-08-28 so that untrained models can be  
             plotted beside the trained ones on both axes at once, which is what  
             the fork figure does instead of drawing a shaded reference band.  

Both are reported against the two things they have to be compared with: the
epoch-1000 values from the trajectory, and the converged class A and class B
values. This also supplies the chance reference the bimodality figure should be
drawn against, which otherwise has to be asserted as ~1/n.

Reading, stated before running:
  - init dc_frac at the ~1/n unstructured level in every seed => the direction is  
    built, and no model starts with it.  
  - a heavy right tail at init, overlapping class A's converged range => the  
    lottery-ticket reading is live and Section 3.4 has a problem.  

Note on what this can and cannot settle: agreement between init and chance rules
out the direction being PRESENT at init. It does not rule out some other property
of the initial weights predisposing a run toward building it -- that is
init_geometry_predictor.py, and it found no predictor.

Run: python -m instruments.init_bit_baseline

## `population_rate_analysis`

**How should the per-order rate variation be reported? (2026-08-05)**

The class A rate varies a lot by group order -- D_59 4/4, D_47 6/8, D_63 2/10 --
but several orders contribute only two runs, so eyeballing the table invites
over-reading. This decides what can honestly be said, rather than leaving the
question to prose.

Three things, in the order a reviewer would ask them:

1. RATES WITH UNCERTAINTY. Wilson score intervals per order. With two runs the
   interval is nearly the whole unit interval, which is the point.  

2. IS THE VARIATION REAL? Permutation test. Statistic = Pearson chi-square of the
   order-by-class table; the null shuffles class labels across all runs, which  
   preserves both the per-order run counts and the overall class balance. Reported  
   two ways: excluding the two intermediate runs, and counting them as non-A, so  
   the answer does not hinge on how they are binned.  

3. IS IT REALLY ABOUT GROUP ORDER, OR ABOUT TIME? Class A runs grok later, and the
   orders differ in how long they take to grok, so "order" and "slow grokking" are  
   confounded. Within each order (which holds group size fixed), report the AUC of  
   grok epoch predicting class. Consistent within-order AUC above 0.5 would say the  
   association is with training time, not with the group itself. Uses the dense  
   sweep, the only runs with per-epoch checkpoints.  

Run: python -m instruments.population_rate_analysis

## `adversarial_checks`

**Adversarial checks on the redundant-bit finding (2026-08-05).**

Written to attack our own result, not to confirm it. Four checks:

CHECK 1 -- NORMALIZATION ARTIFACT.
  The claim "class A's frequency-based type margin equals class B's" (2.16 vs 2.15)  
  normalized each margin by that model's logit standard deviation. But class A's  
  logits INCLUDE the bit's contribution, which inflates its denominator and  
  deflates its reported frequency margin. The agreement could be manufactured.  
  Three normalizations, which fail differently:  
    n_logit : sd of the model's own element logits            (the original)  
    n_perp  : sd of logits recomputed with t removed from U   (bit excluded)  
    n_geom  : ||x|| * mean_c||D_c||, pure geometry, no logits at all  
  If the classes stay equal under all three, the result is not a normalizer effect.  

CHECK 2 -- IS THE ZERO-PARAMETER READOUT TOO WEAK?
  "Class B has no global type direction" comes from projecting onto t, which is  
  fixed in advance. A reviewer will say a FITTED direction would find one. So fit  
  a linear probe on the FINAL RESIDUAL (the vector that feeds the output layer --  
  the strongest place to look) with a train/test split, and report held-out  
  accuracy. If class B sits at chance with a fitted probe, the absence is real.  

CHECK 3 -- POOLING ACROSS GROUP ORDERS.
  Class composition differs by order (D_47 is mostly class A, D_63 mostly class B),  
  so a pooled comparison could manufacture equality. Report per order.  

CHECK 4 -- IS IT ACTUALLY BIMODAL?
  The classes are threshold-defined (auc >= 0.9 vs <= 0.7). Report the sorted  
  distributions of auc_t and dc_frac over all 35 runs with the largest gap, so a  
  continuum would be visible rather than hidden by the binning.  

Run: python -m instruments.adversarial_checks

## `residual_probe_robustness`

**Is class B's 50% at the final residual the model's property or the optimizer's?**

(2026-08-13)

The claim "nothing linear in the final residual predicts the output type in class
B" rests on adversarial_checks.py CHECK 2, which fits a logistic probe with Adam
(lr 0.05, 1000 steps) on RAW residual features -- no standardization, no
regularization, no learning-rate search. The hidden-layer probe in
lin_out_all_seeds.py standardizes its inputs; this one does not. Class A reaches
100% either way, but class A is an easy target: one direction, enormous margin. A
weak signal on badly-scaled features is exactly what an unregularized fixed-step
optimizer fails to find, and it would report chance while doing so.

So: same site, same labels, five fits of increasing strength. If class B stays at
chance through all of them, the absence is the model's.

  raw_adam  : the original -- raw features, Adam lr 0.05, 1000 steps  
  std_adam  : identical but standardized features  
  std_sweep : standardized, best held-out accuracy over lr in {0.01, 0.05, 0.2}  
              at 3000 steps (optimistically biased ON PURPOSE -- picking the best  
              of several fits inflates the number, which is the direction that  
              makes a chance result meaningful)  
  ridge     : closed-form least squares on +-1 labels, standardized, small ridge.  
              No optimizer at all, so it cannot underfit.  
  mlp       : one hidden layer of 64 units. Not a linear probe -- it answers the  
              different question of whether the type is recoverable at this site  
              AT ALL, which a reviewer will ask.  

Reading, stated before running:
  - class B at chance under ridge and the sweep => the linear absence is real and  
    not an artifact of how the probe was fit.  
  - class B rising under any of them => adversarial_checks CHECK 2 understated the  
    signal and every sentence resting on it needs revising.  
  - mlp well above linear in class B => the type IS present at the residual but  
    not linearly, which is a different claim from "not present" and would have to  
    be stated that way.  

Run: python -m instruments.residual_probe_robustness

## `perturbation_sensitivity`

**Are class A models generically more fragile, or is the difference an artifact?**

(2026-08-05)

dc_surgical_ablation.py found that removing class A's type bit costs ~12%
accuracy -- but so does a perturbation of the same magnitude that leaves the type
bit perfectly intact (noise-only, 0.869 vs surgical 0.884). So the damage is not
attributable to the type bit.

That leaves one confound. Class A has a large offset along t to remove; class B's
is ~0. "Class B is robust" may only mean that nothing was done to it. This applies
a MATCHED perturbation to both classes and compares.

Perturbation: balanced +/- offsets along a random unit direction v, applied to the
output-layer rows, scaled so that the induced change in logits is a target
fraction f of that model's own logit spread:

    delta = f * sd(logits) / mean|x . v|  

so f is comparable across models regardless of weight or logit scale. Random v
rather than t, because t is exactly the direction the classes differ in, and using
it would re-introduce the confound.

Reading, stated before running:
  - Both classes degrade alike => class A's apparent fragility was an artifact of  
    perturbation size, and robustness is not part of the fork.  
  - Class A degrades faster at matched f => class A really is the more brittle  
    solution, which is a genuine difference and worth reporting.  

Run: python -m instruments.perturbation_sensitivity

## `shuffle_analysis`

**Analyze the label-shuffle control models (conc + lattice).**

Falsifier for the whole instrument: a pure memorizer (trained on shuffled
labels) must NOT show clean single-tones or lattice structure. If it does, the
conc/GCR and lattice-tracking results are pipeline artifacts.

Persisted 2026-08-13: the result was previously readable only in a stray
runs/shuffle_control_RESULTS.txt, while the appendix promises it as a control.
Both arms now write to results/shuffle_control.csv.

Run after training:  python -m instruments.shuffle_analysis

## `make_label_shuffle`

**Label-shuffle control data: permute the RESULT token across train lines.**

Destroys the group map (input pair -> product) while preserving the label
distribution: each (g1, g2) keeps its inputs but gets a random result drawn from
the shuffled pool. A model can only MEMORIZE this (no generalizable structure),
so any Fourier/single-tone concentration the analysis reports on it would be a
pipeline artifact rather than learned group computation.

Fixed seed for reproducibility. Only train.txt is shuffled (val stays real ->
val_acc measures generalization, which must sit at chance).

Usage: python -m instruments.make_label_shuffle <in_train> <out_train> [seed]

## `d59_basis_test`

**D_59 binary basis test (conc arm) -- workstream 2, frozen Experiment 1.**

PRIMARY metric = conc (Fourier energy concentration over the rotation index),
per the 2026-07-05 resolution (Amendment 7). C_H is the required SECONDARY and
is not built yet -- this run is the conc arm only.

Reduction: Eric's four-quadrant (fix g1, vary g2 over rotation index, condition
on fixed g1 with a0 robustness sweep), validated on Z_113 + D_29.

Pre-registered outcomes (frozen, do not adjust post hoc):
  (a) single-frequency dominates (high conc, few active freqs) -> GCR-like  
  (b) delta-localization on reflection cosets -> coset-like (needs C_H to confirm;  
      conc alone shows LOW conc / spread as the coset signature)  
  (c) mixed/ambiguous -> descriptive finding, no post-hoc verdict  

Baselines: random-init model (here) + chance ref (random length-59 vector).
Label-shuffle control is the remaining planned baseline -> needs a short
memorization training run; FLAGGED as pending, not run here.

Run:  python -m instruments.d59_basis_test

## `d45_lattice_test`

**D_45 graded lattice-tracking test -- the DISCRIMINATING experiment.**

Pre-registered in PLAN.md "Pre-registered experiment 3" (frozen 2026-07-06 BEFORE
any D_45 frequency was inspected). n=45=3^2*5: 22 folded rotation-index DFT bins
k=1..22, each living on the quotient D_{45/gcd(k,45)}. The proper quotients the
model could compute in are D_{45/d} for d in {3,5,9,15}, footprint = multiples of
d (M_d).

Question: do a grokked model's ACTIVE frequencies track this divisor lattice
(coset-structured), or are they lattice-blind (GCR/Fourier)?

PRIMARY statistic B (Eric's choice): min-p hypergeometric surprise.
  x_d = |A cap M_d|;  p_d = P(X >= x_d) under Hypergeom(N=22, K=|M_d|, draws=m);  
  T_B = -log min_d p_d;  argmin_d = tracked quotient level (the GRADED readout).  
  Significance via uniform-m-subset null (operationalizes "GCR = gcd-blind").  
SECONDARY statistic A (coarse omnibus): T_A = |A cap (M3 U M5)|, exact hypergeom.

Active set A (per seed, frozen): union over four quadrants of dominant k* of
neurons with conc >= 0.5 (certified conc, a0 robustness sweep). m = |A|.

Instrument is SELF-TESTED on synthetic active sets (calibrate()) before any
checkpoint is touched. Run:  python -m instruments.d45_lattice_test

## `converged_type_ablation`

**The same two edits at convergence, in the same units as the early ones. (2026-08-28)**

The lifecycle argument compares the cost of removing the type contrast against a
magnitude-matched control that leaves it intact, early in training and at the
end. Early, that comparison is in TYPE accuracy (early_bit_ablation.py,
matched_acc_bit_ablation.py, chance 0.50). At convergence, the only measurement
of the pair was dc_surgical_ablation.py, which reports OVERALL accuracy.

Those are different quantities, and a figure that puts the early points and the
converged points on one axis is comparing type accuracy against overall accuracy.
This closes the gap: the same two edits, the same type-accuracy measurement, at
each run's final checkpoint.

  base       : untouched  
  surgical   : the type-constant contrast removed, per-row deviations kept  
  noise_only : balanced +/- offsets of the same magnitude along t_hat, which  
               leaves the contrast exactly intact  

Population is the 25 generalized dense-checkpoint runs, so the converged point
belongs to the same runs as the trajectory that leads to it. That is the
comparison the lifecycle claim needs; the 16-run class-A population mean in
Section 3.3 answers a different question, in different units, and is not a
substitute.

Reading, stated before running:
  - the gap closes to nothing => the contrast is superseded, which is the claim.  
  - the gap stays open => the contrast is still load-bearing at convergence, and  
    Section 3.3's overall-accuracy result would need reconciling with this one.  

Run: python -m instruments.converged_type_ablation

## `edit_error_profile`

**What do the Section-3.3 edits actually break? (2026-08-28)**

Table 2 compares the four output-layer edits on OVERALL accuracy alone, where
three of them land within two points of each other. Equal cost is not equal
effect: an edit can cost eleven points by confusing the type, by confusing the
index, or by knocking the prediction off both. Overall accuracy cannot tell
those apart, and the argument in that subsection turns on which it is.

This measures all three accuracies under each edit, plus the composition of the
errors -- right index and wrong type, wrong index and right type, or both wrong
-- on the sixteen class-A runs of the population of record, and on class B for
the comparison the appendix makes.

  base       : untouched  
  crude      : the whole projection onto t_hat removed  
  surgical   : the type-constant contrast removed, per-row deviations kept  
  noise_only : balanced +/- offsets of the same magnitude, contrast left intact  

The type-partner distinction is the one to watch. An edit that makes the model
choose the correct index with the wrong type is an edit that broke the type
decision. An edit whose errors are wrong in both coordinates broke something
else, and its cost in type accuracy is a byproduct.

Run: python -m instruments.edit_error_profile

## `early_edit_error_profile`

**Does the EARLY ablation cost look like a severed type decision? (2026-08-28)**

Section 3.4 calls the contrast load-bearing at epoch 7,000 because removing it
costs 0.24 of type accuracy while the matched control costs 0.006. But that is
the same statistic that reads 0.115 vs 0.014 at convergence, where the paper
calls the contrast inert -- rescued there by edit_error_profile.py, which shows
the converged surgical errors are 99.4% wrong on BOTH axes (generic wreckage)
and only 0.5% right-index-wrong-type (the signature a broken type decision must
produce). The early claim has no such second measurement. This is it.

Same checkpoints as early_bit_ablation.py (epochs 3/5/7/9K, dense sweep), same
edits, but recording what the errors are made of. Because the base model is
still memorizing at these epochs (index accuracy ~ the train fraction), the
vs-truth composition is dominated by errors the base model already makes, so
the load-bearing question lives in the DELTA: over pairs whose type the base
model got right and the edit got wrong (typ_break), did the prediction move to
the type-partner of the base prediction -- same predicted index, flipped type
(partner_frac)? That is what cutting a type wire looks like. The same columns
are computed for the matched control.

Predictions, stated before running:
  - Load-bearing claim SURVIVES if early surgical breakage is large and lands  
    on the type-partner (partner_frac high), while the control breaks little.  
  - Load-bearing claim DIES if early surgical breakage looks like convergence:  
    predictions scattered to unrelated tokens at rates a matched displacement  
    also produces. Then "genuinely load-bearing before generalization" was the  
    same shape artifact the paper warns about, and Section 3.4 must be recut.  

Differences from early_bit_ablation.py, for anyone reconciling numbers: this
measures the full pair table rather than its 3,000-pair subsample, so base and
surgical type accuracies differ from early_bit_ablation.csv in the third
decimal.

Run: python -m instruments.early_edit_error_profile

## `matched_edit_error_profile`

**Task-unit verdict at the stage-confound checkpoints. (2026-08-28)**

early_edit_error_profile.py established task-accuracy parity between the
surgical edit and the matched control at epochs 3/5/7/9K, and Table 2
establishes it at convergence. Between those sites --- the accuracy-matched
checkpoints of matched_acc_bit_ablation.py, val 0.35 to 0.90, median class-A
epochs 16K-29K --- only type units were ever scored, and there the removal
costs up to 0.35. Whether "the task never depends on the contrast" is true
hinges on this window: at val 0.90 the index is mostly right, so a 0.23
type-accuracy cost could surface in task units.

Prediction, stated before running: parity should WEAKEN as validation accuracy
rises, because the pairs whose type the contrast carries increasingly have
correct indices. If surgical task cost exceeds the control's at high val, the
contrast has a genuinely task-load-bearing window and the paper must say
"needed transiently", not "never needed".

Same edits, same stats as early_edit_error_profile.py; checkpoints taken from
matched_acc_bit_ablation.csv (a run enters a cell only if a checkpoint landed
within tolerance of the target, so cell counts vary as in that instrument).

Run: python -m instruments.matched_edit_error_profile

## `timeline_medians`

**The Timeline paragraph's numbers, from a stated rule. (2026-08-29)**

app:lifecycle's Timeline paragraph quoted three kinds of median epoch
(readout saturation, orthogonal margin at 90%, grokking) whose generating
rule was never written down; the grok medians were verified against the
trajectory CSV, the other two were not, and an ad-hoc reconstruction gave
different values (8K and 21K/13K against the quoted 7K and 25K/17K). This
instrument defines the rules and becomes the paragraph's sole source.

All statistics read bit_emergence_trajectory.csv, per run:
  grok epoch     : first checkpoint with val >= 0.99.  
  best-val epoch : the checkpoint with maximal val (earliest on ties) ---  
                   the CSV-grid stand-in for the paper's measurement site.  
  readout sat.   : first checkpoint from which auc_t stays >= 0.99 at every  
                   later checkpoint in the file (class A only; class B's  
                   readout never leaves chance).  
  perp-90        : first checkpoint from which m_perp stays >= 0.9x its  
                   best-val value at every checkpoint up to best-val.  
Sustained crossings, not first touches, so a spike cannot set the date.
Classes from early_bit_ablation.csv; the two unclassified runs are excluded
from medians and reported per run.

Run: python -m instruments.timeline_medians

## `verify_twin_runs`

**Are the doubly trained seeds really one trajectory recorded twice? (2026-08-28)**

Sixteen (order, seed) pairs were trained in BOTH the prime/composite sweep
(checkpoints every 20,000 epochs) and the dense-checkpoint sweep (every 1,000):
D_47 seeds 0-4, D_61 seeds 0-7, D_63 seeds 0-2. The provenance paragraph of the
paper (app:runs) claims each pair is a single deterministic trajectory, so the
two directories must hold bit-identical weights at every checkpoint epoch they
share. This instrument checks exactly that, for every pair and every shared
epoch, by comparing all tensors of the two model_state_dicts.

If any tensor ever differs, the paper's "counted once" accounting is wrong and
the population must treat the copies as distinct runs. Report the failure; do
not paper over it.

Run: python -m instruments.verify_twin_runs

## `figure_fork`

**What splits the population, and what does not. (2026-08-28)**

Two measurements over the same 35 runs, on the two axes of one panel.

  y  median spectral concentration of the index computation  
     (index_conc_population.csv). Every generalized run is high here. This is  
     the axis universality survives on.  
  x  shared-direction strength, the fraction of the D_c vectors' energy carried  
     by their common part (type_structure_spectrum.csv). This is the axis the population  
     splits on.  

The 210 freshly initialized models are plotted too, on both axes
(init_bit_baseline.csv). They are a control, not a reference band: they occupy
the region a model sits in before it has learned anything, which is what makes
the two claims visible at once. Class B sits directly above them -- same
strength, an index circuit built -- rather than merely "low".

Nothing is coloured by class: the classes are defined by the sign readout in
the text, and this figure's role is to show that the geometry divides on its
own. The gap does that work or it does not. The two runs the readout leaves
unclassified (d47_seed6, d61_seed4) are drawn as triangles; on this axis they
sit one on each side of the gap.

Run:  python -m instruments.figure_fork
Writes paper/v2/figures/fork.pdf.

## `figure_deflation`

**Equal cost, different damage. (2026-08-28)**

Ablating the type contrast and displacing the rows by the same amount without
touching it cost the same, break different things, and neither breaks the type
decision. Table 2 reports overall accuracy, which cannot separate those: an
ablation can cost eleven points by confusing the type, by confusing the index,
or by knocking the prediction off both.

So both accuracies, on the two axes of one panel, one point per run per
ablation. Distance from the corner is how much the ablation cost; direction from
the corner is what it cost.

  contrast ablated    on the diagonal -- index and type fall together  
  whole projection    on top of it, which is itself a result  
  matched control     against the top edge -- index falls, type does not  

The bottom right stays empty. Right index and wrong type is what a broken type
decision looks like, and no ablation produces it.

Colour encodes the ablation, which is something we did to the model, not a class
inferred from the same numbers -- so unlike the fork figure it is not the
figure restating its own labelling.

Reads edit_error_profile.csv. Computes nothing new.

Run:  python -m instruments.figure_deflation
Writes paper/v2/figures/deflation.pdf.

## `figure_lifecycle`

**Built before it could multiply, still load-bearing at the end. (2026-08-28)**

(a) shared-direction strength against training time, with each run's clock
      normalized by its own grokking epoch. A run's x = 1 is the moment it  
      generalizes, so "the direction is built before the model can multiply"  
      is one vertical line rather than 25 scattered markers, and runs that  
      grok at wildly different epochs become comparable. The axis is linear:  
      nothing happens in the first tenth, so a log axis gave a third of the  
      panel to a flat stretch and squeezed the plateau after grokking, which  
      is where the direction is kept rather than removed.  
  (b) type accuracy lost to each ablation, against validation accuracy rather  
      than epoch, for the same reason: runs are compared at equal competence.  
      The four early points come from matched_acc_bit_ablation.csv, the final  
      one from converged_type_ablation.csv, which measures the same ablations  
      the same way at each run's last checkpoint.  

Both panels use type accuracy or the geometry that feeds it, never overall
accuracy, so nothing here is compared across units.

Class B is drawn faintly in (b): it has no shared direction to ablate, so both
of its series sit at zero. That is the control, not a finding.

The two runs the sign readout leaves unclassified (d47_seed6, d61_seed4) are in
no class statistic anywhere in the paper. Panel (a) draws them in grey; panel
(b) excludes them from both class means.

Run:  python -m instruments.figure_lifecycle
Writes paper/v2/figures/lifecycle.pdf.

## `das_type_subspace`

**Distributed Alignment Search (DAS) for the output-TYPE variable (2026-07-11).**

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

## `type_xor_probe`

**Is output type an UNLINEARIZED XOR of input types? (2026-07-13)**

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

## `rung3_dihedral`

**Rung 3 of the calibration ladder: smallest non-abelian (D_n), per-quadrant.**

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

## `rung0_synthetic`

**Rung 0 of the calibration ladder: DFT / concentration sanity on KNOWN signals.**

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

## `prime_composite_analyze`

**Analyze the prime-vs-composite sweep: does the one-sided output-type override**

recur in primes and not composites? (2026-07-08)

Runnable anytime -- it analyzes whatever grokked checkpoints exist so far
(sweep anchors D_59/D_45 + runs/prime_composite_sweep/*). For each grokked model:
  override_score = mean(rs,sr acc) - mean(rr,ss acc) after ablating the top-50  
                   neurons ranked by |corr(output-type, preact)|.  
  ~99 => clean rotation-output override (D_59 signature); ~0 => no override.  

Run:  python -m instruments.prime_composite_analyze

## `native_d63_base_rate`

**Native D_63 base rate under the SHARP metrics (2026-08-04).**

Control group for the unembedding-transplant experiment (PLAN.md Experiment 4).
The transplants are 40K-horizon D_63 runs scored by lin_out + the zero-parameter
readout AUC. Their baseline must be measured the SAME way on the SAME kind of run,
or the comparison is unit-mismatched.

The dense checkpoint sweep's ten D_63 seeds are the right control: same group
order, same frozen recipe, same 40K horizon, all ten grokked. They were previously
labelled only by the COARSE override metric (which gave 1 localized / 9 not) --
that number is NOT in the same units as the transplant readout and must not be
used as its baseline.

Per seed:
  best_val  : best val accuracy (grokked iff >= 0.99)  
  lin_out   : linear decode of output type from the MLP hidden layer  
  auc_t     : zero-parameter readout along t = mean_c(U[r_c]-U[s_c]) of that run's  
              OWN unembedding (nothing fitted)  
  common    : ||mean_c D_c||^2 / mean_c||D_c||^2 of its own unembedding  
  override  : the coarse metric, for cross-reference with the older labelling  

Run: python -m instruments.native_d63_base_rate
