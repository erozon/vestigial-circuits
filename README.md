# Presence Is Not Use — code and measurements

Thirty-five one-layer transformers, trained to multiply in dihedral groups
$D_n$ until they generalize, and the instruments that measure what they learned.

Every run computes the modular-arithmetic half of the task with Fourier
circuits, as reported for cyclic groups. The other half — deciding whether the
product is a rotation or a reflection — looks like it splits the population:
sixteen models carry an explicit type direction in their unembedding, a linear
probe reads it perfectly, and deleting it costs eleven points of accuracy, while
the rest carry nothing a probe can find. Both of those measurements mislead. A
magnitude-matched edit that leaves the direction intact costs the same as
deleting it, and both classes decide the type from the same index frequencies at
margins equal as far as we can measure. What the direction is worth to the task
depends on when you ask: nothing against the matched control while the model
still cannot multiply, a real but two-point peak as it climbs to full accuracy,
nothing again at convergence — and it is never removed.

This repository is the measurement layer, not the paper. It exists so that any
per-run number can be re-derived from the released weights in one command.

## On the use of AI

Claude Code wrote most of the code here — the instruments, the release tooling,
the figures — and proposed a large share of the measurements they perform. My
role was closer to a research supervisor's: direction, taste, and judging which
attempts were worth following.

The paper's "On the use of AI" section says more. Errors are mine.

## Quickstart

```bash
pip install -r requirements.txt

# unpack the released final checkpoints anywhere, then point at them
export DIHEDRAL_RUNS=/path/to/checkpoints

python -m instruments.lin_out_all_seeds       # class labels, hidden-layer decode
python -m instruments.dc_component_ablation   # the eleven-point deletion, and its control
git diff results/                             # against the committed numbers
```

Each instrument prints the table a human reads and writes one row per run to
`results/<name>.csv`. Those CSVs are committed: a diff on one is a diff on a
claim. Full recipe in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## What answers what

Anchored on the question rather than a section number, because section numbers
move and nothing outside the LaTeX keeps them honest.

| question | instrument |
|---|---|
| the index is computed on few Fourier frequencies | `index_conc_population` |
| geometry of the type-difference vectors | `unembed_type_geometry` |
| zero-parameter type readout | `unembed_type_direction_readout` |
| the same readout at a fixed zero threshold | `sign_readout` |
| class labels, hidden-layer decode | `lin_out_all_seeds` |
| whole-projection edit, and a random-direction control | `dc_component_ablation` |
| contrast removal, matched displacement, shuffle | `dc_surgical_ablation` |
| spectrum of the type difference over the index | `type_structure_spectrum` |
| type and index share their frequencies | `type_frequency_alignment` |
| type margin, with and without the direction | `type_margin_decomposition` |
| equivalence interval on the class difference | `margin_equivalence_interval` |
| when the direction forms | `bit_emergence_trajectory` |
| whether it is load-bearing early | `early_bit_ablation` |
| the same contrast indexed on competence, not epoch | `matched_acc_bit_ablation` |
| absent at initialization | `init_bit_baseline` |
| generalization rate by group order | `population_rate_analysis` |
| adversarial controls | `adversarial_checks` |
| probe robustness across residual sites | `residual_probe_robustness` |
| accuracy against perturbation size | `perturbation_sensitivity` |
| label-shuffle control | `shuffle_analysis` |
| what each ablation breaks, not just what it costs | `edit_error_profile` |
| the same ablation pair at the final checkpoint | `converged_type_ablation` |
| what the direction is worth to the task, early | `early_edit_error_profile` |
| the same, indexed on competence | `matched_edit_error_profile` |
| the timeline of Section 3.4, as medians | `timeline_medians` |
| the doubly trained seeds are one trajectory | `verify_twin_runs` |

Three scripts draw the paper's figures from those CSVs and write them to
`figures/`: `figure_fork`, `figure_deflation`, `figure_lifecycle`. They read
committed results and measure nothing, so they run without the checkpoints.

One paragraph per instrument, generated from its docstring, in
[`docs/INSTRUMENTS.md`](docs/INSTRUMENTS.md). The 35 runs, their class and their
provenance are in [`docs/POPULATION.md`](docs/POPULATION.md).

## Layout

```
src/           model, data generation, training
instruments/   the measurements, and the scripts that draw the figures
scripts/       the sweeps that produced the runs
results/       every CSV and console log the paper cites
figures/       written by the figure scripts; not checked in
docs/          population, instruments, reproduction recipe
```

## Checkpoints

Weights are released separately because they do not belong in a clone.

- **Final checkpoints** — the 35 runs of the population of record plus the two
  label-shuffle memorizers the appendix's control needs. 30 MB. Most instruments
  run from these alone. GitHub release asset.
- **Trajectory checkpoints** — every saved epoch of the 30 dense runs, plus the
  coarse copy of each doubly trained seed. Archived with a DOI. Seven
  instruments need it; `docs/INSTRUMENTS.md` names them under **Which archive
  each one needs**, derived from the sources. Their CSVs are committed here, so
  the developmental analysis is checkable without the download; only re-deriving
  it needs the weights.

Both carry a `MANIFEST.sha256`. Both are stripped to weights: the Adam moments
and the scheduler state are three quarters of a raw checkpoint and no measurement
reads them, so the released files cannot be resumed from. Re-training does not
need them.

The whole repository has been checked against these two archives and nothing
else: every committed CSV regenerates byte-identical from them.

## Reproducibility

Training draws no random numbers after initialization: seeds are set for Python,
NumPy and Torch, dropout is zero, and training is full-batch, so nothing
shuffles. What is left is floating point. We provide seeds and the exact
training commands, but do not guarantee bit-exact reproduction: reduction order
depends on the thread count, the BLAS backend and the PyTorch version, and
80,000 optimizer steps through a phase transition can amplify a last-bit
difference. We expect replication to be qualitative — the split should reappear
at a similar rate, and the ablation and developmental measurements should reach
the same conclusions. The class of any individual seed may differ, which is why
the per-run numbers here should be checked against the released checkpoints
rather than against re-trained models.

Environment of record: Python 3.12.8, PyTorch 2.10.0, macOS on Apple Silicon,
CPU, 4 intra-op threads. This is inferred from the machine's install history,
not recorded at the time; `train.py` now prints it at startup.

Train on CPU. PyTorch's MPS backward pass produces gradients orders of magnitude
too large once batch × sequence length exceeds 32,768, which every full-batch run
here does ([pytorch#177116](https://github.com/pytorch/pytorch/issues/177116)).

## Licence

Code MIT, artifacts (CSVs, logs, checkpoints) CC BY 4.0. See `LICENSE`.
