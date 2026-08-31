# Reproducing the numbers

Two paths. Re-measuring the released checkpoints reproduces every number in the
paper and takes about an hour. Re-training reproduces the population and takes
150–250 CPU-hours.

## Re-measure (the short path)

Nothing here reads the training data. The instruments rebuild the multiplication
table from `src/dihedral.py` and read weights, so the checkpoints are the only
download.

```bash
pip install -r requirements.txt

curl -L -o final_checkpoints.tar.gz <release asset URL>
tar xf final_checkpoints.tar.gz
shasum -c checkpoints/MANIFEST.sha256

export DIHEDRAL_RUNS=$PWD/checkpoints
python -m instruments.lin_out_all_seeds
```

Then any instrument, in any order, with two exceptions:

- `unembed_type_geometry` and `unembed_type_direction_readout` join a column
  from `lin_out_all_seeds`.
- `margin_equivalence_interval` reads `adversarial_checks`.

Run those producers first. A consumer run too early does not fail — it leaves
the joined column blank and says so on the console, which is the quietest way to
get a wrong CSV. `docs/INSTRUMENTS.md` repeats this warning.

Each instrument writes `results/<name>.csv`, overwriting the committed copy.
Compare with `git diff results/`. Small last-digit differences are expected on a
different BLAS backend; a changed class label or a changed sign is not.

`results/logs/` holds the console output of the run that produced each committed
CSV, so a disagreement can be traced without re-running anything.

## The figures

```bash
python -m instruments.figure_fork
python -m instruments.figure_deflation
python -m instruments.figure_lifecycle
```

These read the committed CSVs and write PDFs to `figures/`. They need no
checkpoints, so they run on a fresh clone. Set `DIHEDRAL_FIGURES` to write
somewhere else.

## The instruments that need trajectories

Seven instruments read per-epoch checkpoints rather than final weights; they are
listed under **Which archive each one needs** in `docs/INSTRUMENTS.md`, which is
generated from the sources so it cannot drift. Unpack the trajectory archive and
point `DIHEDRAL_RUNS` at it. They stop with a message naming the missing
checkpoints if you have only the final ones.

That archive also carries the coarse copy of each doubly trained seed, which is
what `verify_twin_runs` compares against the dense copy.

Their CSVs are committed, so skipping this download costs the ability to
re-derive them, not the ability to check the analysis.

## Re-train (the long path)

```bash
scripts/run_prime_composite_sweep.sh    # D_47..D_63, phase 1
scripts/run_prime_composite_phase2.sh   # continues the runs that had not settled
scripts/run_seed_sweep.sh               # D_59, five seeds
scripts/run_dense_checkpoint_sweep.sh   # the trajectory runs, frequent checkpoints
```

Each script generates its own data — the split is seeded, so the datasets are
not shipped — and writes into `runs/`. One run is 80,000 full-batch epochs, about
six hours on CPU. 47 runs were trained to obtain the 35 that generalize; see
`docs/POPULATION.md` for which did not and where they stopped.

Train on CPU. PyTorch's MPS backward pass produces gradients orders of magnitude
too large once batch × sequence length exceeds 32,768, which every run here
exceeds ([pytorch#177116](https://github.com/pytorch/pytorch/issues/177116)).

Expect the population to differ. See the reproducibility note in the README: the
class of any individual seed is not guaranteed across machines or PyTorch
versions, so a re-trained `d47_seed0` landing in the other class neither confirms
nor falsifies anything in the paper. What should replicate is the split itself,
at a similar rate, and every conclusion drawn from the edits and the trajectories.
