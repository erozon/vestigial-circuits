"""Persist instrument output so figures and prose read numbers, not stdout. (2026-08-13)

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
"""
import os

# results/ sits beside the instruments in this repo and one level above them in
# the released one, where they live in instruments/. Resolve rather than assume,
# so the same file works in both without a rewrite at build time.
_BESIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_ABOVE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DIR = os.environ.get("DIHEDRAL_RESULTS") or (_BESIDE if os.path.isdir(_BESIDE)
                                             else _ABOVE)


def path(name):
    return os.path.join(DIR, f"{name}.csv")


def save(name, header, rows):
    """Write rows to results/<name>.csv. Returns the path, and prints it, because
    the instruments announce their own side effects."""
    os.makedirs(DIR, exist_ok=True)
    p = path(name)
    ncol = len(header.split(","))
    with open(p, "w") as f:
        f.write(header + "\n")
        for r in rows:
            if len(r) != ncol:
                raise ValueError(f"{name}: row has {len(r)} fields, header has "
                                 f"{ncol}: {r}")
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"  wrote {len(rows)} rows to {p}")
    return p


def _num(s):
    try:
        return float(s)
    except ValueError:
        return s


def load(name):
    """results/<name>.csv as a list of dicts, numeric where possible."""
    with open(path(name)) as f:
        header = f.readline().strip().split(",")
        return [dict(zip(header, [_num(v) for v in line.strip().split(",")]))
                for line in f if line.strip()]


def column(name, field, key="run"):
    """{run: value} for a single field -- for joining one instrument's output onto
    another's without loading a dataframe library."""
    return {r[key]: r[field] for r in load(name)}


# Figures resolve the same way, for the same reason. In this repo they belong
# beside the paper; in the released one there is no paper, so they go in a
# figures/ directory at the root. A script that hardcodes either path writes
# outside the tree in the other layout.
_PAPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper", "v2", "figures")
_FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
FIGDIR = os.environ.get("DIHEDRAL_FIGURES") or (_PAPER if os.path.isdir(_PAPER)
                                                else _FIGS)


def figure_path(name):
    """Where <name>.pdf goes. Creates the directory."""
    os.makedirs(FIGDIR, exist_ok=True)
    return os.path.join(FIGDIR, f"{name}.pdf")
