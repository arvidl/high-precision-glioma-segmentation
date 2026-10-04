# Build system — `make` targets

This page documents the **two Makefiles** in the repository and how to drive a
full revision build from the command line. It is the authoritative reference
for both human contributors and AI-assisted coding tools.

> Run all commands from the repository root unless stated otherwise:
> `cd /path/to/high-precision-glioma-segmentation`

For first-time software installation on Apple Silicon / MPS or Ubuntu NVIDIA /
CUDA, see [`software_installation.md`](software_installation.md). This page
focuses on the Makefile targets once the environment is ready.

## Why a Makefile at all?

In an *integrated paper-and-code repo* it is easy for the figures shown in the
PDF to drift away from the code that produced them. Routing every build through
`make` means the manuscript PDF and the underlying figure files all come from
a single, reproducible chain:

```
code  →  scripts  →  outputs/figures  →  paper/figs  →  paper/main.pdf
```

CI (`.github/workflows/ci.yml`) runs lint and smoke tests on Ubuntu and
macOS. It does not rebuild figures or `paper/main.pdf`. Build that PDF
locally with `make paper`. See *What CI does* below.

## The two Makefiles

| File | Role |
|---|---|
| `Makefile` (repo root) | **Project orchestrator.** Chains: code → figures → figure sync → paper PDF. This is the file you normally interact with. |
| `paper/Makefile` | **Manuscript builder.** Knows how to turn LaTeX sources into PDFs via `latexmk`. The top-level Makefile delegates to it via `$(MAKE) -C paper …`. |

## Top-level targets

| Command | What it does | When to use |
|---|---|---|
| `make help` | Lists targets with one-line descriptions. | Any time you forget the names. |
| `make install` | `uv sync --extra dev` + installs pre-commit hooks. | Once, after `git clone`. |
| `make test` | `pytest -q -m "not slow and not gpu"` — runs the smoke tests. | Before every commit. |
| `make lint` | `ruff check .` and `ruff format --check .` (read-only). | Before pushing; CI mirrors this. |
| `make format` | `ruff format .` then `ruff check --fix .` (writes). | When the editor hasn't autoformatted. |
| `make figures` | Runs the computed-figure targets listed in `FIGURE_TARGETS` (cohort table, Dice table, runtime table, agreement panel, anatomical-profile panels, LUMIERE figures, TumorSynth table). Needs staged exams and derivatives. | After pipeline outputs exist and a figure or table should be regenerated. |
| `make sync` | Runs `scripts/sync_figs_to_paper.py` to mirror `outputs/figures/` and `outputs/tables/` into `paper/figs/`. | After `make figures`, before rebuilding the PDF from those new files. |
| `make paper` | `make -C paper paper` → produces `paper/main.pdf` via `latexmk` from the committed manuscript sources. | Whenever you want the author-manuscript PDF. |
| `make docs` | Renders `docs/*.md` design notes to matching `docs/*.pdf` via `pandoc --pdf-engine=xelatex`. | After editing a design note in `docs/`. |
| `make all` | `figures`, then `sync`, then `paper`. | Full regeneration once cohort derivatives exist. |
| `make clean` | Wipes `outputs/`, TeX build artefacts (`.aux`, `.log`, `.bbl`, …), and the lint caches. | When something looks stale or before tagging a release. |

## `paper/Makefile` targets

These are usually called *via* the top-level Makefile, but you can invoke them
directly from `paper/` when iterating on LaTeX only:

| Command (from `paper/`) | What it does |
|---|---|
| `make paper` | `latexmk -pdf -bibtex main.tex` → `main.pdf` |
| `make all` | The manuscript PDF |
| `make watch` | `latexmk -pvc` — continuous rebuild on save (handy as a second terminal next to TeXShop) |
| `make clean` | `latexmk -c` — remove intermediate files but keep the PDF |
| `make veryclean` | `latexmk -C` + remove `.bbl`, `.run.xml` — full reset |

`paper/Makefile` builds `main.pdf` from `main.tex`, `references.bib`, and
`abbrv.bst` next to the source. `refs/` still holds extra BibTeX styles.

## Typical command-line sessions

### First time after cloning

```bash
cd /path/to/high-precision-glioma-segmentation
make install           # uv sync --extra dev + pre-commit install
make test              # sanity: smoke tests pass
make help              # browse remaining targets
```

### Iterating on code that affects a figure

```bash
# edit src/hpgs/viz/hitplot.py …
make figures           # regenerate computed figures and tables
make sync              # copy outputs/ into paper/figs
make paper             # rebuild paper/main.pdf
open paper/main.pdf
```

### Iterating on LaTeX text only (no figure changes)

The maintainer normally stays in TeXShop and presses ⌘T. From the command line:

```bash
make paper             # latexmk is incremental; this is fast
```

Or, from inside `paper/`, run the watcher:

```bash
cd paper
make watch             # rebuilds on every save until you hit ^C
```

### Full regeneration before a commit

```bash
make clean
make all
git status             # check what changed
```

## Useful flags

- **Override the Python launcher.** The Makefile defaults to `uv run …`. If you
  use conda instead, keep one conda environment active for the whole session and
  override the launchers:
  ```bash
  make figures PY=python
  make test PYTEST=pytest
  make lint RUFF=ruff
  ```
- **Select the inference device.** The segmentation adapter accepts `auto`,
  `cpu`, `cuda`, and `mps`; `auto` tries CUDA, then MPS, then CPU.
  ```bash
  make smoke-segment SMOKE_DEVICE=mps     # Apple Silicon
  make smoke-segment SMOKE_DEVICE=cuda    # Ubuntu NVIDIA
  make segment-all SEGMENT_DEVICE=cuda SEGMENT_SKIP_EXISTING=1
  ```
- **Just the LaTeX, skipping figure regeneration:**
  ```bash
  make -C paper paper
  ```
- **Continuous LaTeX rebuild** (handy alongside TeXShop):
  ```bash
  make -C paper watch
  ```
- **Dry-run any target:**
  ```bash
  make -n all            # prints commands without executing
  ```
- **Parallel `latexmk` jobs:**
  ```bash
  make -C paper paper JOBS=2
  ```

## How TeXShop fits in

`make paper` and TeXShop produce the **same PDF from the same sources** —
they're two paths to the same destination, both using `pdflatex` + `bibtex`.
Day-to-day workflow: write text in TeXShop (⌘T to typeset, SyncTeX to jump),
and use `make paper` to rebuild `paper/main.pdf` from the committed sources.
Run `make sync` first when `outputs/figures/` has newer files than `paper/figs/`.

## What CI does

`.github/workflows/ci.yml` runs one job on every push and PR to `main`:

1. **`lint-and-test`** (Ubuntu-latest and macOS-14, Python 3.11): `uv sync --extra dev` installs the runtime plus `dev` tooling, then runs `ruff check . && ruff format --check .` and `pytest -q -m "not slow and not gpu"`. Mirrors `make install` (minus the `pre-commit install` step, which is a no-op in CI's ephemeral checkout) followed by `make lint` and `make test`.

The manuscript PDF is rebuilt locally with `make paper`.

If `make lint && make test` works locally it will work in CI, and vice versa.
