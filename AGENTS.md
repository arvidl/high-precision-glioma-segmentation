# Agent / AI-assistant guide for HPGS

This file is read by Cursor and other AI coding assistants. The same rules apply to
human contributors. The canonical conventions are in `.cursor/rules`.

## Mental model

This repository reproduces the analyses in Lundervold et al., *High precision
segmentation of glioma and surroundings: a feasibility study using multiparametric
MRI and deep learning* (Journal of Medical Engineering & Technology, 2026).
The published PDF is `article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf`.

Optimise for:

1. Reproducibility: every figure and table should be regenerable from the
   library code, scripts, and notebooks.
2. Readable, well-typed Python that a reader can audit quickly.
3. Determinism (seed = 20260415).

The author manuscript is `paper/main.tex`. Build it with `make paper`.

## Before editing

1. Check `.cursor/rules` for hard conventions.
2. Look at `tests/` — most modules have a smoke test; failing tests are the
   first thing to fix.
3. Do not commit imaging data or anything that could re-identify a subject.

## Preferred edits

- Small, focused diffs.
- New functionality lives in `src/hpgs/<subpkg>/`; CLI glue in `scripts/` or
  `src/hpgs/cli.py`.
- Notebooks are demonstrations and should be thin shims over library code.
- Never bake notebook outputs into git (pre-commit hook strips them).

## Forbidden

- Committing real patient data or anything that could re-identify a subject.
- Adding silently failing fallbacks (e.g., `try/except: pass`); raise or log.
- Changing the random seed without coordinated updates to the cohort YAML and
  manuscript Methods.
- Rewriting from scratch when an incremental refactor would do.
