# high-precision-glioma-segmentation (HPGS)

Reproducible pipeline for high-precision segmentation of glioma and surroundings using
multiparametric MRI, FreeSurfer 8.2.0 (SynthSeg/SynthSR), fastMONAI, and nnU-Net v2.

This repository reproduces *High precision segmentation of glioma and
surroundings: a feasibility study using multiparametric MRI and deep learning*
(Lundervold et al., Journal of Medical Engineering & Technology, 2026).
The published article is
[`article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf`](article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf).

## What's here

This is an **integrated paper-and-code repository**: the manuscript LaTeX source,
bibliography, and final figures live alongside the pipeline code that produces them.

```
.
├── src/hpgs/         # library code (io, preprocess, parcellate, segment, register,
│                     # hitplot, metrics, viz)
├── notebooks/        # didactic and figure-reproducing notebooks
├── scripts/          # CLI entry points + figure-sync utility
├── configs/          # YAML configs per run (Hydra-friendly)
├── data/             # data access instructions (no binary data committed)
├── tests/            # pytest, small synthetic volumes
├── docs/             # documentation
├── article/          # published journal PDF
├── paper/            # author manuscript: main.tex + references.bib + figs/
│   ├── main.tex
│   ├── references.bib
│   ├── figs/         # figures and table fragments used by main.tex
│   └── Makefile
├── outputs/          # (git-ignored) figures and tables produced by notebooks
├── Makefile          # top-level: figures → sync → paper
└── .cursor/          # project conventions for AI-assisted coding
```

## Hardware targets

- **Primary development:** Apple Silicon (MBP M4 Max, 128 GB unified memory) with
  PyTorch MPS.
- **Secondary:** Linux + CUDA (for full 50-subject batch runs if throughput demands).
- **CI:** `ubuntu-latest` and `macos-14` (Apple Silicon) with a tiny synthetic dataset.

## Quick start

Prerequisites:

- FreeSurfer 8.2.0 installed (`$FREESURFER_HOME` set, `SetUpFreeSurfer.sh` sourced)
- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended; canonical for this repo) or conda
- A LaTeX distribution (MacTeX recommended) if you intend to build the
  manuscript locally — see [`docs/build.md`](docs/build.md) and
  [`paper/README.md`](paper/README.md).

For platform-specific setup, including Apple Silicon / MPS and Ubuntu NVIDIA /
CUDA, use the full
[`software installation and reproduction guide`](docs/software_installation.md).

```bash
git clone git@github.com:arvidl/high-precision-glioma-segmentation.git
cd high-precision-glioma-segmentation

# uv path (preferred on macOS and Ubuntu)
conda deactivate     # leave (base)
make install         # uv sync --extra dev + pre-commit install
make test            # smoke tests
make help            # browse all targets

# conda fallback
conda env create -f environment.yml
conda activate hpgs
pytest -q
```

If you go for the `uv` path, you don't need to "activate" anything for daily work — the Makefile and `uv run` always reach
into `./.venv` automatically. `make install` syncs the `dev` extra as well, so it installs
`pytest`, `ruff`, `pre-commit`, `jupyterlab`, and the rest of the development toolchain before
installing the git hooks.

If you created `.venv` before that change and want to refresh it manually, run:
```bash
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

If you ever want an interactive shell that has the project env on PATH
(e.g. for `python -i`, `jupyter lab`), use:

```bash
source .venv/bin/activate         # optional; only when you want the env on PATH
# … work …
deactivate
```

Most day-to-day operations are driven through `make` — see
[`docs/build.md`](docs/build.md) for the full reference (top-level orchestrator
+ `paper/Makefile`, all targets, common workflows, and how the build chain maps
to CI). For hardware-specific inference checks, start with:

```bash
# Apple Silicon
make smoke-segment SMOKE_DEVICE=mps

# Ubuntu + NVIDIA CUDA
make smoke-segment SMOKE_DEVICE=cuda
```

## Reproducing the article

A fresh clone does not include the MRI exams. UCSF-PDGM v5 comes from TCIA and
LUMIERE from its public archive. FreeSurfer 8.2.0 is an external install.
The files in `paper/figs/` are the figures used by the manuscript. Regenerating
the full n=50 cohort, including the Appendix 5 TumorSynth comparison, needs
those exams and the cohort derivatives; see
[`docs/tumorsynth_appendix5_reproducibility.md`](docs/tumorsynth_appendix5_reproducibility.md).
`make test` checks that the metric builders agree with the committed tables.

1. Obtain UCSF-PDGM v5 from TCIA and extract to `/path/to/UCSF-PDGM-v5`.
2. Extract the leak-safe, clinically stratified 50-subject cohort:
   ```bash
   uv run python scripts/extract_ucsfpdgm.py \
       --src /path/to/UCSF-PDGM-v5 \
       --dst ./data/ucsf_pdgm_cohort50
   ```
3. Run the pipeline for a single subject:
   ```bash
   uv run python scripts/run_subject.py --subject 0005 \
       --config configs/default.yaml
   ```
4. Batch everything:
   ```bash
   uv run python scripts/run_cohort.py --config configs/default.yaml
   ```
5. Compute quantitative metrics (Dice, HD95, VE, Sens/Spec) and regenerate tables:
   ```bash
   uv run python scripts/compute_metrics.py
   ```
6. Regenerate paper figures, sync them, and rebuild the manuscript PDF:
   ```bash
   make figures   # execute notebooks/99_reproduce_paper_figures.ipynb
   make sync      # copy outputs/figures -> paper/figs
   make paper     # build paper/main.pdf from paper/main.tex
   # or simply: make all
   ```

   To regenerate manuscript Figure 2 end-to-end for the five legacy
   UCSF-PDGM subjects (0020, 0022, 0039, 0066, 0085) — using FS 8.2.0
   `mri_synthsr`, `recon-all-clinical.sh` (which produces `aparc+aseg`
   and `wmparc`), and `mri_synthseg --robust --parc`:
   ```bash
   make figure2-legacy5      # full run: FS clinical + panel rendering
   make figure2-render       # rendering only (re-uses existing derivatives)
   make sync                 # copy outputs/figures/figure2_legacy5 -> paper/figs
   ```
   Crosshair coordinates per subject are pinned in
   [`configs/figure2_legacy5.yaml`](configs/figure2_legacy5.yaml).
   `recon-all-clinical.sh` is CPU-bound on Apple Silicon
   (~15–30 min per subject); the wrapper is idempotent, so re-runs skip
   subjects with existing outputs.

   Open `paper/main.tex` in TeXShop (MacTeX) and press ⌘T, or run `make paper`.
   That builds the author manuscript (`paper/main.pdf`): the same text, figures,
   tables, and reference list as the corrected camera-ready source. It is an
   A4 article, not the journal's house style. The published PDF in `article/`
   remains the version of record.

## Cohort

50 GBM (IDH-wildtype, WHO grade 4) subjects, baseline exam only, no prior biopsy,
known OS, and **excluded from the BraTS21 segmentation training cohort** to avoid
data leakage. Stratified draw: Sex × OS tertile, seed 20260415. IDs and per-subject
metadata: `configs/cohort_ucsfpdgm_n50.yaml`. (https://www.cancerimagingarchive.net/collection/ucsf-pdgm)

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
