# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `notebooks/00_quickstart_single_subject.ipynb`: subject 0005, four MRI
  channels, the MONAI BraTS segmenter, and enhancing-tumor Dice.

## [1.0.0] - 2026-10-04

### Added
- Public code release `v1.0.0` for the 2026 Journal of Medical Engineering &
  Technology article. This repository is that release.
- Committed `uv.lock`, resolved on Apple Silicon with `uv sync --extra dev`
  and checked with `make test`.
- TCIA download steps for UCSF-PDGM version 5, including the
  `UCSF-PDGM-XXXX_nifti/` directory the extractor expects.

### Changed
- `make figures` and `make all` call the existing figure and table targets.
- `make paper` builds `paper/main.pdf` from the committed manuscript sources.

### Added
- Initial public release of the code and manuscript sources for the 2026
  Journal of Medical Engineering & Technology article.
- Cross-platform software installation and reproduction guide covering `uv`,
  Apple Silicon / MPS, Ubuntu NVIDIA / CUDA, FreeSurfer, data staging, staged
  reproduction targets, conda fallback, and troubleshooting.
- Standalone TumorSynth agreement-table producer for aggregating
  `dl_vs_tumorsynth` Dice, HD95, volumetric error, sensitivity, and specificity
  sidecars.
- TumorSynth comparator documentation covering the Ubuntu CUDA nnU-Net v1.7
  setup, wrapper patches, full n=50 run, voxel-space table, and Hit-Plot
  agreement summary.
- Library skeleton: `hpgs.{io,preprocess,parcellate,segment,register,hitplot,metrics,viz}`.
- CLI: `hpgs-extract-ucsfpdgm`, `hpgs-run-subject`, `hpgs-run-cohort`.
- Cohort scripts: `scripts/extract_ucsfpdgm.py`, `scripts/draw_cohort.py`.
- Cohort config: `configs/cohort_ucsfpdgm_n50.yaml` (n=50, leak-safe stratified).
- Default pipeline config: `configs/default.yaml`.
- Smoke tests for metrics, sub-region derivation, and Hit-Plot burden.
- Cursor / AI-assistant rules in `.cursor/rules` and `AGENTS.md`.
- CI: lint + smoke tests on `ubuntu-latest` and `macos-14`.
- **Integrated paper-and-code:** `paper/` subfolder with `main.tex`,
  `references.bib`, BibTeX styles, `paper/Makefile` + `latexmkrc`, and
  `paper/figs/` synced from `outputs/figures/` via `scripts/sync_figs_to_paper.py`.
  Top-level `Makefile` orchestrates `figures → sync → paper`.
  `make paper` builds the author manuscript. The published journal PDF is in
  `article/`.
- **LaTeX authoring conventions** documented in `.cursor/rules`,
  `paper/README.md`, and the top-level `README.md`. Preferred local editor:
  **TeXShop** (MacTeX) on macOS using `pdflatex` + `bibtex`; `latexmk` is the
  command-line path.
- **Build-system reference:** `docs/build.md` — full documentation of the
  top-level `Makefile` (project orchestrator) and `paper/Makefile` (manuscript
  builder), with target tables, typical command-line sessions, useful flags,
  and how the chain maps to CI. Linked from `docs/index.md`, the top-level
  `README.md`, and `paper/README.md`.
- **Figure 2 end-to-end regeneration (WS3/WS4 scaffold):**
  - `hpgs.preprocess.synth_sr` now wraps FreeSurfer 8.2.0 `mri_synthsr`
    (previously `NotImplementedError`).
  - `hpgs.parcellate.recon_all_clinical` wraps `recon-all-clinical.sh` to
    produce `synthSR.mgz`, `aparc+aseg.mgz` and `wmparc.mgz`; idempotent when
    outputs are present.
  - `hpgs.parcellate.resample_like` wraps `mri_convert -rl` for resampling FS
    conformed-space outputs back onto the native mpMRI grid.
  - `hpgs.viz.parse_freesurfer_color_lut` and `colorize_label_volume` turn
    FreeSurferColorLUT.txt into RGB overlays for aparc+aseg/wmparc/synthseg.
  - `scripts/run_fs_clinical.py` runs the FS clinical stack per subject;
    `scripts/render_figure2_panels.py` renders the eight Freeview-style panels
    (a..h) of manuscript Figure 2 as separate PNGs with crosshair markers;
    `scripts/run_figure2_legacy5.py` drives both for all five legacy subjects.
  - `configs/figure2_legacy5.yaml` pins the five crosshair voxel coordinates
    (0020, 0022, 0039, 0066, 0085) and output path.
  - `Makefile` gains `figure2-legacy5` (full re-run) and `figure2-render`
    (figures only, skipping recon-all-clinical).
  - `paper/main.tex` Figure 2 now uses eight separate
    `\includegraphics` (panels a..h) and the caption has been rewritten to
    state that all panels are regenerated end-to-end by HPGS.
  - Smoke tests in `tests/test_figure2_rendering.py` cover the LUT parser,
    label colorisation, and the eight-panel renderer on synthetic volumes.

### Changed
- **Figure 2 orientation fix:** all panels now use conventional *radiological*
  display on canonical RAS data. UCSF-PDGM NIfTIs ship in LPS voxel ordering,
  which caused the axial (and sagittal/coronal) slices to render upside-down /
  mirrored in the pre-revision Figure 2 (visible in panels e and g of the
  original manuscript). `hpgs.io.load_nifti_canonical` now reorients every
  volume to RAS before slicing, `hpgs.io.orig_voxel_to_ras` translates the
  manuscript's Freeview-picked voxel coordinates from the original LPS frame
  into RAS, and the renderer applies `fliplr(rot90(...))` for all three
  planes so that image-left = patient-right (axial/coronal), anterior-left
  (sagittal) and superior-up (axial/sagittal). New tests in
  `tests/test_orientation.py` pin down the load / coord / display semantics.
