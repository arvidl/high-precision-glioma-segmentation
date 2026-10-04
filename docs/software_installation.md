# Software installation and reproduction guide

This guide is the canonical environment recipe for reproducing HPGS on:

- Apple Silicon macOS with PyTorch MPS.
- Ubuntu 24.04 with an NVIDIA GPU and CUDA.

The project uses `uv` and `pyproject.toml` as the primary Python environment
surface. `environment.yml` is kept as a conda fallback for machines where a
binary dependency is easier to obtain from conda-forge, but normal development
and reproduction should start with `uv`.

Run commands from the repository root unless noted otherwise.

## Environment policy

Use this decision rule:

| Situation | Recommended path |
|---|---|
| Fresh macOS or Ubuntu install | `uv sync --extra dev` |
| Day-to-day development | `make ...` targets, which call `uv run ...` |
| Interactive notebook or shell | `source .venv/bin/activate` only when needed |
| CI parity | `make lint && make test` |
| Package solver trouble with a compiled dependency | conda fallback, then override Makefile launchers |
| Full CUDA cohort segmentation | `uv` environment plus explicit PyTorch CUDA validation |

Do not keep conda `base` active while also relying on the repo-local `.venv`.
The clean pattern is:

```bash
conda deactivate 2>/dev/null || true
uv sync --extra dev
uv run python -V
```

### Lockfile strategy

`pyproject.toml` is the human-maintained dependency contract. `uv.lock` is
the resolved contract for CPython 3.11, generated on Apple Silicon with
`make install` and checked with `make test` for release `v1.0.0`. `uv lock`
records platform markers, so the same file is what Linux CI installs.
`make install` selects CPython 3.11. On this Mac, SciPy 1.15 does not load;
the lock resolves SciPy 1.16 or newer.

If you deliberately choose conda, use one environment for the entire session and
override the Makefile launchers:

```bash
conda env create -f environment.yml
conda activate hpgs
make test PYTEST=pytest
make segment-all PY=python SEGMENT_DEVICE=cuda
```

## Common prerequisites

Install these on both platforms:

- Git.
- Python 3.11.
- `uv`.
- FreeSurfer 8.2.0 for SynthSR, SynthSeg, `recon-all-clinical.sh`, and
  `mri_convert`.
- A TeX distribution if rebuilding manuscript PDFs.
- Enough local disk for TCIA UCSF-PDGM v5, extracted cohort data, derivatives,
  FreeSurfer work directories, and outputs.

HPGS does not commit patient imaging data. The expected staged data layout is
documented in `data/README.md`.

## Apple Silicon macOS (MPS)

This is the primary development environment.

### System software

Install:

- Xcode Command Line Tools.
- `uv`.
- FreeSurfer 8.2.0.
- MacTeX if you build the manuscript locally.

Example bootstrap:

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install and source FreeSurfer according to the FreeSurfer 8.2.0 macOS
instructions. In a shell session that runs HPGS:

```bash
export FREESURFER_HOME=/Applications/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
which recon-all-clinical.sh
```

### Python environment

```bash
git clone git@github.com:arvidl/high-precision-glioma-segmentation.git
cd high-precision-glioma-segmentation
conda deactivate 2>/dev/null || true
uv sync --extra dev
uv run pre-commit install
```

Validate PyTorch MPS:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps_available", torch.backends.mps.is_available())
print("mps_built", torch.backends.mps.is_built())
PY
```

Run the CI-equivalent checks:

```bash
make lint
make test
```

Run the three-subject segmentation smoke test on MPS:

```bash
make smoke-segment SMOKE_DEVICE=mps
```

If an individual PyTorch/MONAI operator is not available on MPS, re-run the
same command with `SMOKE_DEVICE=cpu` to distinguish an environment issue from a
model or data issue.

## Ubuntu 24.04 + NVIDIA CUDA

This is the recommended platform for full-cohort GPU segmentation. The RTX
A5000 Laptop GPU with 16 GB VRAM should be sufficient for the MONAI Bundle
BraTS3 inference path, but verify with the smoke test before launching the full
50-subject fan-out.

### System software

Install core packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential git curl ca-certificates pkg-config \
  python3.11 python3.11-venv python3.11-dev \
  make unzip rsync
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

Install TeX only if you will rebuild PDFs on Ubuntu:

```bash
sudo apt install -y latexmk texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-bibtex-extra
```

Install FreeSurfer 8.2.0 from the official Linux distribution and source it in
the shell used for HPGS:

```bash
export FREESURFER_HOME=/opt/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
which recon-all-clinical.sh
which mri_convert
```

The exact FreeSurfer path may differ on your machine. Keep `FREESURFER_HOME`
outside the git repository.

### NVIDIA driver and CUDA validation

Check the host driver first:

```bash
nvidia-smi
```

Record these values in your run notes:

- GPU model.
- Driver version.
- CUDA version reported by `nvidia-smi`.
- Available memory.

For a host reporting CUDA 12.2, the conservative PyTorch choice is a CUDA 12.1
wheel unless the NVIDIA driver is new enough for newer PyTorch CUDA runtime
wheels. The CUDA toolkit installed on the host is less important than the
driver's ability to run the CUDA runtime bundled with the PyTorch wheel.

### Python environment

Start with the project metadata:

```bash
git clone git@github.com:arvidl/high-precision-glioma-segmentation.git
cd high-precision-glioma-segmentation
conda deactivate 2>/dev/null || true
uv sync --extra dev
```

Validate PyTorch and CUDA:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY
```

If `cuda_available` is `False`, do not continue to full segmentation. First
install a CUDA-enabled PyTorch wheel compatible with the driver. Run this after
`uv sync --extra dev`, then validate CUDA again:

```bash
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

After CUDA is validated, run the repo checks:

```bash
make lint
make test
```

## Data staging

### UCSF-PDGM v5

Download **UCSF-PDGM** version 5 from
<https://www.cancerimagingarchive.net/collection/ucsf-pdgm/>
(DOI <https://doi.org/10.7937/tcia.bdgf-8v37>). Sign in, accept the Data
Usage Agreement, and use the collection Download button or the NBIA Data
Retriever. Unpack the archive until the parent directory contains one
`UCSF-PDGM-XXXX_nifti/` folder per exam. In the version-5 package that
parent is named `UCSF-PDGM-v5`. Full notes are in `data/README.md`.

Point the extractor at that directory:

```bash
uv run python scripts/extract_ucsfpdgm.py \
  --src /path/to/UCSF-PDGM-v5 \
  --dst ./data/ucsf_pdgm_cohort50
```

Confirm the locked 50-subject cohort can be resolved:

```bash
make scope-pr7
```

Use the verbose mode when diagnosing missing inputs:

```bash
make scope-pr7 SCOPE_SHOW_PATHS=1
```

### LUMIERE Patient-048

For the longitudinal illustrative case:

```bash
uv run python scripts/extract_lumiere_p048.py \
  --src /path/to/lumiere/Patient-048 \
  --dst ./data/lumiere_p048 \
  --include-legacy-seg
```

The timepoint list and day-axis source of truth is
`configs/lumiere_p048_timepoints.yaml`.

## Reproduction workflow

Run in stages. Each heavy cohort target has a skip-existing option so interrupted
runs can be resumed.

### 1. Smoke tests

On Apple Silicon:

```bash
make smoke-segment SMOKE_DEVICE=mps
```

On Ubuntu CUDA:

```bash
make smoke-segment SMOKE_DEVICE=cuda
```

Expected healthy behavior:

- The MONAI Bundle downloads or loads from cache.
- The three default subjects run without out-of-memory errors.
- Dice values are broadly in the expected BraTS-like range, not near zero.
- Per-subject derivatives are written under `data/derivatives_cohort50/`.

### 2. Full DL segmentation

Apple Silicon:

```bash
make segment-all SEGMENT_DEVICE=mps SEGMENT_SKIP_EXISTING=1
```

Ubuntu CUDA:

```bash
make segment-all SEGMENT_DEVICE=cuda SEGMENT_SKIP_EXISTING=1
```

The script writes:

- `data/derivatives_cohort50/sub-XXXX/seg_dl/sub-XXXX_seg_brats3_dl.nii.gz`
- `data/derivatives_cohort50/sub-XXXX/seg_dl/sub-XXXX_seg_brats3_dl_probs.nii.gz`
- `data/derivatives_cohort50/sub-XXXX/seg_dl/sub-XXXX_seg_brats3_dl.json`
- `data/derivatives_cohort50/sub-XXXX/metrics/sub-XXXX_metrics_dl_vs_gt.json`

### 3. FreeSurfer parcellation

FreeSurfer is CPU-bound and usually dominates wall time. Use a scratch
`SUBJECTS_DIR` outside the repository:

```bash
mkdir -p /scratch/hpgs_fs_subjects
make parcellate-all \
  PARCELLATE_FS_WORK_DIR=/scratch/hpgs_fs_subjects \
  PARCELLATE_THREADS=8 \
  PARCELLATE_SKIP_EXISTING=1
```

For a quick contract test without FreeSurfer:

```bash
make parcellate-all PARCELLATE_BACKEND=dummy PARCELLATE_SUBJECTS="0005 0035"
```

Do not mix dummy parcellations with reviewer-facing full-cohort outputs.

### 4. Hit-Plot and tables

After segmentation and parcellation:

```bash
make hitplot-all HITPLOT_SOURCES="dl dl_prob gt" HITPLOT_SKIP_EXISTING=1
make table2-dl-vs-gt TABLE2_STRICT=1
make agreement-panel AGREEMENT_STRICT=1
make bench-runtime
```

Optional TumorSynth comparator runs are documented separately in
`docs/tumorsynth_comparator_workflow.md`. Treat TumorSynth as a bounded
sensitivity analysis unless its upstream preprocessing requirements are fully
matched.

For the Ubuntu n=50 TumorSynth comparator run, keep a separate legacy nnU-Net
v1.7 environment (for `nnUNet_predict`) rather than installing it into the main
HPGS `.venv`. The validated workflow uses `make tumorsynth-all`, then:

```bash
make hitplot-all HITPLOT_SOURCES=tumorsynth HITPLOT_SKIP_EXISTING=1
make table-tumorsynth-agreement TUMORSYNTH_TABLE_STRICT=1
```

See `docs/tumorsynth_comparator_workflow.md`,
`docs/tumorsynth_appendix5_reproducibility.md`, and
`docs/tumorsynth_smoke_run_notes.md` for model-tree paths, wrapper patches, and
the n=50 summary metrics.

### 5. LUMIERE longitudinal outputs

After staging LUMIERE Patient-048:

```bash
make summarize-lumiere-p048-segmentation
make parcellate-lumiere-p048 LUMIERE_PARCELLATE_BACKEND=freesurfer
make hitplot-lumiere-p048
make figure6-lumiere-volumes
make figure11-lumiere-p048-hitplot
```

Use the dummy backend only for smoke testing:

```bash
make parcellate-lumiere-p048 LUMIERE_PARCELLATE_BACKEND=dummy
```

### 6. Paper rebuild

`make paper` builds `paper/main.pdf` from the sources already in `paper/`,
including the committed figures. After `make figures` has written new
artefacts under `outputs/`:

```bash
make sync
make paper
```

`make all` runs `make figures`, `make sync`, and `make paper`. It needs the
staged exams and pipeline derivatives.

## Conda fallback

Use conda only when `uv` cannot resolve or install a binary dependency on the
target machine:

```bash
conda env create -f environment.yml
conda activate hpgs
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
pytest -q -m "not slow and not gpu"
```

When running Makefile targets inside conda, override the Python launchers so the
Makefile does not call `uv run`:

```bash
make lint RUFF=ruff
make test PYTEST=pytest
make smoke-segment PY=python SMOKE_DEVICE=cuda
make segment-all PY=python SEGMENT_DEVICE=cuda SEGMENT_SKIP_EXISTING=1
```

Avoid installing some packages into conda and others into `.venv` during the
same run. Pick one environment per reproduction attempt.

## Troubleshooting

### `torch.cuda.is_available()` is false

Check:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Likely causes:

- CPU-only PyTorch wheel.
- NVIDIA driver too old for the installed PyTorch CUDA runtime.
- Running inside a shell/container without GPU visibility.

### CUDA out of memory

Try:

```bash
make smoke-segment SMOKE_DEVICE=cpu
make segment-all SEGMENT_DEVICE=cuda SEGMENT_SUBJECTS="0005" SEGMENT_SKIP_EXISTING=1
```

If one subject fails but others fit, keep `SEGMENT_SKIP_EXISTING=1` and inspect
the failing subject's shape and sidecar logs.

### FreeSurfer command not found

Source FreeSurfer in the current shell:

```bash
export FREESURFER_HOME=/opt/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
which recon-all-clinical.sh
```

### `make scope-pr7` reports missing inputs

The extractor either has not been run or was pointed at the wrong TCIA root.
Use:

```bash
make scope-pr7 SCOPE_SHOW_PATHS=1
```

Then compare the expected filenames against `data/README.md`.

### LaTeX build fails

First check whether you need to rebuild the paper at all. If yes:

```bash
make -C paper clean
make paper
```

On Ubuntu, missing `.sty` files usually mean the TeX Live install is too small;
install `texlive-latex-extra` and `texlive-bibtex-extra`.

## What to record for reproducibility

For every full reproduction run, save these in a local run note:

- Git commit SHA.
- OS version.
- `uv --version`.
- `python --version`.
- `torch.__version__`, `torch.version.cuda`, and CUDA/MPS availability.
- `nvidia-smi` output on Ubuntu.
- FreeSurfer version and `FREESURFER_HOME`.
- Data source release paths for UCSF-PDGM and LUMIERE.
- Exact `make` commands used.
- Any subjects skipped or rerun.

These details are enough to explain differences in runtime, GPU behavior, and
generated paper artefacts without committing local machine paths or patient data.
