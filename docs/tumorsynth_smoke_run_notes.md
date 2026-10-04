# mri_TumorSynth Comparator: Method Notes and Smoke-Run Observations

Date: 2026-04-29

This note records the first local `mri_TumorSynth` comparator run for the HPGS
revision. The goal is not to replace the canonical MONAI Bundle BraTS
three-compartment SegResNet backbone, but to test whether TumorSynth can be used
as an additional segmentation-backbone sensitivity comparator on the same staged
mpMRI inputs.

## Method Summary

The HPGS adapter runs TumorSynth as a two-stage command-line workflow:

1. Run `mri_TumorSynth --wholetumor` on the same four staged cohort channels used
   by the MONAI Bundle run:
   `T1c`, `T1`, `T2`, and `FLAIR`.
2. Convert the whole-tumor output into temporary tumor-ROI images by multiplying
   each input channel by the whole-tumor mask.
3. Run `mri_TumorSynth --innertumor` on those ROI images.
4. Remap the TumorSynth outputs to the HPGS/BraTS label convention:
   `0 = background`, `1 = NCR`, `2 = ED`, `4 = ET`.

Current remapping:

- whole-tumor label: `18`
- inner labels mapped to NCR: `3`
- inner labels mapped to ED: `1`
- inner labels mapped to ET: `2`
- unassigned voxels inside the whole-tumor mask are kept as ED, preserving the
  TumorSynth whole-tumor extent.

The TumorSynth wrapper and sidecar record this provenance in
`seg_tumorsynth/sub-XXXX_seg_tumorsynth.json`.

Note: the five-subject raw-label audit on `0005`, `0012`, `0018`, `0026`, and
`0035` found that this local TumorSynth installation's inner labels align as
`1 -> ED`, `2 -> ET`, and `3 -> NCR`. Earlier smoke runs used two other
assumptions (`1,3 -> NCR; 2 -> ET`, then `1 -> NCR; 2 -> ED; 3 -> ET`); do not
use those stale compartment metrics in a table or figure.

## Important Methodological Caveat

TumorSynth documentation expects skull-stripped inputs registered to the SRI-24
template. The HPGS comparator run deliberately uses the same staged mpMRI inputs
as the MONAI Bundle run to keep the comparison input-matched. Therefore, these
results should be described as a bounded sensitivity/comparator analysis, not as
a benchmark of TumorSynth under its ideal preprocessing regime.

This caveat has also been added to `paper/main.tex`.

## Local Setup Notes

Installed executable:

```bash
/Applications/freesurfer/8.2.0/bin/mri_TumorSynth
```

Model root:

```bash
/Users/arvid/nnUNet
```

Required model trees are present:

```bash
/Users/arvid/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Tumor
/Users/arvid/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task003_InnerTumor
```

The FreeSurfer wrapper checks the `nnUNet_v1.7/...` path, while `nnUNet_predict`
looked under `/Users/arvid/nnUNet/3d_fullres/...`. A non-destructive symlink was
therefore added:

```bash
/Users/arvid/nnUNet/3d_fullres -> /Users/arvid/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres
```

On `sub-0026`, the stock FreeSurfer 8.2.0 wrapper also failed after the first
modality-specific `nnUNet_predict` pass because it contains `((i++))` under
`set -e`; when `i=0`, this arithmetic command returns status 1 and aborts the
script. A local patched copy was created:

```bash
/Users/arvid/nnUNet/mri_TumorSynth_hpgs
```

The only change is:

```bash
((i++)) || true
```

Runtime environment:

- conda env: `nnUNet_v1.7`
- `nnUNet_predict` is available under `/opt/anaconda3/envs/nnUNet_v1.7/bin`
- FSL commands are available under `/Users/arvid/fsl/share/fsl/bin`
- CPU mode is required on the MBP M4 Max; CUDA is not available.

## First Smoke Run: `sub-0005`

Command:

```bash
conda run -n nnUNet_v1.7 bash -lc '
  export PATH=/opt/anaconda3/envs/nnUNet_v1.7/bin:$PATH
  export NNUNET_ENV_DIR=/Users/arvid/nnUNet
  export nnUNet_raw_data_base=/Users/arvid/nnUNet/nnUNet_raw_data_base
  export nnUNet_preprocessed=/Users/arvid/nnUNet/nnUNet_preprocessed
  export RESULTS_FOLDER=/Users/arvid/nnUNet
  make tumorsynth-all \
    TUMORSYNTH_COMMAND=/Applications/freesurfer/8.2.0/bin/mri_TumorSynth \
    TUMORSYNTH_NNUNET_DIR=/Users/arvid/nnUNet \
    TUMORSYNTH_CPU=1 \
    TUMORSYNTH_THREADS=8 \
    BASELINE_SUBJECTS="0005"
'
```

Outcome:

- status: success
- elapsed time: `8651.4 s` (`~2 h 24 min`)
- output mask:
  `data/derivatives_cohort50/sub-0005/seg_tumorsynth/sub-0005_seg_tumorsynth.nii.gz`
- sidecar:
  `data/derivatives_cohort50/sub-0005/seg_tumorsynth/sub-0005_seg_tumorsynth.json`
- paired metrics:
  `data/derivatives_cohort50/sub-0005/metrics/sub-0005_metrics_dl_vs_tumorsynth.json`

DL vs TumorSynth metrics for `sub-0005`:

| Compartment | Dice | HD95 (mm) | Absolute VE (mm3) |
|---|---:|---:|---:|
| WT | 0.841 | 3.74 | 5789 |
| TC | 0.398 | 19.80 | 14136 |
| ET | 0.702 | 2.83 | 79 |

Observations:

- The TumorSynth workflow completed end-to-end on Apple Silicon CPU mode, despite
  the upstream documentation warning that TumorSynth currently only works on x86
  architectures.
- Runtime is the limiting factor. At `TUMORSYNTH_THREADS=8`, one subject took
  about 2.4 hours. Full-cohort n=50 execution on the MBP would therefore be
  impractical without further acceleration or running only a selected subset.
- TC agreement was much weaker than WT and ET for this subject. This should be
  visually inspected before interpreting aggregate numbers, because the result
  may reflect label-remapping assumptions, preprocessing mismatch, or genuine
  segmenter disagreement.

## Second Smoke Run: `sub-0026`

Command:

```bash
make tumorsynth-all \
  TUMORSYNTH_COMMAND=/Users/arvid/nnUNet/mri_TumorSynth_hpgs \
  TUMORSYNTH_NNUNET_DIR=/Users/arvid/nnUNet \
  TUMORSYNTH_CPU=1 \
  TUMORSYNTH_THREADS=16 \
  BASELINE_SUBJECTS="0026" \
  BASELINE_SKIP_EXISTING=1
```

Outcome:

- status: success
- elapsed time: `1715.4 s` (`~28.6 min`)
- output mask:
  `data/derivatives_cohort50/sub-0026/seg_tumorsynth/sub-0026_seg_tumorsynth.nii.gz`
- sidecar:
  `data/derivatives_cohort50/sub-0026/seg_tumorsynth/sub-0026_seg_tumorsynth.json`
- paired metrics:
  `data/derivatives_cohort50/sub-0026/metrics/sub-0026_metrics_dl_vs_tumorsynth.json`

DL vs TumorSynth metrics for `sub-0026`:

| Compartment | Dice | HD95 (mm) | Absolute VE (mm3) |
|---|---:|---:|---:|
| WT | 0.893 | 6.32 | 1942 |
| TC | 0.644 | 9.00 | 25382 |
| ET | 0.839 | 3.74 | 134 |

Observations:

- The patched wrapper fixed the `((i++))` / `set -e` failure.
- Increasing from `TUMORSYNTH_THREADS=8` to `TUMORSYNTH_THREADS=16` reduced the
  observed runtime from `~2 h 24 min` on `sub-0005` to `~28.6 min` on `sub-0026`.
  This is not a controlled benchmark because the subjects differ, but it is a
  strong practical indication that subsequent TumorSynth runs should use 16
  threads on this machine.
- Agreement was better than for `sub-0005`, especially ET (`0.839` vs `0.702`)
  and TC (`0.644` vs `0.398`), but TC remains the weakest compartment.

## Suggested Next Step

Run a small, intentionally bounded TumorSynth subset before considering any
larger cohort pass.

Recommended next candidates:

- `sub-0012`
- `sub-0018`
- `sub-0035`

Suggested command:

```bash
conda run -n nnUNet_v1.7 bash -lc '
  export PATH=/opt/anaconda3/envs/nnUNet_v1.7/bin:$PATH
  export NNUNET_ENV_DIR=/Users/arvid/nnUNet
  export nnUNet_raw_data_base=/Users/arvid/nnUNet/nnUNet_raw_data_base
  export nnUNet_preprocessed=/Users/arvid/nnUNet/nnUNet_preprocessed
  export RESULTS_FOLDER=/Users/arvid/nnUNet
  make tumorsynth-all \
    TUMORSYNTH_COMMAND=/Users/arvid/nnUNet/mri_TumorSynth_hpgs \
    TUMORSYNTH_NNUNET_DIR=/Users/arvid/nnUNet \
    TUMORSYNTH_CPU=1 \
    TUMORSYNTH_THREADS=16 \
    BASELINE_SUBJECTS="0012 0018 0035" \
    BASELINE_SKIP_EXISTING=1
'
```

After completion, compare runtime and metrics against `sub-0005` and `sub-0026`
before deciding whether TumorSynth is worth including as more than a small
sensitivity subset.

## Ubuntu CUDA Full-Cohort Run: `n=50`

Date: 2026-05-05

The Ubuntu RTX A5000 Laptop GPU setup completed the full UCSF-PDGM n=50
TumorSynth comparator run with zero per-subject failures:

```text
ok=50  failed=0  skipped=0  no-paired-metrics=0
```

Runtime was approximately 165--278 s per subject with `TUMORSYNTH_THREADS=16`
and GPU-enabled nnU-Net v1.7 available through a dedicated `uv` environment.
The HPGS orchestration still used the project `.venv`; the legacy TumorSynth
environment only supplied `nnUNet_predict` via `PATH`.

### Ubuntu installation summary

The TumorSynth / nnU-Net v1.7 environment was created separately from the HPGS
environment:

```bash
uv venv --python 3.8 .venv-nnunet-v17-cuda
source .venv-nnunet-v17-cuda/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
uv pip install \
  tqdm scipy scikit-image scikit-learn SimpleITK pandas nibabel \
  matplotlib medpy batchgenerators pydicom
uv pip install --no-deps \
  "git+https://github.com/MIC-DKFZ/nnUNet.git@7f1e273fa1021dd2ff00df2ada781ee3133096ef"
```

The local model root was:

```bash
/media/arvid/prj/nnUNet
```

with required model trees:

```bash
/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Tumor
/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task003_InnerTumor
```

The FreeSurfer 8.2.0 `mri_tumorsynth` wrapper required two local Linux patches
in `/media/arvid/prj/nnUNet/bin/mri_tumorsynth_hpgs`:

```bash
mktemp -d -t TumorSynth_XXXXXX
((i++)) || true
```

### Full-cohort command

```bash
source .venv-nnunet-v17-cuda/bin/activate
export PATH="$PWD/.venv-nnunet-v17-cuda/bin:$PATH"
export NNUNET_ENV_DIR=/media/arvid/prj/nnUNet
export nnUNet_raw_data_base=/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_raw_data_base
export nnUNet_preprocessed=/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_preprocessed
export RESULTS_FOLDER=/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_trained_models

make tumorsynth-all \
  PY="$PWD/.venv/bin/python" \
  TUMORSYNTH_COMMAND=/media/arvid/prj/nnUNet/bin/mri_tumorsynth_hpgs \
  TUMORSYNTH_NNUNET_DIR=/media/arvid/prj/nnUNet \
  TUMORSYNTH_THREADS=16 \
  BASELINE_SKIP_EXISTING=1
```

### Aggregated voxel-space metrics

Produced by:

```bash
make table-tumorsynth-agreement TUMORSYNTH_TABLE_STRICT=1
```

Median [IQR] over the n=50 cohort:

| Metric | WT | TC | ET |
|---|---:|---:|---:|
| Dice | 0.894 [0.866, 0.921] | 0.860 [0.793, 0.901] | 0.672 [0.559, 0.736] |
| HD95 (mm) | 4.06 [3.20, 5.83] | 4.30 [3.04, 6.38] | 4.00 [3.00, 5.39] |
| \|VE\| (mm3) | 6544 [3547, 12452] | 2652 [906, 6561] | 3704 [793, 7590] |
| Sensitivity | 0.878 [0.831, 0.914] | 0.894 [0.777, 0.946] | 0.660 [0.507, 0.762] |
| Specificity | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] |

Denominator note: ET HD95 used n=49/50 because one subject had infinite HD95;
ET sensitivity used n=49/50 because one subject had undefined sensitivity.

### Hit-Plot agreement

Produced by:

```bash
make hitplot-all HITPLOT_SOURCES=tumorsynth HITPLOT_SKIP_EXISTING=1

uv run python scripts/build_hitplot_agreement_panel.py \
  --prediction dl \
  --reference tumorsynth \
  --out-pdf outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.pdf \
  --out-json outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.json \
  --out-csv outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.csv \
  --strict
```

Full-cohort Hit-Plot agreement:

| Compartment | n pairs | CCC | Mean diff | 95% LoA | Mean \|dV\| |
|---|---:|---:|---:|---:|---:|
| WT | 8600 | 0.967 | -0.387 | [-8.59, +7.81] | 72.5 |
| TC | 8600 | 0.941 | -0.082 | [-5.96, +5.80] | 34.3 |
| ET | 8600 | 0.881 | -0.178 | [-5.30, +4.94] | 39.1 |

### Interpretation

The n=50 run shows strong DL-vs-TumorSynth agreement for WT and TC and lower,
more variable agreement for ET. This supports using TumorSynth as a bounded
segmentation-backbone sensitivity comparator. It should still not be described
as a benchmark of TumorSynth under its ideal preprocessing regime, because this
HPGS run deliberately uses the same native staged mpMRI inputs as the MONAI
Bundle pipeline rather than SRI-24-registered TumorSynth inputs.
