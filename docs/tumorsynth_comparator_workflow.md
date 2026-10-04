# mri_TumorSynth Comparator Workflow

This note records the optional `mri_TumorSynth` comparator workflow added for the
JMET revision. The canonical segmentation backbone for the manuscript remains
the MONAI Bundle BraTS three-compartment SegResNet; TumorSynth is available as a
bounded sensitivity/comparator run on the same staged mpMRI inputs.

## Inputs

The runner uses the same per-subject cohort inputs as `make segment-all`:

- `sub-XXXX_T1_bias.nii.gz`
- `sub-XXXX_T1c_bias.nii.gz`
- `sub-XXXX_T2_bias.nii.gz`
- `sub-XXXX_FLAIR_bias.nii.gz`

They are resolved from `data/ucsf_pdgm_cohort50/sub-XXXX/` through the shared
`hpgs.io.resolve_subject_inputs` contract. TumorSynth's upstream documentation
expects skull-stripped, SRI-24-registered inputs; this workflow deliberately
keeps the HPGS "same input" contract so the comparison is an input-matched
sensitivity analysis, not a claim that TumorSynth was run in its ideal
preprocessing regime.

## Running

TumorSynth requires a local executable (`mri_TumorSynth` or `mri_tumorsynth`)
and an installed nnU-Net v1.7 model tree containing the TumorSynth weights.

```bash
make tumorsynth-all \
  TUMORSYNTH_COMMAND=mri_TumorSynth \
  TUMORSYNTH_NNUNET_DIR=/path/to/nnUNet \
  BASELINE_SUBJECTS="0005 0026" \
  TUMORSYNTH_CPU=1
```

If `TUMORSYNTH_NNUNET_DIR` is omitted, the runner falls back to `NNUNET_ENV_DIR`.
Use `BASELINE_SKIP_EXISTING=1` to resume a partial run. Use
`BASELINE_REQUIRE_DL=0` to write TumorSynth masks even before the paired MONAI
DL output exists; in that mode the `metrics_dl_vs_tumorsynth.json` file is
skipped.

### Ubuntu CUDA setup used for the n=50 run

The full UCSF-PDGM cohort comparator run was performed on Ubuntu 24.04 with an
NVIDIA RTX A5000 Laptop GPU. The HPGS Python environment remains the ordinary
repo `.venv`; TumorSynth / nnU-Net v1.7 runs from a separate legacy environment
that only supplies `nnUNet_predict` to the FreeSurfer wrapper via `PATH`.

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

The model root is local and is not committed:

```bash
/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task002_Tumor
/media/arvid/prj/nnUNet/nnUNet_v1.7/nnUNet_trained_models/nnUNet/3d_fullres/Task003_InnerTumor
```

Two local wrapper patches were required for the FreeSurfer 8.2.0
`mri_tumorsynth` shell script on Linux:

```bash
mktemp -d -t TumorSynth_XXXXXX
((i++)) || true
```

The patched copy is passed explicitly with `TUMORSYNTH_COMMAND` so the
system-wide FreeSurfer installation remains untouched.

Run a GPU smoke subject without `TUMORSYNTH_CPU=1`:

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
  BASELINE_SUBJECTS="0068"
```

Full-cohort run (resume-safe):

```bash
make tumorsynth-all \
  PY="$PWD/.venv/bin/python" \
  TUMORSYNTH_COMMAND=/media/arvid/prj/nnUNet/bin/mri_tumorsynth_hpgs \
  TUMORSYNTH_NNUNET_DIR=/media/arvid/prj/nnUNet \
  TUMORSYNTH_THREADS=16 \
  BASELINE_SKIP_EXISTING=1
```

## Two-Stage Remapping

The adapter calls TumorSynth twice:

1. Whole-tumor mode on the four input channels.
2. Inner-tumor mode on temporary ROI images made by multiplying each input
   channel by the whole-tumor mask.

The outputs are remapped into the project-wide BraTS label convention:

- `0 = background`
- `1 = NCR / necrotic or non-enhancing tumor core`
- `2 = ED / edema`
- `4 = ET / enhancing tumor`

Default TumorSynth remapping:

- whole-tumor label: `18`
- inner labels mapped to NCR: `3`
- inner labels mapped to ED: `1`
- inner labels mapped to ET: `2`
- unassigned voxels inside the whole-tumor mask are kept as ED, preserving the
  TumorSynth whole-tumor extent.

This default is based on a five-subject raw-label audit
(`0005`, `0012`, `0018`, `0026`, `0035`) against both the MONAI Bundle DL output
and the UCSF-PDGM reference masks. Override these if another installed
TumorSynth model reports a different inner-label order:

```bash
make tumorsynth-all \
  TUMORSYNTH_NNUNET_DIR=/path/to/nnUNet \
  TUMORSYNTH_WHOLE_LABEL=18 \
  TUMORSYNTH_NCR_LABELS=3 \
  TUMORSYNTH_ED_LABELS=1 \
  TUMORSYNTH_ET_LABELS=2
```

## Outputs

For each subject, the runner writes:

- `data/derivatives_cohort50/sub-XXXX/seg_tumorsynth/sub-XXXX_seg_tumorsynth.nii.gz`
- `data/derivatives_cohort50/sub-XXXX/seg_tumorsynth/sub-XXXX_seg_tumorsynth.json`
- `data/derivatives_cohort50/sub-XXXX/seg_tumorsynth/sub-XXXX_tumorsynth_wholetumor_raw.nii.gz`
- `data/derivatives_cohort50/sub-XXXX/seg_tumorsynth/sub-XXXX_tumorsynth_innertumor_raw.nii.gz`
- `data/derivatives_cohort50/sub-XXXX/metrics/sub-XXXX_metrics_dl_vs_tumorsynth.json`

The TumorSynth mask can then be included in downstream Hit-Plot and agreement
workflows:

```bash
make hitplot-all HITPLOT_SOURCES=tumorsynth

uv run python scripts/build_hitplot_agreement_panel.py \
  --prediction dl \
  --reference tumorsynth \
  --out-pdf outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.pdf \
  --out-json outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.json \
  --out-csv outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.csv
```

The paired voxel-space metrics can also be aggregated into a standalone table:

```bash
make table-tumorsynth-agreement TUMORSYNTH_TABLE_STRICT=1
```

This writes:

- `outputs/tables/table_tumorsynth_agreement.tex`
- `outputs/tables/table_tumorsynth_agreement.csv`
- `outputs/tables/table_tumorsynth_agreement.json`

The table intentionally stays separate from `table3_segmenter_agreement`, whose
columns remain tied to the original Raidionics / `segment_glioma` / BraTS21-manual
design.

## Appendix 5 reproducibility

The manuscript's Appendix 5 (TumorSynth sensitivity analysis) requires a full
`n=50` comparator run with the audited inner-label mapping, downstream Hit-Plot
artefacts, `make sync`, and response-prose reconciliation. Because
`data/derivatives_cohort50/` is gitignored, a fresh clone may contain synced
`paper/figs/` assets without local sidecars.

See [`tumorsynth_appendix5_reproducibility.md`](tumorsynth_appendix5_reproducibility.md)
for the step-by-step runbook, validation gates, and decision table.
