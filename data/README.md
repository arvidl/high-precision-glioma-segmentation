# Data

**No binary data is committed to this repository.** This file documents how to
obtain and stage the datasets used in the article.

## Datasets

### UCSF-PDGM v5 (primary)

- Source: TCIA. DOI: <https://doi.org/10.7937/tcia.bdgf-8v37>
- Release: v5 (May 2025) — fixes DTI_eddy_noreg header and adds rotated bvecs.
- Size: ~142 GB across 501 exam folders (495 unique subjects + 6 follow-ups).
- License: Data Usage Agreement on TCIA — confirm before any redistribution.
- The released NIfTI files are already skull-stripped and intra-subject coregistered/resampled to the 3D T2/FLAIR space at 1 mm isotropic resolution, so the primary HPGS workflow uses them as provided and limits preprocessing to visual alignment QC.

### Download

1. Open the TCIA collection page:
   <https://www.cancerimagingarchive.net/collection/ucsf-pdgm/>.
2. Sign in to TCIA and accept the Data Usage Agreement when prompted.
3. Download **version 5** (May 2025) with the collection Download button or
   the NBIA Data Retriever. The package is about 142 GB.
4. Unpack it until you can see one directory per exam, named
   `UCSF-PDGM-XXXX_nifti`. In the version-5 package that parent directory is
   `UCSF-PDGM-v5`. Pass that parent directory as `--src`.

Point the extractor at the root that contains the
`UCSF-PDGM-XXXX_nifti/` directories:

```bash
uv run python scripts/extract_ucsfpdgm.py \
    --src /Volumes/T7/UCSF-PDGM-v5 \
    --dst ./data/ucsf_pdgm_cohort50
```

This stages the **50-subject leak-safe stratified cohort** (see
`configs/cohort_ucsfpdgm_n50.yaml`) into a clean BIDS-ish layout:

```
data/ucsf_pdgm_cohort50/
├── sub-0005/
│   ├── sub-0005_T1_bias.nii.gz
│   ├── sub-0005_T1c_bias.nii.gz
│   ├── sub-0005_T2_bias.nii.gz
│   ├── sub-0005_FLAIR_bias.nii.gz
│   ├── sub-0005_tumor_segmentation.nii.gz
│   ├── sub-0005_brain_parenchyma_segmentation.nii.gz
│   ├── sub-0005_brain_segmentation.nii.gz
│   ├── sub-0005_T1.nii.gz                    (optional)
│   ├── sub-0005_ADC.nii.gz                   (optional)
│   ├── sub-0005_DTI_eddy_FA.nii.gz           (optional)
│   ├── sub-0005_DTI_eddy_MD.nii.gz           (optional)
│   ├── sub-0005_SWI_bias.nii.gz              (optional)
│   └── sub-0005_ASL.nii.gz                   (optional)
├── sub-0012/
└── ...
```

### LUMIERE (longitudinal illustrative case)

- Source: <https://github.com/ysuter/gbm-data-longitudinal>
  (Suter et al. 2022, *Sci Data* 9:768; DOI:
  <https://doi.org/10.1038/s41597-022-01881-7>).
  License: non-commercial use only.
- Used for the Patient-048 longitudinal panels (volume trajectories and
  the longitudinal Hit-Plot).

After downloading the LUMIERE archive locally, stage Patient-048 with:

```bash
uv run python scripts/extract_lumiere_p048.py \
    --src ~/path/to/lumiere/Patient-048 \
    --dst ./data/lumiere_p048
```

This stages the six timepoints listed in
`configs/lumiere_p048_timepoints.yaml` (the cohort-level source of truth
for the day-axis and the RANO timeline) into a clean BIDS-ish layout:

```
data/lumiere_p048/
├── manifest.json                          # provenance written by the extractor
├── tp-week-000-1/                         # day 0    (Pre-Op)
│   ├── tp-week-000-1_T1.nii.gz
│   ├── tp-week-000-1_CT1.nii.gz
│   ├── tp-week-000-1_T2.nii.gz
│   └── tp-week-000-1_FLAIR.nii.gz
├── tp-week-000-2/                         # day 3    (Post-Op, CRET)
├── tp-week-013/                           # day 91   (SD)
├── tp-week-023/                           # day 161  (CR)
├── tp-week-045/                           # day 315  (PD, T2-Progr.)
└── tp-week-049/                           # day 343  (PD, T2-Progr.)
```

Pass `--include-legacy-seg` to also stage the LUMIERE-team HD-GLIO-AUTO
and DeepBraTumIA segmentations as comparator overlays. The primary
tumour segmentation in the round-3 revision is regenerated with the
unified MONAI Bundle BraTS SegResNet on the staged channel stacks, in
keeping with the round-2 commitment to evaluate every dataset with the
same engine.

### BGO (local, not redistributed)

- Single illustrative case from the local Bergen Glioma cohort.
- Stage under `data/bgo/sub-XXXX/` with the same channel naming as UCSF-PDGM.

## Derivatives

Pipeline outputs (parcellations, predicted masks, Hit-Plot tables) land under
`data/derivatives/sub-XXXX/...`. Everything in `data/` is git-ignored.
