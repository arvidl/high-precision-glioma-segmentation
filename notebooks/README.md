# Notebooks

Each notebook is a thin shim over `src/hpgs`. Notebook outputs are stripped on
commit (`nbstripout` pre-commit hook).

The teaching notebook on the current `main` branch is
[`00_quickstart_single_subject.ipynb`](00_quickstart_single_subject.ipynb):
subject 0005, the four MRI channels, the MONAI BraTS segmenter, and Dice
against the dataset reference. It is not in tag `v1.0.0`. It needs the
extracted cohort described in [`../data/README.md`](../data/README.md).

| Notebook | In the repository | Purpose |
|---|---|---|
| `00_quickstart_single_subject.ipynb` | yes | One subject: channels, segmentation, enhancing-tumor Dice. |
| `qc_segmentation_outliers.ipynb` | yes | QC audit of cohort segmentation outliers. |
| `10`–`50`, `99` | no | Further lessons and a one-shot figure rebuild. Figure regeneration is `make figures`. |
