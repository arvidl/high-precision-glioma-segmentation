# Notebooks

Each notebook is a thin shim over `src/hpgs` and is regenerable end-to-end. Notebook
outputs are stripped on commit (`nbstripout` pre-commit hook).

| #  | Notebook                                          | Purpose |
|----|---------------------------------------------------|---------|
| 00 | `00_quickstart_single_subject.ipynb`              | Load one UCSF-PDGM subject, view channels, run a minimal pipeline. |
| 10 | `10_cohort_preprocessing.ipynb`                   | Stage the n=50 cohort, compute summary table (Table 1). |
| 20 | `20_tumor_segmentation_fastMONAI.ipynb`           | fastMONAI multiclass segmentation tutorial adapted to BraTS-style labels. |
| 30 | `30_benchmark_segmentglioma_vs_raidionics.ipynb`  | Head-to-head benchmark with Dice/HD95/VE/Sens/Spec. |
| 40 | `40_hitplot_generation.ipynb`                     | Per-parcel tumor-burden tables and Hit-Plot figures. |
| 50 | `50_quantitative_metrics.ipynb`                   | Pooled metrics, plots, statistics — answers R2-2 and R2-10. |
| 99 | `99_reproduce_paper_figures.ipynb`                | One-shot regeneration of every figure and table in the manuscript. |

Until the .ipynb files are committed, sketch each notebook as a script first under
`notebooks/scratch/`, then convert with `jupytext` or `nbconvert`. This keeps Cursor
diffs reviewable.
