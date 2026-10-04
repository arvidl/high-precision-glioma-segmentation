# `paper/figs/`

Final figure and table assets included in `paper/main.tex` via
`\includegraphics{figs/...}` and `\input{figs/table*}`.

These are mirrored from `../../outputs/figures/` and `../../outputs/tables/`
(where the producer scripts under `../../scripts/` write them) by
`../../scripts/sync_figs_to_paper.py`. Nothing in this folder should be
hand-edited; rebuilding the relevant Make target + running `make sync`
(or `make paper`) regenerates the mirrored asset.

The producer chain for computed artefacts is `make figures` (the targets in
`FIGURE_TARGETS`) -> `make sync` -> `make paper`. Static Freeview panels and
third-party illustrations in this folder have no producer target. `make paper`
alone rebuilds `paper/main.pdf` from the files already here.

## Inventory: manuscript figure/table -> producer

Manuscript numbering follows `paper/main.aux` (verified with
`grep newlabel main.aux`). File numbering inside `figs/`
sometimes differs from manuscript numbering (see *Numbering caveat*
below).

### Tables

| Manuscript | Asset in `figs/`               | Producer                                                  |
|------------|--------------------------------|-----------------------------------------------------------|
| Table 1    | `table1_cohort_summary.tex`    | `scripts/build_cohort_table.py` (`make cohort-metadata`)  |
| Table 2    | (hand-typed in `main.tex`, line ~519) | none -- 5-subject characteristics table is literal LaTeX  |
| Table 3    | `table2_dl_vs_gt.tex`          | `scripts/build_table2_dl_vs_gt.py` (`make table2-dl-vs-gt`) |
| Table 4    | `table4_runtime.tex`           | `scripts/bench_runtime.py` (`make bench-runtime`)         |
| Appendix 5 table | `table_tumorsynth_agreement.tex` | `scripts/build_table_tumorsynth_agreement.py` (`make table-tumorsynth-agreement`) |

### Figures

| Manuscript  | Asset(s) in `figs/`                                                                                                       | Producer                                                                                                                                |
|-------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Fig. 1      | `glioma-localization-figs-UCSF-PDGM-0085-pipeline.png`                                                                    | legacy v1 panel composite (static); inputs are public UCSF-PDGM-0085                                                                    |
| Fig. 2      | `fig_hitplot_agreement_dl_vs_gt.pdf`                                                                                      | `scripts/build_hitplot_agreement_panel.py` (`make agreement-panel`); regenerate with `make agreement-panel && make sync`                |
| Fig. 3      | `glioma-localization-figs-UCSF-PDGM-0020_panel_[a-h].png`                                                                 | `scripts/render_figure2_panels.py` (regenerated end-to-end with FS 8.2.0; cf. provenance paragraph at `main.tex:987`)          |
| Fig. 4      | `glioma-localization-figs-UCSF-PDGM-0020-tumor-contours-on-wmparc.png` (panels a, b) + `sub-0020_panel_i_hitplot_gt.pdf` (panel c) | panels (a, b): legacy v1 `Freeview` exports (static). panel (c): `scripts/render_per_subject_hitplot_bar.py`                            |
| Fig. 5      | `glioma-localization-figs-UCSF-PDGM-0022-tumor-contours-on-wmparc.png` + `sub-0022_panel_i_hitplot_gt.pdf`                | same pattern as Fig. 4                                                                                                                  |
| Fig. 6      | `LUMIERE-Patient-048-hd-glio-synthseg.png`                                                                                | **external** -- produced with FreeSurfer 7.4.1 `recon-all-clinical` on the public LUMIERE archive; recipe not vendored in this repo     |
| Fig. 7      | `BGO-D_combined_tumor_volume_trajectories_from_preop.png`                                                                 | **external** -- produced by analysis notebooks in the (not-yet-public) `glioma-recurrence` repo, on REK-restricted BGO-D data           |
| Fig. 8      | `BGO-D_preoperative_raidionics.png`                                                                                       | **external** -- same provenance as Fig. 7                                                                                               |
| Fig. 9      | `glioma-localization-figs-calabrese-2022.png` (APPENDIX 1)                                                                | third-party (Calabrese 2022, CC BY 4.0); used as a published-data illustration                                                          |
| Fig. 10     | `glioma-localization-figs-UCSF-PDGM-0020-mpMRI.png` (APPENDIX 1)                                                          | legacy v1 panel composite (static)                                                                                                      |
| Fig. 11     | `Suter2022_fig2.png` (APPENDIX 1)                                                                                         | third-party (Suter 2022, CC BY 4.0); used as a published-data illustration                                                              |
| Figs. 12-14 | `glioma-localization-figs-UCSF-PDGM-{0039,0066,0085}-tumor-contours-on-wmparc.png` + `sub-{0039,0066,0085}_panel_i_hitplot_gt.pdf` (APPENDIX 2) | panels: legacy v1 `Freeview` exports. Hit-Plot bar: `scripts/render_per_subject_hitplot_bar.py`                                         |
| Fig. 15     | `supp_legacy5_dlvsgt_panel.pdf` (APPENDIX 3)                                                                              | `scripts/build_legacy5_dlvsgt_panel.py` (`make supp-legacy5-dlvsgt-panel`)                                                              |
| Appendix 4  | `fig_functional_anatomy_hitplot_gt.pdf`                                                                                   | `scripts/build_functional_anatomy_hitplot.py` (`make functional-anatomy-hitplot`)                                                       |
| Appendix 5  | `fig_hitplot_agreement_dl_vs_tumorsynth.pdf`                                                                              | `scripts/build_hitplot_agreement_panel.py` (`make agreement-panel AGREEMENT_REFERENCE=tumorsynth AGREEMENT_OUT_STEM=fig_hitplot_agreement_dl_vs_tumorsynth`) |
| Appendix 5  | `fig_functional_anatomy_hitplot_tumorsynth.pdf`                                                                           | `scripts/build_functional_anatomy_hitplot.py` (`make functional-anatomy-hitplot FUNCTIONAL_HITPLOT_SOURCE=tumorsynth`)                  |

### Files in `figs/` not currently referenced in the manuscript

- `fig_qc_sub0396_supplementary.pdf` -- worst-case QC exemplar from
  `notebooks/qc_segmentation_outliers.ipynb`. Generated for the round-2
  outlier audit but not included in the final manuscript. Kept as a
  supplementary asset (referenced from the QC notebook; see
  `paper/revision_tracker.md` for context).
- `UCSF-PDGM-0085_AnatLobProfileSunburstPlot.pdf` -- legacy v1 sunburst
  plot for UCSF-PDGM-0085. The corresponding `\includesvg` line in
  `main.tex` is commented out (line ~1747). Kept for
  potential future use.
- `BGO-D_{0..4}_synthseg_{Enhancing-Tumor,Edema,Tumor-postop}_radial_plot.png`
  -- per-time-point BGO-D Hit-Rate radial plots. The corresponding
  `\begin{figure}` blocks in `main.tex` (lines 1126-1178) are
  wrapped in `\begin{comment}...\end{comment}` and therefore do **not**
  appear in the published PDF. PNGs retained because the comment block
  may be re-enabled in a future revision.

## Gitignored producer outputs

`scripts/build_table3_segmenter_agreement.py` (`make table3-segmenter-agreement`)
produces a four-column segmenter-agreement table whose output
(`figs/table3_segmenter_agreement.*`) is mirrored into this folder by
`make sync` but is **not** referenced anywhere in `main.tex`.
The producer is retained as future-work scaffolding only (see
*Numbering caveat* below) and the artefacts are gitignored to keep
the repo lean.

All other figure and table assets referenced by `main.tex`
are tracked in git, so a clean clone can build the manuscript with
`latexmk paper/main.tex` without first running any producer
target.

## Numbering caveat

Two producer-side numbers in `figs/`, `scripts/`, and the `Makefile`
do **not** match the final manuscript numbers. They retain the
round-1 / pre-PR-7 numbering for diff hygiene; the mapping is:

| Producer-side                                              | Manuscript-side |
|------------------------------------------------------------|-----------------|
| `figs/table2_dl_vs_gt.tex`, `scripts/build_table2_dl_vs_gt.py`, `make table2-dl-vs-gt` | **Table 3** -- the round-2 5-subject characteristics table at `main.tex:519` is now Table 2 (hand-typed; no producer) |
| `scripts/render_figure2_panels.py`                         | **Fig. 3** -- round-2 inserted the new Hit-Plot agreement panel as Fig. 2, pushing the v1 UCSF-PDGM-0020 8-panel grid down to Fig. 3 |

We did not rename the producer files (`table2_dl_vs_gt.*`,
`render_figure2_panels.py`) to match the new manuscript numbers in
order to avoid a wide rename diff across `Makefile`, tests, and
`revision_tracker.md`. The producer-side number reflects the *order
in which the artefact was first produced*, not the position it
currently occupies in the manuscript.

`scripts/build_table3_segmenter_agreement.py` produces a four-column
segmenter-agreement table that was scoped (PR-7h) but is **not**
included in the final manuscript. Its output
(`figs/table3_segmenter_agreement.*`) is gitignored and the producer
is retained as future-work scaffolding.

## Provenance for figures produced outside this repository

Fig. 6 (LUMIERE Patient-048), Figs. 7-8 (BGO-D), and Figs. 9-13
(BGO-D per-time-point radial plots) were generated outside the public
`high-precision-glioma-segmentation` repository. Their producer code
and (for BGO-D) the underlying patient data are available through the
contact routes described in the manuscript's
*Code and data availability* section (`main.tex:1539`).
