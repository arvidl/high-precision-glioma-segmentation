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

Manuscript numbers below are those in the current `paper/main.pdf`
(floats use `[H]`, so they follow source order in `paper/main.tex`).
File names inside `figs/` sometimes keep an older producer number
(see *Numbering caveat*).

### Tables

| Manuscript | Asset in `figs/` | Producer |
| --- | --- | --- |
| Table 1 | `table1_cohort_summary.tex` | `scripts/build_cohort_table.py` (`make cohort-metadata`) |
| Table 2 | hand-typed in `main.tex` (five illustrative UCSF-PDGM subjects) | none |
| Table 3 | `table2_dl_vs_gt.tex` | `scripts/build_table2_dl_vs_gt.py` (`make table2-dl-vs-gt`) |
| Table 4 | `table4_runtime.tex` | `scripts/bench_runtime.py` (`make bench-runtime`) |
| Table 5 | hand-typed in `main.tex` (LUMIERE Patient-048 comparator Dice) | none |
| Table 6 (Appendix 5) | `table_tumorsynth_agreement.tex` | `scripts/build_table_tumorsynth_agreement.py` (`make table-tumorsynth-agreement`) |

### Figures

| Manuscript | Asset(s) in `figs/` | Producer |
| --- | --- | --- |
| Fig. 1 | `glioma-localization-figs-UCSF-PDGM-0085-pipeline.png` | static legacy composite of public UCSF-PDGM-0085 |
| Fig. 2 | `fig_hitplot_agreement_dl_vs_gt.pdf` | `scripts/build_hitplot_agreement_panel.py` (`make agreement-panel`) |
| Fig. 3 | `glioma-localization-figs-UCSF-PDGM-0020_panel_[a-h].png` | `scripts/render_figure2_panels.py` (`make figure2-legacy5` or `make figure2-render`) |
| Fig. 4 | `glioma-localization-figs-UCSF-PDGM-0020-tumor-contours-on-wmparc.png` and `sub-0020_panel_i_hitplot_gt.pdf` | contours: static Freeview export. Hit-Plot bar: `scripts/render_per_subject_hitplot_bar.py` |
| Fig. 5 | `glioma-localization-figs-UCSF-PDGM-0022-tumor-contours-on-wmparc.png` and `sub-0022_panel_i_hitplot_gt.pdf` | same pattern as Fig. 4 |
| Fig. 6 | `fig6_lumiere_p048_volumes.pdf` | `scripts/build_figure6_lumiere_p048_volumes.py` (`make figure6-lumiere-volumes`) |
| Fig. 7 | `LUMIERE-Patient-048-hd-glio-synthseg.png` | external: Freeview on FreeSurfer 7.4.1 `recon-all-clinical` outputs. The manuscript says the composition recipe is available from the corresponding author |
| Fig. 8 | `fig11_lumiere_p048_hitplot.pdf` | `scripts/build_figure11_lumiere_p048_hitplot.py` (`make figure11-lumiere-p048-hitplot`) |
| Fig. 9 (Appendix 1) | `glioma-localization-figs-calabrese-2022.png` | third-party illustration (Calabrese 2022, CC BY 4.0) |
| Fig. 10 (Appendix 1) | `glioma-localization-figs-UCSF-PDGM-0020-mpMRI.png` | static legacy composite |
| Fig. 11 (Appendix 1) | `Suter2022_fig2.png` | third-party illustration (Suter 2022, CC BY 4.0) |
| Figs. 12–14 (Appendix 2) | `glioma-localization-figs-UCSF-PDGM-0039-tumor-contours-on-synthseg.png`, `glioma-localization-figs-UCSF-PDGM-0066-tumor-contours-on-wmparc.png`, `glioma-localization-figs-UCSF-PDGM-0085-tumor-contours-on-wmparc.png`, and `sub-{0039,0066,0085}_panel_i_hitplot_gt.pdf` | contours: static Freeview exports. Hit-Plot bars: `scripts/render_per_subject_hitplot_bar.py` |
| Fig. 15 (Appendix 3) | `supp_legacy5_dlvsgt_panel.pdf` | `scripts/build_legacy5_dlvsgt_panel.py` (`make supp-legacy5-dlvsgt-panel`) |
| Fig. 16 (Appendix 4) | `fig_functional_anatomy_hitplot_gt.pdf` | `scripts/build_functional_anatomy_hitplot.py` (`make functional-anatomy-hitplot`) |
| Fig. 17 (Appendix 5) | `fig_hitplot_agreement_dl_vs_tumorsynth.pdf` | `scripts/build_hitplot_agreement_panel.py` (`make agreement-panel AGREEMENT_REFERENCE=tumorsynth`) |
| Fig. 18 (Appendix 5) | `fig_functional_anatomy_hitplot_tumorsynth.pdf` | `scripts/build_functional_anatomy_hitplot.py` (`make functional-anatomy-hitplot FUNCTIONAL_HITPLOT_SOURCE=tumorsynth`) |

### Files in `figs/` that `main.tex` does not include

These files are leftover assets. They are not figures in the current
manuscript or in the journal PDF.

- `BGO-D_combined_tumor_volume_trajectories_from_preop.png` and
  `BGO-D_preoperative_raidionics.png` — drafts from an earlier Bergen
  glioma cohort analysis. No BGO figure remains in `main.tex`.
- `fig_qc_sub0396_supplementary.pdf` — QC exemplar from
  `notebooks/qc_segmentation_outliers.ipynb`.
- Older composites and unused panel exports
  (`glioma-localization-figs-UCSF-PDGM-*_a_c.png` and the paired
  `_b_d`, `_e_g`, `_f_h` sheets; `*-hitplot-on-wmparc.png`; per-subject
  `*_panel_[a-h].png` files other than the eight UCSF-PDGM-0020 panels
  in Figure 3).
- JSON sidecars next to Figures 6 and 8
  (`fig6_lumiere_p048_volumes.json`, `fig11_lumiere_p048_hitplot.json`).

## Gitignored producer outputs

`scripts/build_table3_segmenter_agreement.py` (`make table3-segmenter-agreement`)
produces a four-column segmenter-agreement table. `make sync` can mirror
`table3_segmenter_agreement.*` into this folder, and `main.tex` does not
include it. The producer is future-work scaffolding. The artefacts are
gitignored.

Every figure and table that `main.tex` includes is tracked in git, so a
clean clone can build the manuscript with `make paper` without first
running a producer target.

## Numbering caveat

Two producer-side names do not match the manuscript numbers. They keep
the name from the commit that first produced the artefact:

| Producer-side | Manuscript |
| --- | --- |
| `figs/table2_dl_vs_gt.tex`, `scripts/build_table2_dl_vs_gt.py`, `make table2-dl-vs-gt` | **Table 3**. Table 2 is the hand-typed five-subject characteristics table |
| `scripts/render_figure2_panels.py`, `make figure2-legacy5`, `make figure2-render` | **Figure 3**. Figure 2 is the Hit-Plot agreement panel |
| `fig6_lumiere_p048_volumes.*`, `make figure6-lumiere-volumes` | **Figure 6** |
| `fig11_lumiere_p048_hitplot.*`, `make figure11-lumiere-p048-hitplot` | **Figure 8** |

`scripts/build_table3_segmenter_agreement.py` is the unused
segmenter-agreement table described above. It is not manuscript Table 3.
