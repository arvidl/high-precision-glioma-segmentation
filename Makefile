# Top-level Makefile for high-precision-glioma-segmentation.
# Orchestrates code → figures → paper.

PY        ?= uv run python
PYTEST    ?= uv run pytest
RUFF      ?= uv run ruff

# Computed manuscript artefacts. Each prerequisite is an existing target.
# Static Freeview and third-party panels in paper/figs/ have no producer here.
FIGURE_TARGETS := cohort-metadata table2-dl-vs-gt bench-runtime \
                  agreement-panel functional-anatomy-hitplot figs-anat-profile \
                  figure6-lumiere-volumes figure11-lumiere-p048-hitplot \
                  table-tumorsynth-agreement

.PHONY: help install test lint format figures figure2-legacy5 figure2-render \
        cohort-metadata smoke-segment scope-pr7 parcellate-all segment-all \
        baselines-all tumorsynth-all hitplot-all table2-dl-vs-gt table3-segmenter-agreement \
        table-tumorsynth-agreement \
        agreement-panel lobe-heatmap functional-anatomy-hitplot bench-runtime sync paper docs all clean \
        legacy5-hitplot-gt legacy5-figs-anat-profile supp-legacy5-dlvsgt-panel \
        figs-anat-profile figure6-lumiere-volumes \
        summarize-lumiere-p048-segmentation parcellate-lumiere-p048 \
        hitplot-lumiere-p048 figure11-lumiere-p048-hitplot

FIGURE2_CONFIG   ?= configs/figure2_legacy5.yaml
FIGURE2_FS_CONFIG ?= configs/legacy5.yaml
FIGURE2_THREADS  ?= 6

COHORT_YAML      ?= configs/cohort_ucsfpdgm_n50.yaml
COHORT_CSV       ?= data/UCSF-PDGM-metadata_v5.csv

help:
	@echo "Targets:"
	@echo "  install             uv sync the project (runtime + dev)"
	@echo "  test                run pytest"
	@echo "  lint                ruff check"
	@echo "  format              ruff format"
	@echo "  figures             regenerate computed manuscript figures and tables (needs staged data; see docs/software_installation.md)"
	@echo "  figure2-legacy5     regenerate FS 8.2.0 derivatives + Figure 2 panels a..h for all 5 legacy subjects"
	@echo "  figure2-render      regenerate Figure 2 panels a..h only (skip FS clinical stage)"
	@echo "  cohort-metadata     join cohort YAML + UCSF-PDGM metadata CSV -> Table 1"
	@echo "  smoke-segment       run unified BraTS3 segmenter (MONAI bundle) on 3 cohort subjects (PR-8 smoke)"
	@echo "  scope-pr7           dry-run scope check: enumerate the n=50 cohort and the planned per-subject artefact paths (no inference)"
	@echo "  parcellate-all      PR-7c real fan-out: FreeSurfer 8.2.0 recon-all-clinical wmparc on every cohort subject, resampled onto the native mpMRI grid; writes parcellation/{wmparc_native.nii.gz, wmparc_lut.json, wmparc.json}"
	@echo "  segment-all         PR-7d real fan-out: unified BraTS3 segmenter on every cohort subject; writes seg_dl/ + metrics/dl_vs_gt JSON per Section 4 of docs/scope_pr7.md"
	@echo "  baselines-all       PR-7e real fan-out: Raidionics + segment_glioma on every cohort subject; writes seg_<baseline>/ + metrics/dl_vs_<baseline> JSONs"
	@echo "  tumorsynth-all      Optional comparator: run mri_TumorSynth on the same cohort inputs; writes seg_tumorsynth/ + metrics/dl_vs_tumorsynth JSONs"
	@echo "  hitplot-all         PR-7f real fan-out: produce Hit-Plot CSVs per subject, including optional TumorSynth when available"
	@echo "  table2-dl-vs-gt     PR-7g: aggregate metrics/dl_vs_gt JSONs into Table 2 (median [IQR] per compartment x metric); writes outputs/tables/table2_dl_vs_gt.{tex,csv,json}"
	@echo "  table3-segmenter-agreement  PR-7h: aggregate the three metrics/dl_vs_{raidionics,segmentglioma,gt} JSONs into Table 3 (4-col LaTeX: Metric (comp) x {Raidionics, segment_glioma, BraTS21-manual}); writes outputs/tables/table3_segmenter_agreement.{tex,csv,json}"
	@echo "  table-tumorsynth-agreement  Optional comparator: aggregate metrics/dl_vs_tumorsynth JSONs into a standalone TumorSynth table; writes outputs/tables/table_tumorsynth_agreement.{tex,csv,json}"
	@echo "  agreement-panel     PR-7i: build the Hit-Plot agreement panel (Bland-Altman per (subject, region) cell, CCC + mean |dV| per compartment); writes outputs/figures/fig_hitplot_agreement_dl_vs_gt.{pdf,csv,json}"
	@echo "  lobe-heatmap        PR-7l: cohort-level successor to Figure 15: lobe x compartment glioma-distribution heatmap (median share of compartment volume in lobe + per-cell prevalence n=K/N); writes outputs/figures/fig_lobe_compartment_heatmap_<source>.{pdf,csv,json}"
	@echo "  functional-anatomy-hitplot  exploratory appendix: wmparc label-adjacency Hit-Plot groups (lobar + motor/language/deep-midline) with OS-tertile panels; writes outputs/figures/fig_functional_anatomy_hitplot_<source>.{pdf,csv,json}"
	@echo "  bench-runtime       aggregate per-subject elapsed_s from seg_dl + parcellation sidecars into Table 4 (per-pipeline-stage wall-clock); writes outputs/tables/table4_runtime.{tex,csv,json}"
	@echo "  legacy5-hitplot-gt  Phase C: compute the GT-only Hit-Plot CSV per legacy-5 detail subject (Figs. 4, 5, 12, 13, 14 main-text bar chart); writes data/derivatives_legacy5/sub-XXXX/hitplot/sub-XXXX_hitplot_gt.csv"
	@echo "  legacy5-figs-anat-profile  Phase C: render the per-subject (i)-panel bar chart for all 5 legacy-5 detail subjects from the GT Hit-Plot CSVs; writes outputs/figures/legacy5/sub-XXXX_panel_i_hitplot_gt.{pdf,png}"
	@echo "  supp-legacy5-dlvsgt-panel  Phase C: build the supplementary 5x1 DL-vs-GT Hit-Plot panel from 5 leak-safe n=50 cohort subjects (S-D rule); writes outputs/figures/supp_legacy5_dlvsgt_panel.{pdf,png,json}"
	@echo "  figs-anat-profile   Phase C umbrella: figure2-render + legacy5-hitplot-gt + legacy5-figs-anat-profile + supp-legacy5-dlvsgt-panel"
	@echo "  figure6-lumiere-volumes  R3 Fig. 6 replacement: longitudinal tumour-volume trajectory + RANO panel for LUMIERE Patient-048; reads data/lumiere_p048/derivatives/segmentation_summary.csv + configs/lumiere_p048_timepoints.yaml; writes outputs/figures/fig6_lumiere_p048_volumes.{pdf,png,json}"
	@echo "  summarize-lumiere-p048-segmentation  R3 Fig. 6 QC: per-tp segmentation table + DL-vs-{HD-GLIO, DeepBraTumIA} Dice/JSC for LUMIERE Patient-048; reads data/lumiere_p048/derivatives/; writes outputs/inventory/lumiere_p048_segmentation_summary.{json,md}"
	@echo "  parcellate-lumiere-p048  R3 Step G.2: real FreeSurfer 8.2.0 recon-all-clinical fan-out across all 6 LUMIERE Patient-048 timepoints; writes per-tp parcellation/{wmparc_native.nii.gz, wmparc_lut.json, wmparc.json} + parcellation_summary.{csv,json}. ~3-12 h CPU. Override LUMIERE_PARCELLATE_BACKEND=dummy for the network-free smoke."
	@echo "  hitplot-lumiere-p048  R3 Step G.3: per-tp Hit-Plot CSVs for LUMIERE Patient-048 (DL hard, DL probs, HD-GLIO, DeepBraTumIA) joining the FS wmparc with the four segmentation sources; writes per-tp hitplot/tp-<week>_hitplot_<source>.csv + hitplot_summary.{csv,json}. Requires Step G.2 outputs on disk (~2 min for the full 6 tp x 4 sources fan-out)."
	@echo "  figure11-lumiere-p048-hitplot  R3 Step G.4a: build Figure 11 (longitudinal Hit-Plot heatmap, top-K wmparc regions x 6 tp, two-panel WT/ET) from the per-tp DL Hit-Plot CSVs; writes outputs/figures/fig11_lumiere_p048_hitplot.{pdf,png,json}. The sync target mirrors them into paper/figs/."
	@echo "  sync                copy outputs/figures and outputs/tables -> paper/figs"
	@echo "  paper               build paper/main.pdf from the committed manuscript sources"
	@echo "  docs                render docs/*.md design notes to docs/*.pdf via pandoc"
	@echo "  all                 figures + sync + paper"
	@echo "  clean               remove TeX build artefacts and outputs/"

install:
	uv python install 3.11
	uv sync --python 3.11 --extra dev
	uv run --python 3.11 pre-commit install

test:
	$(PYTEST) -q -m "not slow and not gpu"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

figures: $(FIGURE_TARGETS)

figure2-legacy5:
	$(PY) scripts/run_figure2_legacy5.py \
	    --config $(FIGURE2_CONFIG) \
	    --fs-config $(FIGURE2_FS_CONFIG) \
	    --threads $(FIGURE2_THREADS)

figure2-render:
	$(PY) scripts/run_figure2_legacy5.py \
	    --config $(FIGURE2_CONFIG) \
	    --fs-config $(FIGURE2_FS_CONFIG) \
	    --threads $(FIGURE2_THREADS) \
	    --skip-fs

cohort-metadata:
	$(PY) scripts/build_cohort_table.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV)

# PR-8 smoke test: run the unified MONAI Bundle BraTS3 segmenter on 3 cohort
# subjects and print Dice / |VE| against the dataset-provided reference masks.
# Override SMOKE_SUBJECTS, SMOKE_BACKEND or SMOKE_DEVICE on the command line:
#   make smoke-segment SMOKE_SUBJECTS="0005 0026" SMOKE_DEVICE=mps
SMOKE_SUBJECTS  ?=
SMOKE_BACKEND   ?= monai_bundle
SMOKE_DEVICE    ?= auto

smoke-segment:
	$(PY) scripts/smoke_predict_brats3.py \
	    --backend $(SMOKE_BACKEND) \
	    --device $(SMOKE_DEVICE) \
	    $(foreach s,$(SMOKE_SUBJECTS),--subject $(s))

# PR-7 scoping smoke check: dry-run the cohort fan-out runner on the locked
# cohort YAML (configs/cohort_ucsfpdgm_n50.yaml). Enumerates the 50 subjects,
# checks that every required input NIfTI is on disk, and prints the per-subject
# artefact paths that the real PR-7 runner will write. No inference.
# Override SCOPE_SHOW_PATHS=1 for a verbose per-subject WOULD-WRITE listing.
SCOPE_SHOW_PATHS ?=

scope-pr7:
	$(PY) scripts/segment_all_cohort50.py \
	    --dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    $(if $(SCOPE_SHOW_PATHS),--show-paths,)

# PR-7c real cohort fan-out: run FreeSurfer 8.2.0 recon-all-clinical on
# every subject in the locked cohort YAML, resample wmparc.mgz onto the
# native mpMRI grid (nearest-neighbour), and write the Section 4
# parcellation contract (wmparc_native.nii.gz + wmparc_lut.json +
# provenance sidecar).  FreeSurfer takes 30 min--2 h per subject on
# CPU; this is the long pole of the PR-7 pipeline (see
# docs/scope_pr7.md § 6). Override PARCELLATE_BACKEND=dummy for a
# CI-runnable synthetic parcellation that does not require FreeSurfer
# on the host (used by the test suite and by smoke pipelines).
#     make parcellate-all PARCELLATE_FS_WORK_DIR=/scratch/fs_subjects
#     make parcellate-all PARCELLATE_BACKEND=dummy PARCELLATE_SUBJECTS="0005 0026"
# Pass PARCELLATE_SKIP_EXISTING=1 to resume a partial run.
PARCELLATE_BACKEND       ?= freesurfer
PARCELLATE_INPUT_CHANNEL ?= T1_bias
PARCELLATE_FS_WORK_DIR   ?=
PARCELLATE_THREADS       ?= 4
PARCELLATE_SUBJECTS      ?=
PARCELLATE_SKIP_EXISTING ?=

parcellate-all:
	$(PY) scripts/parcellate_all_cohort50.py \
	    --no-dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --backend $(PARCELLATE_BACKEND) \
	    --input-channel $(PARCELLATE_INPUT_CHANNEL) \
	    --threads $(PARCELLATE_THREADS) \
	    $(if $(PARCELLATE_FS_WORK_DIR),--fs-work-dir $(PARCELLATE_FS_WORK_DIR),) \
	    $(foreach s,$(PARCELLATE_SUBJECTS),--subject $(s)) \
	    $(if $(PARCELLATE_SKIP_EXISTING),--skip-existing,)

# PR-7d real cohort fan-out: run the unified MONAI Bundle BraTS3 segmenter
# on every subject in the locked cohort YAML and write the Section 4
# derivatives layout (seg_dl/ + metrics/dl_vs_gt JSON). Override
# SEGMENT_DEVICE on the command line for Apple-silicon (mps) or CUDA:
#     make segment-all SEGMENT_DEVICE=mps
#     make segment-all SEGMENT_SUBJECTS="0005 0026" SEGMENT_DEVICE=cpu
# Pass SEGMENT_SKIP_EXISTING=1 to resume a partial run without re-doing
# subjects whose seg_dl outputs are already on disk.
SEGMENT_BACKEND       ?= monai_bundle
SEGMENT_DEVICE        ?= auto
SEGMENT_SUBJECTS      ?=
SEGMENT_SKIP_EXISTING ?=

segment-all:
	$(PY) scripts/segment_all_cohort50.py \
	    --no-dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --backend $(SEGMENT_BACKEND) \
	    --device $(SEGMENT_DEVICE) \
	    $(foreach s,$(SEGMENT_SUBJECTS),--subject $(s)) \
	    $(if $(SEGMENT_SKIP_EXISTING),--skip-existing,)

# PR-7e baseline fan-out: run Raidionics and segment_glioma on every
# cohort subject, write seg_<baseline>/ masks + metrics/dl_vs_<baseline>
# JSONs that the PR-7h Table 3 producer and the PR-7i Hit-Plot agreement
# panel will consume. TumorSynth is also available via BASELINE_BASELINES
# with BASELINE_BACKEND=command, but the dedicated tumorsynth-all target
# below is clearer for operator runs.
#     make baselines-all BASELINE_BACKEND=dummy
#     make baselines-all BASELINE_BACKEND=docker BASELINE_SUBJECTS="0005 0026"
#     make baselines-all BASELINE_BACKEND=command BASELINE_BASELINES=tumorsynth TUMORSYNTH_NNUNET_DIR=/path/to/nnUNet
# Pass BASELINE_SKIP_EXISTING=1 to resume a partial run; pass
# BASELINE_REQUIRE_DL=0 to write baseline masks even when the PR-7d DL
# output for a subject is missing (the paired-metrics JSON is then
# skipped with a warning instead of failing the subject).
BASELINE_BACKEND        ?= docker
BASELINE_BASELINES      ?=
BASELINE_SUBJECTS       ?=
BASELINE_SKIP_EXISTING  ?=
BASELINE_REQUIRE_DL     ?= 1
TUMORSYNTH_COMMAND      ?= mri_TumorSynth
TUMORSYNTH_NNUNET_DIR   ?=
TUMORSYNTH_THREADS      ?= 8
TUMORSYNTH_WORK_DIR     ?=
TUMORSYNTH_CPU          ?=
TUMORSYNTH_WHOLE_LABEL  ?= 18
TUMORSYNTH_NCR_LABELS   ?= 3
TUMORSYNTH_ED_LABELS    ?= 1
TUMORSYNTH_ET_LABELS    ?= 2

baselines-all:
	$(PY) scripts/run_baseline_segmenters_cohort50.py \
	    --no-dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --backend $(BASELINE_BACKEND) \
	    $(foreach b,$(BASELINE_BASELINES),--baseline $(b)) \
	    $(foreach s,$(BASELINE_SUBJECTS),--subject $(s)) \
	    $(if $(BASELINE_SKIP_EXISTING),--skip-existing,) \
	    $(if $(TUMORSYNTH_COMMAND),--tumorsynth-command $(TUMORSYNTH_COMMAND),) \
	    $(if $(TUMORSYNTH_NNUNET_DIR),--tumorsynth-nnunet-dir $(TUMORSYNTH_NNUNET_DIR),) \
	    --tumorsynth-threads $(TUMORSYNTH_THREADS) \
	    $(if $(TUMORSYNTH_WORK_DIR),--tumorsynth-work-dir $(TUMORSYNTH_WORK_DIR),) \
	    $(if $(TUMORSYNTH_CPU),--tumorsynth-cpu,) \
	    --tumorsynth-whole-tumor-label $(TUMORSYNTH_WHOLE_LABEL) \
	    --tumorsynth-inner-ncr-labels $(TUMORSYNTH_NCR_LABELS) \
	    --tumorsynth-inner-ed-labels $(TUMORSYNTH_ED_LABELS) \
	    --tumorsynth-inner-et-labels $(TUMORSYNTH_ET_LABELS) \
	    $(if $(filter 0,$(BASELINE_REQUIRE_DL)),--no-require-dl,--require-dl)

# Optional mri_TumorSynth comparator on the same four staged mpMRI inputs as
# the MONAI Bundle run. Requires a local TumorSynth installation plus the
# nnU-Net v1.7 model root (pass TUMORSYNTH_NNUNET_DIR or export NNUNET_ENV_DIR).
# Example:
#     make tumorsynth-all TUMORSYNTH_NNUNET_DIR=/opt/nnUNet TUMORSYNTH_CPU=1
tumorsynth-all:
	$(PY) scripts/run_baseline_segmenters_cohort50.py \
	    --no-dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --backend command \
	    --baseline tumorsynth \
	    $(foreach s,$(BASELINE_SUBJECTS),--subject $(s)) \
	    $(if $(BASELINE_SKIP_EXISTING),--skip-existing,) \
	    $(if $(TUMORSYNTH_COMMAND),--tumorsynth-command $(TUMORSYNTH_COMMAND),) \
	    $(if $(TUMORSYNTH_NNUNET_DIR),--tumorsynth-nnunet-dir $(TUMORSYNTH_NNUNET_DIR),) \
	    --tumorsynth-threads $(TUMORSYNTH_THREADS) \
	    $(if $(TUMORSYNTH_WORK_DIR),--tumorsynth-work-dir $(TUMORSYNTH_WORK_DIR),) \
	    $(if $(TUMORSYNTH_CPU),--tumorsynth-cpu,) \
	    --tumorsynth-whole-tumor-label $(TUMORSYNTH_WHOLE_LABEL) \
	    --tumorsynth-inner-ncr-labels $(TUMORSYNTH_NCR_LABELS) \
	    --tumorsynth-inner-ed-labels $(TUMORSYNTH_ED_LABELS) \
	    --tumorsynth-inner-et-labels $(TUMORSYNTH_ET_LABELS) \
	    $(if $(filter 0,$(BASELINE_REQUIRE_DL)),--no-require-dl,--require-dl)

# PR-7f: drive compartment_region_matrix + probabilistic_hitplot across
# every cohort subject, reading the PR-7c parcellation + PR-7d DL seg +
# PR-7d DL softmax probs + PR-7e baseline masks + dataset GT, and
# writing the Hit-Plot CSVs per Section 4 of docs/scope_pr7.md, including
# optional TumorSynth when seg_tumorsynth/ exists.
# Missing parcellation artefacts skip the subject with a warning by
# default; set HITPLOT_REQUIRE_PARCELLATION=0 to silence those skips
# (useful before PR-7c ships). Set HITPLOT_SKIP_EXISTING=1 to resume a
# partial cohort run without regenerating CSVs already on disk.
#     make hitplot-all
#     make hitplot-all HITPLOT_SOURCES="dl dl_prob" HITPLOT_SUBJECTS="0005 0026"
#     make hitplot-all HITPLOT_REQUIRE_PARCELLATION=0
HITPLOT_COHORT_ROOT          ?= data/ucsf_pdgm_cohort50
HITPLOT_DERIV_ROOT           ?= data/derivatives_cohort50
HITPLOT_SOURCES              ?=
HITPLOT_SUBJECTS             ?=
HITPLOT_SKIP_EXISTING        ?=
HITPLOT_REQUIRE_PARCELLATION ?= 1

hitplot-all:
	$(PY) scripts/compute_hitplot_cohort50.py \
	    --no-dry-run \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --cohort-root $(HITPLOT_COHORT_ROOT) \
	    --deriv-root $(HITPLOT_DERIV_ROOT) \
	    $(foreach s,$(HITPLOT_SOURCES),--source $(s)) \
	    $(foreach s,$(HITPLOT_SUBJECTS),--subject $(s)) \
	    $(if $(HITPLOT_SKIP_EXISTING),--skip-existing,) \
	    $(if $(filter 0,$(HITPLOT_REQUIRE_PARCELLATION)),--no-require-parcellation,--require-parcellation)

# PR-7g: build Table 2 (DL vs UCSF-PDGM reference) from the per-subject
# metrics_dl_vs_gt.json sidecars written by segment-all. Override
# TABLE2_STRICT=1 to make the run fail on the first missing / malformed
# sidecar; otherwise missing sidecars are warned about and excluded.
TABLE2_DERIV_ROOT ?= data/derivatives_cohort50
TABLE2_OUT_TEX    ?= outputs/tables/table2_dl_vs_gt.tex
TABLE2_OUT_CSV    ?= outputs/tables/table2_dl_vs_gt.csv
TABLE2_OUT_JSON   ?= outputs/tables/table2_dl_vs_gt.json
TABLE2_STRICT     ?=

table2-dl-vs-gt:
	$(PY) scripts/build_table2_dl_vs_gt.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --metrics-root $(TABLE2_DERIV_ROOT) \
	    --out-tex $(TABLE2_OUT_TEX) \
	    --out-csv $(TABLE2_OUT_CSV) \
	    --out-json $(TABLE2_OUT_JSON) \
	    $(if $(TABLE2_STRICT),--strict,)

# PR-7h: aggregate the three PR-7d/PR-7e paired-metrics JSONs into the
# four-column Table 3 (Raidionics | segment_glioma | BraTS21-manual).
# Knobs mirror table2-dl-vs-gt; set TABLE3_STRICT=1 to make the run
# fail on the first missing / malformed sidecar in any of the three
# pair families.
TABLE3_DERIV_ROOT ?= data/derivatives_cohort50
TABLE3_OUT_TEX    ?= outputs/tables/table3_segmenter_agreement.tex
TABLE3_OUT_CSV    ?= outputs/tables/table3_segmenter_agreement.csv
TABLE3_OUT_JSON   ?= outputs/tables/table3_segmenter_agreement.json
TABLE3_STRICT     ?=

table3-segmenter-agreement:
	$(PY) scripts/build_table3_segmenter_agreement.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --metrics-root $(TABLE3_DERIV_ROOT) \
	    --out-tex $(TABLE3_OUT_TEX) \
	    --out-csv $(TABLE3_OUT_CSV) \
	    --out-json $(TABLE3_OUT_JSON) \
	    $(if $(TABLE3_STRICT),--strict,)

# Optional TumorSynth comparator: aggregate the per-subject paired metrics
# written by tumorsynth-all into a standalone table. Kept separate from
# table3-segmenter-agreement because TumorSynth is framed as a bounded
# sensitivity analysis, not part of the original Table 3 comparator set.
TUMORSYNTH_TABLE_DERIV_ROOT ?= data/derivatives_cohort50
TUMORSYNTH_TABLE_OUT_TEX    ?= outputs/tables/table_tumorsynth_agreement.tex
TUMORSYNTH_TABLE_OUT_CSV    ?= outputs/tables/table_tumorsynth_agreement.csv
TUMORSYNTH_TABLE_OUT_JSON   ?= outputs/tables/table_tumorsynth_agreement.json
TUMORSYNTH_TABLE_STRICT     ?=

table-tumorsynth-agreement:
	$(PY) scripts/build_table_tumorsynth_agreement.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --metrics-root $(TUMORSYNTH_TABLE_DERIV_ROOT) \
	    --out-tex $(TUMORSYNTH_TABLE_OUT_TEX) \
	    --out-csv $(TUMORSYNTH_TABLE_OUT_CSV) \
	    --out-json $(TUMORSYNTH_TABLE_OUT_JSON) \
	    $(if $(TUMORSYNTH_TABLE_STRICT),--strict,)

# PR-7i: build the Hit-Plot agreement panel from the PR-7f per-subject
# Hit-Plot CSVs (pred + ref). Default comparison is dl-vs-gt (the
# headline figure for R2.8); flip --prediction / --reference to point
# the same producer at any other (pred, ref) pair from the accepted-
# source set {dl, gt, raidionics, segmentglioma}.  Set
# AGREEMENT_STRICT=1 to fail on the first missing / malformed CSV;
# omit it for a tolerant run that produces the panel from whatever
# subjects are available and reports the rest as warnings.
AGREEMENT_HITPLOT_ROOT ?= data/derivatives_cohort50
AGREEMENT_PREDICTION   ?= dl
AGREEMENT_REFERENCE    ?= gt
AGREEMENT_OUT_PDF      ?= outputs/figures/fig_hitplot_agreement_$(AGREEMENT_PREDICTION)_vs_$(AGREEMENT_REFERENCE).pdf
AGREEMENT_OUT_JSON     ?= outputs/figures/fig_hitplot_agreement_$(AGREEMENT_PREDICTION)_vs_$(AGREEMENT_REFERENCE).json
AGREEMENT_OUT_CSV      ?= outputs/figures/fig_hitplot_agreement_$(AGREEMENT_PREDICTION)_vs_$(AGREEMENT_REFERENCE).csv
AGREEMENT_STRICT       ?=

agreement-panel:
	$(PY) scripts/build_hitplot_agreement_panel.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --hitplot-root $(AGREEMENT_HITPLOT_ROOT) \
	    --prediction $(AGREEMENT_PREDICTION) \
	    --reference $(AGREEMENT_REFERENCE) \
	    --out-pdf $(AGREEMENT_OUT_PDF) \
	    --out-json $(AGREEMENT_OUT_JSON) \
	    --out-csv $(AGREEMENT_OUT_CSV) \
	    $(if $(AGREEMENT_STRICT),--strict,)

# PR-7l: cohort lobe x compartment glioma-distribution heatmap. The
# cohort-level successor to the n=1 UCSF-PDGM-0085 sunburst (Figure 15
# in the original manuscript). Reads the per-subject PR-7f Hit-Plot
# CSVs + the matching wmparc parcellation LUTs and writes a heatmap +
# audit JSON + audit CSV under outputs/figures/. Default source is the
# UCSF-PDGM reference (gt); set LOBE_HEATMAP_SOURCE=dl for the same
# producer pointed at the deep-learning prediction (sensitivity check).
LOBE_HEATMAP_HITPLOT_ROOT ?= data/derivatives_cohort50
LOBE_HEATMAP_SOURCE       ?= gt
LOBE_HEATMAP_OUT_PDF      ?= outputs/figures/fig_lobe_compartment_heatmap_$(LOBE_HEATMAP_SOURCE).pdf
LOBE_HEATMAP_OUT_JSON     ?= outputs/figures/fig_lobe_compartment_heatmap_$(LOBE_HEATMAP_SOURCE).json
LOBE_HEATMAP_OUT_CSV      ?= outputs/figures/fig_lobe_compartment_heatmap_$(LOBE_HEATMAP_SOURCE).csv
LOBE_HEATMAP_STRICT       ?=

lobe-heatmap:
	$(PY) scripts/build_lobe_compartment_heatmap.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --hitplot-root $(LOBE_HEATMAP_HITPLOT_ROOT) \
	    --source $(LOBE_HEATMAP_SOURCE) \
	    --out-pdf $(LOBE_HEATMAP_OUT_PDF) \
	    --out-json $(LOBE_HEATMAP_OUT_JSON) \
	    --out-csv $(LOBE_HEATMAP_OUT_CSV) \
	    $(if $(LOBE_HEATMAP_STRICT),--strict,)

# Exploratory appendix: functional-anatomy Hit-Plot summaries using only
# existing wmparc labels (no tractography / fMRI / OAR proximity / outcome
# model). Default source is the UCSF-PDGM reference (gt); set
# FUNCTIONAL_HITPLOT_SOURCE=dl or tumorsynth for sensitivity runs.
FUNCTIONAL_HITPLOT_ROOT     ?= data/derivatives_cohort50
FUNCTIONAL_HITPLOT_SOURCE   ?= gt
FUNCTIONAL_HITPLOT_OUT_PDF  ?= outputs/figures/fig_functional_anatomy_hitplot_$(FUNCTIONAL_HITPLOT_SOURCE).pdf
FUNCTIONAL_HITPLOT_OUT_JSON ?= outputs/figures/fig_functional_anatomy_hitplot_$(FUNCTIONAL_HITPLOT_SOURCE).json
FUNCTIONAL_HITPLOT_OUT_CSV  ?= outputs/figures/fig_functional_anatomy_hitplot_$(FUNCTIONAL_HITPLOT_SOURCE).csv
FUNCTIONAL_HITPLOT_STRICT   ?=
FUNCTIONAL_HITPLOT_NO_OS    ?=

functional-anatomy-hitplot:
	$(PY) scripts/build_functional_anatomy_hitplot.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --hitplot-root $(FUNCTIONAL_HITPLOT_ROOT) \
	    --source $(FUNCTIONAL_HITPLOT_SOURCE) \
	    --out-pdf $(FUNCTIONAL_HITPLOT_OUT_PDF) \
	    --out-json $(FUNCTIONAL_HITPLOT_OUT_JSON) \
	    --out-csv $(FUNCTIONAL_HITPLOT_OUT_CSV) \
	    $(if $(FUNCTIONAL_HITPLOT_STRICT),--strict,) \
	    $(if $(FUNCTIONAL_HITPLOT_NO_OS),--no-os-tertile,)

# PR-12 (R2.12): aggregate the per-subject elapsed_s values already
# captured in the seg_dl/ (PR-7d) and parcellation/ (PR-7c) sidecars
# into Table 4 (per-pipeline-stage wall-clock on the host that ran
# the cohort). Tolerates partial parcellation runs --- the FreeSurfer
# row reports n_used/n_expected in the cell when the fan-out is still
# in flight.
TABLE4_DERIV_ROOT ?= data/derivatives_cohort50
TABLE4_OUT_TEX    ?= outputs/tables/table4_runtime.tex
TABLE4_OUT_CSV    ?= outputs/tables/table4_runtime.csv
TABLE4_OUT_JSON   ?= outputs/tables/table4_runtime.json

bench-runtime:
	$(PY) scripts/bench_runtime.py \
	    --cohort-yaml $(COHORT_YAML) \
	    --metadata-csv $(COHORT_CSV) \
	    --derivatives-root $(TABLE4_DERIV_ROOT) \
	    --out-tex $(TABLE4_OUT_TEX) \
	    --out-csv $(TABLE4_OUT_CSV) \
	    --out-json $(TABLE4_OUT_JSON)

# Phase C (R2.6 / R2.8 follow-up): per-subject anatomical-profile bar
# charts for the 5 legacy detail subjects + a supplementary 5x1
# DL-vs-GT panel built from 5 leak-safe n=50 cohort subjects (S-D
# rule). The legacy-5 main-text panels are GT-only by construction
# (the 5 subjects are leak-unsafe for DL inference; see
# configs/cohort_legacy5.yaml). The supplementary DL-vs-GT panel
# answers R2.8 at the per-subject level using cohort subjects where DL
# inference is legitimate test-set inference.
LEGACY5_COHORT_YAML  ?= configs/cohort_legacy5.yaml
LEGACY5_DERIV_ROOT   ?= data/derivatives_legacy5
LEGACY5_COHORT_ROOT  ?= data/ucsf_pdgm_legacy5
LEGACY5_HITPLOT_TOPK ?= 12
SUPP_DLVSGT_TOPK     ?= 8
SUPP_DLVSGT_OUT_PDF  ?= outputs/figures/supp_legacy5_dlvsgt_panel.pdf
SUPP_DLVSGT_OUT_PNG  ?= outputs/figures/supp_legacy5_dlvsgt_panel.png
SUPP_DLVSGT_OUT_JSON ?= outputs/figures/supp_legacy5_dlvsgt_panel.json

legacy5-hitplot-gt:
	$(PY) scripts/compute_legacy5_hitplot_gt.py \
	    --no-dry-run \
	    --cohort-yaml $(LEGACY5_COHORT_YAML) \
	    --cohort-root $(LEGACY5_COHORT_ROOT) \
	    --deriv-root $(LEGACY5_DERIV_ROOT)

legacy5-figs-anat-profile: legacy5-hitplot-gt
	@for sid in 0020 0022 0039 0066 0085; do \
	    $(PY) scripts/render_per_subject_hitplot_bar.py \
	        --csv $(LEGACY5_DERIV_ROOT)/sub-$$sid/hitplot/sub-$${sid}_hitplot_gt.csv \
	        --subject $$sid \
	        --source gt \
	        --top-k $(LEGACY5_HITPLOT_TOPK) \
	        --out-pdf outputs/figures/legacy5/sub-$${sid}_panel_i_hitplot_gt.pdf \
	        --out-png outputs/figures/legacy5/sub-$${sid}_panel_i_hitplot_gt.png ; \
	done

supp-legacy5-dlvsgt-panel:
	$(PY) scripts/build_legacy5_dlvsgt_panel.py \
	    --agreement-csv $(AGREEMENT_OUT_CSV) \
	    --hitplot-root $(AGREEMENT_HITPLOT_ROOT) \
	    --top-k $(SUPP_DLVSGT_TOPK) \
	    --out-pdf $(SUPP_DLVSGT_OUT_PDF) \
	    --out-png $(SUPP_DLVSGT_OUT_PNG) \
	    --out-json $(SUPP_DLVSGT_OUT_JSON)

# Umbrella: regenerate every artefact behind the per-subject anatomical
# profile figures (Figs. 4, 5, 12, 13, 14 + supplementary DL-vs-GT
# panel). Does NOT re-run FreeSurfer; that is gated by figure2-legacy5.
figs-anat-profile: figure2-render legacy5-hitplot-gt legacy5-figs-anat-profile supp-legacy5-dlvsgt-panel

# R3 Fig. 6 replacement: rebuild the longitudinal tumour-volume trajectory
# + RANO panel for LUMIERE Patient-048 from the per-tp CSV that
# scripts/segment_lumiere_p048.py writes. Pure plotting + summary; no
# inference, no network, < 2 s on CPU. Override LUMIERE_P048_SUMMARY to
# point the producer at a different segmentation run (e.g. a CUDA re-run
# on a GPU box, or a `--backend dummy` smoke output).
LUMIERE_P048_SUMMARY  ?= data/lumiere_p048/derivatives/segmentation_summary.csv
LUMIERE_P048_MANIFEST ?= configs/lumiere_p048_timepoints.yaml
LUMIERE_P048_FIG_DIR  ?= outputs/figures
LUMIERE_P048_BASENAME ?= fig6_lumiere_p048_volumes

figure6-lumiere-volumes:
	$(PY) scripts/build_figure6_lumiere_p048_volumes.py \
	    --summary-csv $(LUMIERE_P048_SUMMARY) \
	    --manifest $(LUMIERE_P048_MANIFEST) \
	    --out-dir $(LUMIERE_P048_FIG_DIR) \
	    --basename $(LUMIERE_P048_BASENAME)

# R3 Fig. 6 QC: per-tp segmentation table + DL-vs-{HD-GLIO, DeepBraTumIA}
# Dice/JSC for LUMIERE Patient-048. Pure I/O + numpy; no inference, no
# network. Writes a JSON sidecar (machine-readable) and a Markdown mirror
# (human-readable, suitable to paste into the revision tracker).
LUMIERE_P048_DERIVS   ?= data/lumiere_p048/derivatives
LUMIERE_P048_QC_JSON  ?= outputs/inventory/lumiere_p048_segmentation_summary.json
LUMIERE_P048_QC_MD    ?= outputs/inventory/lumiere_p048_segmentation_summary.md

summarize-lumiere-p048-segmentation:
	$(PY) scripts/summarize_lumiere_p048_segmentation.py \
	    --derivatives-root $(LUMIERE_P048_DERIVS) \
	    --summary-csv $(LUMIERE_P048_SUMMARY) \
	    --json-out $(LUMIERE_P048_QC_JSON) \
	    --md-out $(LUMIERE_P048_QC_MD)

# R3 Step G.2: real FreeSurfer 8.2.0 recon-all-clinical fan-out across all
# 6 LUMIERE Patient-048 timepoints (seeded on the registered T1 channel,
# tp -> separate FS subject `p048_<week>` so longitudinal scans are not
# conflated). Writes the per-tp parcellation contract on the registered
# RAS grid (wmparc_native.nii.gz + wmparc_lut.json + wmparc.json) plus a
# cohort-level parcellation_summary.{csv,json}. ~30-90 min/tp on Apple
# silicon CPU; total 3-12 h across the 6 tp.
#
# Override LUMIERE_PARCELLATE_BACKEND=dummy for the CI-runnable synthetic
# 9-region wmparc that does not require FreeSurfer on the host (used by
# the test suite and by the smoke pipeline). Override
# LUMIERE_PARCELLATE_FS_WORK_DIR to relocate the multi-GB FS scratch (it
# defaults to data/lumiere_p048/derivatives/freesurfer/, which is
# .gitignored); pass LUMIERE_PARCELLATE_SKIP_EXISTING=1 to resume a partial
# run after a kill / reboot.
#
# Examples:
#     make parcellate-lumiere-p048
#     make parcellate-lumiere-p048 LUMIERE_PARCELLATE_BACKEND=dummy
#     make parcellate-lumiere-p048 LUMIERE_PARCELLATE_FS_WORK_DIR=/scratch/fs_p048
#     make parcellate-lumiere-p048 LUMIERE_PARCELLATE_SKIP_EXISTING=1
LUMIERE_PARCELLATE_BACKEND       ?= freesurfer
LUMIERE_PARCELLATE_FS_WORK_DIR   ?= data/lumiere_p048/derivatives/freesurfer
LUMIERE_PARCELLATE_THREADS       ?= 4
LUMIERE_PARCELLATE_TIMEPOINTS    ?=
LUMIERE_PARCELLATE_SKIP_EXISTING ?=

parcellate-lumiere-p048:
	$(PY) scripts/parcellate_lumiere_p048.py \
	    --no-dry-run \
	    --manifest $(LUMIERE_P048_MANIFEST) \
	    --derivatives-root $(LUMIERE_P048_DERIVS) \
	    --backend $(LUMIERE_PARCELLATE_BACKEND) \
	    --threads $(LUMIERE_PARCELLATE_THREADS) \
	    $(if $(LUMIERE_PARCELLATE_FS_WORK_DIR),--fs-work-dir $(LUMIERE_PARCELLATE_FS_WORK_DIR),) \
	    $(foreach tp,$(LUMIERE_PARCELLATE_TIMEPOINTS),--timepoint $(tp)) \
	    $(if $(LUMIERE_PARCELLATE_SKIP_EXISTING),--skip-existing,)

# R3 Step G.3 (Hit-Plot computer for LUMIERE Patient-048): joins the
# FreeSurfer wmparc produced by Step G.2 with the four segmentation
# sources (DL hard, DL softmax probs, HD-GLIO-AUTO, DeepBraTumIA-native)
# and writes per-tp Hit-Plot CSVs + a cohort-level summary, ready to be
# folded into Figure 11. Requires the parcellation, segmentation, and
# (optional) comparator NIfTIs already on disk under
# data/lumiere_p048/derivatives/. Comparator label maps are silently
# remapped into the BraTS-3 scheme (HD-GLIO emits WT+ET only; DBT emits
# WT+TC+ET) so a single producer powers every source.
#
# Override HITPLOT_LUMIERE_TIMEPOINTS / HITPLOT_LUMIERE_SOURCES to compute
# only a subset, and HITPLOT_LUMIERE_SKIP_EXISTING=1 to re-aggregate the
# cohort summary without recomputing per-tp CSVs.
#
# Examples:
#     make hitplot-lumiere-p048
#     make hitplot-lumiere-p048 HITPLOT_LUMIERE_TIMEPOINTS="week-013 week-049"
#     make hitplot-lumiere-p048 HITPLOT_LUMIERE_SOURCES="dl dl_prob"
#     make hitplot-lumiere-p048 HITPLOT_LUMIERE_SKIP_EXISTING=1
HITPLOT_LUMIERE_TIMEPOINTS    ?=
HITPLOT_LUMIERE_SOURCES       ?=
HITPLOT_LUMIERE_SKIP_EXISTING ?=

hitplot-lumiere-p048:
	$(PY) scripts/compute_hitplot_lumiere_p048.py \
	    --manifest $(LUMIERE_P048_MANIFEST) \
	    --derivatives-root $(LUMIERE_P048_DERIVS) \
	    $(foreach tp,$(HITPLOT_LUMIERE_TIMEPOINTS),--timepoint $(tp)) \
	    $(foreach src,$(HITPLOT_LUMIERE_SOURCES),--source $(src)) \
	    $(if $(HITPLOT_LUMIERE_SKIP_EXISTING),--skip-existing,)

# R3 Step G.4a: builds Figure 11, the round-3 closing figure for the
# LUMIERE Patient-048 longitudinal arc -- a two-panel (WT, ET) heatmap of
# the top-K wmparc regions across all 6 timepoints, computed from the
# per-tp DL Hit-Plot CSVs (Step G.3). Override FIGURE11_TOP_K to widen /
# narrow the heatmap.
FIGURE11_TOP_K ?= 12

figure11-lumiere-p048-hitplot:
	$(PY) scripts/build_figure11_lumiere_p048_hitplot.py \
	    --manifest $(LUMIERE_P048_MANIFEST) \
	    --hitplot-root $(LUMIERE_P048_DERIVS) \
	    --top-k $(FIGURE11_TOP_K)

sync:
	$(PY) scripts/sync_figs_to_paper.py

paper:
	$(MAKE) -C paper paper

PANDOC      ?= pandoc
PDF_ENGINE  ?= xelatex
DOC_SOURCES := docs/design_hitplot_alternatives.md \
               docs/scope_pr7.md
DOC_PDFS    := $(DOC_SOURCES:.md=.pdf)

docs: $(DOC_PDFS)

docs/%.pdf: docs/%.md
	$(PANDOC) $< -o $@ \
	    --pdf-engine=$(PDF_ENGINE) \
	    --syntax-highlighting tango \
	    -V colorlinks=true

all: figures sync paper

clean:
	$(MAKE) -C paper veryclean
	rm -rf outputs/ .pytest_cache .mypy_cache .ruff_cache
