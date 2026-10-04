# PR-7 — Scope: cohort fan-out, Tables 2 & 3, Hit-Plot agreement panel

This document is the contract for the n=50 cohort fan-out: segmentation
metrics, the Hit-Plot agreement panel, and the table producers.

---

## 1. Goal of PR-7

Take the unified MONAI Bundle BraTS-3 segmenter validated on n=3 in
PR-8 (commit `c65ec02`, mean WT-Dice 0.895 / TC 0.893 / ET 0.777) and
produce, on the locked n=50 UCSF-PDGM cohort
(`configs/cohort_ucsfpdgm_n50.yaml`):

1. Per-subject DL segmentation outputs in a stable on-disk schema
   (Section 4).
2. **Table 2** — DL vs dataset-provided reference: per-subject and
   summary Dice / HD95 / |VE| / sensitivity / specificity by BraTS
   compartment (WT, TC, ET).
3. **Table 3** — DL vs `Raidionics` vs `segment_glioma` paired
   comparison on the same 50 subjects, same metrics.
4. **Hit-Plot–GT vs Hit-Plot–DL agreement panel** — per-cell agreement
   between the Hit-Plot computed from the dataset-provided reference
   masks (`Hit-Plot-GT`) and from the unified DL output
   (`Hit-Plot-DL`), reported as a quantitative answer to "sensitivity
   to segmentation variability" (R2.8).
5. **Probabilistic Hit-Plot** — same matrix as (4) but with mean ± std
   propagated from the per-class softmax (the second member of the
   Hit-Plot family that this revision commits to per
   `docs/design_hitplot_alternatives.md`).

Everything else from `docs/design_hitplot_alternatives.md`
(cortical-surface, tract / OAR proximity, Hit-Plot-to-outcome models)
is explicitly **out of scope** for PR-7 and remains future work.

---

## 2. Inputs (already in the repo)

| Input | Path | Status |
|---|---|---|
| Cohort YAML (n=50, seed=20260415) | `configs/cohort_ucsfpdgm_n50.yaml` | locked |
| Per-subject NIfTI volumes (T1/T1c/T2/FLAIR, tumor mask, brain mask) | `data/ucsf_pdgm_cohort50/sub-XXXX/` | present |
| UCSF-PDGM v5 metadata CSV | `data/UCSF-PDGM-metadata_v5.csv` | present |
| MONAI bundle (cached) | `~/.cache/hpgs/monai_bundles/brats_mri_segmentation/` | present (v0.4.8) |
| Unified segmenter | `src/hpgs/segment/unified.py` | done (PR-8) |
| Cohort loader | `src/hpgs/io/cohort.py::load_cohort_metadata` | done |
| Subject input resolver | `src/hpgs/io/__init__.py::resolve_subject_inputs` | done |
| Basic Hit-Plot | `src/hpgs/hitplot/__init__.py::parcel_tumor_burden` | done (volumetric only) |
| Basic metrics (Dice, sens, spec, |VE|) | `src/hpgs/metrics/__init__.py` | done |
| Hausdorff-95 | `src/hpgs/metrics/__init__.py::hausdorff95` | **TODO (NotImplementedError)** |
| FreeSurfer 8.2.0 SynthSeg / `recon-all-clinical` wrappers | `src/hpgs/parcellate/__init__.py` | done |
| Per-subject SynthSeg / `wmparc` parcellations on the 50 cohort subjects | `data/derivatives_cohort50/sub-XXXX/parcellation/` | **cohort runner DONE (PR-7c, `make parcellate-all`); real FreeSurfer runs still pending on the host (the long pole of § 7)** |

The last row is the most expensive prerequisite and is discussed in
Section 7.

---

## 3. Two open design questions — RESOLVED

These were carried over from `docs/design_segmentation_role.md`
§ "Open questions" and from the original PR-7 scoping discussion. Both
are now **LOCKED** to the default recommendations (commit 2026-04-17).
Any future change requires a single commit on this file with a one-line
rationale and a coordinated update to `paper/main.tex` and
the affected table / panel producer scripts in Section 5.

### Q1 — What goes into Table 3 as the reference set?

**LOCKED: Option B.** Table 3 is framed as a *segmenter agreement*
table with **four columns**:

1. **DL (MONAI Bundle BraTS3 SegResNet, v0.4.8)** — the unified
   segmenter committed to in `docs/design_segmentation_role.md`.
2. **Raidionics** — open-source baseline, atlas-space pipeline.
3. **`segment_glioma`** — pretrained PICTURE-project baseline (the
   model used for the BGO data).
4. **BraTS21-manual (UCSF-PDGM)** — the dataset-provided
   manually-corrected reference mask, treated here as one more
   segmenter (a human-curated one) rather than as ground truth. This
   reuses the framing already in the Discussion `\rev{...}` block,
   which deliberately calls these masks a "reference" rather than
   ground truth.

Considered and rejected:
- *Option A (3 columns; reference reported separately in Table 2
  only):* loses the cleanest cross-segmenter comparison the reviewers
  asked for in R2.5; would also force a separate panel to compare
  human-curated against DL on the same axes as Raidionics /
  `segment_glioma`.

Rationale recap: (i) reviewer R2.5 asked specifically for a rigorous
cross-segmenter comparison; (ii) keeping the reference mask as a
fourth column makes "DL vs. human-curated reference" and "DL vs.
automated baselines" comparable on identical metrics in a single
table; (iii) per-cell pairings remain interpretable because BraTS21
manual was the *actual* reference for Table 2, so the diagonal of the
DL-vs-BraTS21 column reproduces Table 2 exactly --- a built-in
consistency check.

Implementation consequences (already reflected in Sections 4 and 5):
- The artefact contract keeps `metrics/sub-XXXX_metrics_dl_vs_*.json`
  for each of `gt`, `raidionics`, `segmentglioma` (the BraTS21-manual
  is the `gt` row).
- `scripts/build_table3_segmenter_agreement.py` reads all three
  metrics JSONs and emits a four-column table (DL is the fourth column,
  with each of the other three columns reporting the metric pair
  `(DL, other)`).

### Q2 — Agreement metric of record for Tables 2 / 3 and the Hit-Plot panel

**LOCKED: Option C.** A small fixed bundle of metrics is reported in
each artefact, with one role per artefact:

| Artefact | Per-compartment headline metrics | Visual / paired summary |
|---|---|---|
| Table 2 (DL vs. UCSF-PDGM reference) | Dice, HD95, |VE|, sensitivity, specificity (median + IQR over the 50 subjects) | --- |
| Table 3 (segmenter agreement) | Dice, HD95, |VE| pairwise (DL vs. each of the other three columns) | CCC (concordance correlation coefficient) per compartment in a footnote row |
| Hit-Plot agreement panel (Hit-Plot-DL vs. Hit-Plot-GT) | --- | Bland–Altman per cell (visual); CCC per compartment as the headline summary number; cohort-mean per-cell |delta-volume| as a one-number summary |
| Probabilistic Hit-Plot summary | mean ± std per cell from the per-class softmax | --- |

Considered and rejected:
- *Option A alone (Bland–Altman + Pearson r):* good for the panel,
  but Pearson r alone over-credits proportional bias; CCC is the
  reviewer-defensible single-number summary.
- *Option B alone (CCC):* loses the visual, per-cell story the
  Hit-Plot panel needs to answer R2.8 ("sensitivity to segmentation
  variability") at a glance.

Rationale recap: (i) Dice / HD95 / |VE| are the per-voxel headline
numbers reviewers expect in segmentation papers; (ii) CCC is the
single number the reviewers asked for to summarise agreement
quantitatively across cells / compartments; (iii) Bland–Altman is the
*visual* device that makes the agreement panel readable; carrying
both is cheap and the producers all consume the same per-subject
JSONs.

Implementation consequences (already reflected in Sections 4 and 5):
- `src/hpgs/metrics/__init__.py` must finish `hausdorff95` *and* gain
  a `concordance_correlation_coefficient` helper (small addition,
  ~30 LOC, covered by a new `tests/test_metrics_ccc.py`).
- `scripts/build_hitplot_agreement_panel.py` produces a Bland–Altman
  per cell and reports CCC per compartment in the figure caption (the
  caption is regenerated as a `\rev{...}` block in `main.tex`).

---

## 4. Per-subject artefact contract

PR-7 writes **one directory per subject** under
`data/derivatives_cohort50/sub-XXXX/`. The directory layout is fixed
and documented here so that downstream Table-2 / Table-3 / Hit-Plot
panel code can rely on it. The dry-run stub
(`scripts/segment_all_cohort50.py --dry-run`) prints these paths
verbatim for every subject without running any inference.

```
data/derivatives_cohort50/sub-XXXX/
  seg_dl/                              # Section 5.1 — unified MONAI Bundle
    sub-XXXX_seg_brats3_dl.nii.gz       # uint8 label map; NCR=1, ED=2, ET=4
    sub-XXXX_seg_brats3_dl_probs.nii.gz # float32 4D (X,Y,Z,3); channels in PROB_CHANNEL_ORDER ("TC", "WT", "ET" sigmoid) -- the exact order is also recorded as ``probability_channel_order`` in the seg_dl JSON sidecar so downstream readers never have to guess
    sub-XXXX_seg_brats3_dl.json         # backend, bundle SHA, channel order, runtime, dice-vs-ref

  parcellation/                        # Section 5.2 — SynthSeg / recon-all-clinical
    sub-XXXX_wmparc_native.nii.gz       # int16 label map in native subject space
    sub-XXXX_wmparc_lut.json            # {label_id (str): anatomical name} for ~100 regions

  hitplot/                             # Section 5.3 — Hit-Plot family v1+prob
    sub-XXXX_hitplot_dl.csv             # (compartment x region) volumes from DL mask
    sub-XXXX_hitplot_dl_prob.csv        # mean and std of (compartment x region) from softmax
    sub-XXXX_hitplot_gt.csv             # same matrix from dataset-provided reference mask
    sub-XXXX_hitplot_raidionics.csv     # same matrix from Raidionics mask          (Q1=B only)
    sub-XXXX_hitplot_segmentglioma.csv  # same matrix from segment_glioma mask      (Q1=B only)

  metrics/                             # Section 5.4 — flat per-subject JSONs feeding Tables 2/3
    sub-XXXX_metrics_dl_vs_gt.json      # dice / hd95 / abs_ve / sens / spec per WT/TC/ET
    sub-XXXX_metrics_dl_vs_raidionics.json
    sub-XXXX_metrics_dl_vs_segmentglioma.json

  seg_raidionics/                      # PR-7e extension — Raidionics baseline (Q1=B)
    sub-XXXX_seg_raidionics.nii.gz      # uint8 BraTS labels {0,1,2,4}
    sub-XXXX_seg_raidionics.json        # backend, version, runtime, paths

  seg_segmentglioma/                   # PR-7e extension — segment_glioma baseline (Q1=B)
    sub-XXXX_seg_segmentglioma.nii.gz
    sub-XXXX_seg_segmentglioma.json
```

**PR-7e schema convention.** The two paired-metrics files
`metrics/sub-XXXX_metrics_dl_vs_<baseline>.json` use the same Q2-C
metric bundle as `metrics_dl_vs_gt.json` (Dice / HD95 / |VE| /
sensitivity / specificity per WT/TC/ET) so a single Table 3 reader can
walk both `dl_vs_gt` and `dl_vs_<baseline>` files. Convention:
**prediction = DL, reference = baseline**. Dice / HD95 / |VE| are
symmetric so the convention only matters for sensitivity / specificity
(de-emphasised in Q2 = C, but recorded for schema parity with Table 2).

Cohort-level outputs (single files, not per-subject):

```
outputs/tables/
  table2_dl_vs_gt.tex                  # Table 2 producer reads metrics/*.json
  table3_segmenter_agreement.tex       # Table 3 producer reads the three metrics_dl_vs_*.json
outputs/figures/
  fig_hitplot_agreement_dl_vs_gt.pdf   # Hit-Plot agreement panel
```

**Schema versioning.** A `schema_version: "1.0"` field will be written
to every JSON sidecar; bumping the schema will require a coordinated
update to the readers in `src/hpgs/metrics/` and the table producers.

---

## 5. New code to write in PR-7

| Module / script | est. LOC | complexity | reviewer points |
|---|---:|---|---|
| `scripts/segment_all_cohort50.py` (replace dry-run stub with the real fan-out runner) --- **DONE (PR-7d)**: `--no-dry-run` calls `hpgs.segment.unified.predict_brats3` (MONAI Bundle BraTS3, channel order `(T1c,T1,T2,FLAIR)` locked in PR-8) on every cohort subject, writes the Section-4 contract under `data/derivatives_cohort50/sub-XXXX/seg_dl/` (label map + per-class sigmoid probability map + JSON sidecar) and `metrics/sub-XXXX_metrics_dl_vs_gt.json` with the Q2-C metric bundle (per-compartment Dice / HD95 / |VE| / sensitivity / specificity), reusing the PR-7a `hausdorff95` and `hpgs.segment.derive_subregions`. Per-subject failures are logged and the run continues; supports `--subject ID` (repeatable), `--skip-existing` (resumeable), and `--backend dummy` (the network-free contract exerciser used by the unit tests below). 22 isolated tests in `tests/test_segment_all_cohort50_runner.py`. | 200--300 | low | R1.M2, R2.2 |
| `scripts/parcellate_all_cohort50.py` (FreeSurfer SynthSeg / `recon-all-clinical`) --- **DONE (PR-7c)**: wraps the existing `src/hpgs/parcellate` FreeSurfer shell-outs (`recon_all_clinical` + `resample_like` + `synthseg_label_lut`) behind a new unified `hpgs.parcellate.predict_wmparc` adapter with two backends: `"freesurfer"` (real; runs `recon-all-clinical.sh` under a scratch `SUBJECTS_DIR`, resamples `wmparc.mgz` onto the native mpMRI grid via nearest-neighbour `mri_convert`, parses `FreeSurferColorLUT.txt` and restricts to labels actually present) and `"dummy"` (CI-runnable; synthesises a deterministic 9-region label map + `FS_DUMMY_LUT` LUT subset on the native grid so the PR-7f Hit-Plot cohort runner can be exercised end-to-end without FreeSurfer on the host). Cohort runner mirrors the PR-7d/PR-7e CLI contract (`--dry-run` + `--no-dry-run`, `--subject ID`, `--backend {freesurfer,dummy}`, `--input-channel {T1_bias,T1c_bias,T2_bias,FLAIR_bias}`, `--fs-work-dir`, `--threads`, `--skip-existing`) and writes the Section-4 parcellation contract per subject: `parcellation/sub-XXXX_wmparc_native.nii.gz` (int16 label map on the native mpMRI grid, matching every other PR-7 artefact's shape exactly), `parcellation/sub-XXXX_wmparc_lut.json` (`{label_id (str): anatomical_name (str)}` restricted to labels present, the shape `scripts/compute_hitplot_cohort50.py::_read_wmparc_lut` accepts byte-for-byte), plus a `parcellation/sub-XXXX_wmparc.json` provenance sidecar (`schema_version=1.0`, `backend`, FS version string, `input_channel`, `freesurfer_subjects_dir`, `threads`, `elapsed_s`, `n_labels_present`, output paths). Per-subject failures are logged and the run continues; exit code 0 / 1 / 2 on clean / per-subject-error / cohort-load-or-config-error (the latter catches `--backend freesurfer` without `--fs-work-dir`). New `make parcellate-all` wrapper with env knobs (`PARCELLATE_BACKEND`, `PARCELLATE_INPUT_CHANNEL`, `PARCELLATE_FS_WORK_DIR`, `PARCELLATE_THREADS`, `PARCELLATE_SUBJECTS`, `PARCELLATE_SKIP_EXISTING`). 39 isolated tests in `tests/test_parcellate_all_cohort50_runner.py` covering the `predict_wmparc` dummy-backend contract (int16 dtype, native-grid shape, affine propagation, subject-dependent determinism, LUT subset-of-`FS_DUMMY_LUT`, labels-present-in-label-map round-trip, unknown-backend rejection, `fs_work_dir`-required rejection), the Section-4 artefact contract (`_planned_artefacts` keys + layout + subject-id normalisation), `_check_inputs` (channel present / channel missing / tumor_segmentation missing MISS coherence with PR-7d/PR-7e), `_select_subjects`, `_write_lut_json` (string-int keys, ascending numeric ordering, parent-dir creation), the full `_real_process_subject` end-to-end (artefact existence, NIfTI int16 dtype + native shape, LUT JSON schema, sidecar schema, honouring `--input-channel`, subject-root directory creation, `--skip-existing` short-circuit, missing-input `FileNotFoundError`, LUT values match `FS_DUMMY_LUT` names), and `main()` orchestration (dry-run 0/1, real-run 0/1 under dummy, 2 on cohort-load failure, 2 on `freesurfer`-without-`fs-work-dir`, subject-filter honoured, `--skip-existing` idempotent across back-to-back `main()` calls, PR-7f LUT-reader round-trip integration). | 150--200 | low (wraps existing `src/hpgs/parcellate`) | R2.8 |
| `src/hpgs/metrics/__init__.py` — finish `hausdorff95` + add `concordance_correlation_coefficient` (per Q2-locked decision) + unit tests --- **DONE (PR-7a)**: `tests/test_metrics_hd95.py` (9 tests) and `tests/test_metrics_ccc.py` (11 tests) | 90 | medium | R2.2, R2.5, R2.8 |
| `src/hpgs/hitplot/__init__.py` — extend with `compartment_region_matrix(...)` (matrix instead of just top-k bar plot) and `probabilistic_hitplot(...)` (mean ± std from softmax) --- **DONE (PR-7b)**: `tests/test_hitplot_matrix.py` (11 tests) and `tests/test_hitplot_probabilistic.py` (13 tests); deterministic-mean equals probabilistic-mean for hard `{0,1}` probabilities (cross-check the agreement panel will rely on) | 100--150 | medium | R2.8 |
| `scripts/compute_hitplot_cohort50.py` --- **DONE (PR-7f)**: cohort fan-out runner that drives `hpgs.hitplot.compartment_region_matrix` (deterministic) and `hpgs.hitplot.probabilistic_hitplot` (closed-form Poisson-binomial mean/std from the PR-7d softmax probs NIfTI) across every subject in the locked cohort YAML, writing the five Section-4 CSVs per subject (`hitplot/sub-XXXX_hitplot_{dl,dl_prob,gt,raidionics,segmentglioma}.csv`). Mirrors the PR-7d/PR-7e CLI contract (`--dry-run` + `--no-dry-run`, `--subject ID`, `--source {dl,dl_prob,gt,raidionics,segmentglioma,all}`, `--skip-existing`, `--require-parcellation` / `--no-require-parcellation`) and runs every `(subject, source)` pair independently so one missing baseline NIfTI can't abort the whole cohort. Reads `probability_channel_order` from the PR-7d seg_dl JSON sidecar so the probabilistic Hit-Plot is fed the right (TC, WT, ET) channel order. Ignores subjects whose parcellation artefacts (`parcellation/sub-XXXX_wmparc_native.nii.gz` + `..._wmparc_lut.json`) are absent (warning under `--require-parcellation`, silent under `--no-require-parcellation`). Exit code 0 on clean run, 1 when any `(subject, source)` pair errored, 2 when the cohort YAML / derivatives root is unreachable. New `make hitplot-all` wrapper with env knobs (`HITPLOT_COHORT_ROOT`, `HITPLOT_DERIV_ROOT`, `HITPLOT_SOURCES`, `HITPLOT_SUBJECTS`, `HITPLOT_SKIP_EXISTING`, `HITPLOT_REQUIRE_PARCELLATION`). 50 isolated tests in `tests/test_compute_hitplot_cohort50_runner.py` --- covers the Section-4 artefact contract, source/subject filter expansion (including `all` alias and whitespace-packed tokens), the wmparc LUT loader (string-keyed JSON + empty + non-integer + non-dict rejection), the `probability_channel_order` sidecar reader with three fallback paths, synthetic-cohort end-to-end for each of the 5 sources, the cross-check that hard-{0,1} probs reproduce the deterministic counts bit-for-bit (the invariant the PR-7i agreement panel will rely on), missing-input + shape-mismatch + malformed-JSON failure paths, `--skip-existing` short-circuit with sentinel-preservation, CLI arg-parsing defaults, and `main()` exit codes (0 on clean, 1 on per-pair error, 2 on cohort load failure). | 150 | low | R2.8 |
| `scripts/run_baseline_segmenters_cohort50.py` (Raidionics + `segment_glioma` on n=50; reuses existing legacy-5 baseline code where it exists) --- **DONE (PR-7e, scaffold + dummy backend; PR-7e2 fills in the docker glue)**: new `hpgs.segment.baselines.predict_baseline` adapter (CI-runnable `dummy` backend + `docker` stub raising `NotImplementedError("PR-7e2")` with pinned-digest provenance plumbing in place); cohort runner mirrors the PR-7d structure (subject × baseline loop, `--dry-run` + `--no-dry-run`, `--subject ID` filter, `--baseline {raidionics,segmentglioma,all}` filter, `--skip-existing`, `--require-dl` / `--no-require-dl`); writes the Section-4 extension `seg_<baseline>/sub-XXXX_seg_<baseline>.{nii.gz,.json}` + `metrics/sub-XXXX_metrics_dl_vs_<baseline>.json` (uint8 BraTS labels, JSON-safe NaN/Inf, `schema_version=1.0`, `comparison=dl_vs_<baseline>`, voxel spacing/volume mirrored from PR-7d). Reads the existing PR-7d `seg_dl/sub-XXXX_seg_brats3_dl.nii.gz` and computes the Q2-C paired metrics (Dice / HD95 / |VE| / sensitivity / specificity per WT/TC/ET) under the convention `pred=DL, ref=baseline`. Per-`(subject, baseline)` failures are logged and the run continues; final exit code is non-zero if any pair failed. New `make baselines-all` wrapper (overrides: `BASELINE_BACKEND`, `BASELINE_BASELINES`, `BASELINE_SUBJECTS`, `BASELINE_SKIP_EXISTING`, `BASELINE_REQUIRE_DL`). 34 isolated tests in `tests/test_baseline_segmenters_runner.py` (public surface of `hpgs.segment.baselines`, `_planned_baseline_artefacts` Section-4-extension contract, voxel/JSON helpers mirrored from PR-7d, paired-metrics on identical / empty-reference masks, `_select_subjects` / `_select_baselines` including `all` expansion + dedup + unknown rejection, full `_real_process_subject_baseline` end-to-end with the dummy backend covering sidecar schema, paired-metrics JSON schema, BraTS uint8 NIfTI, `--require-dl` strict failure, `--no-require-dl` baseline-only mode, `--skip-existing` short-circuit, per-baseline distinctness regression guard). | 200--300 | medium | R2.5 |
| `scripts/build_table2_dl_vs_gt.py` (reads `metrics/*.json`, emits `outputs/tables/table2_dl_vs_gt.tex`) --- **DONE (PR-7g)**: validates each sidecar against the PR-7d schema (`schema_version=1.0`, `comparison=dl_vs_gt`), aggregates per-(compartment, metric) median [Q1, Q3] with NaN/Inf-safe denominators (`null` -> `n_missing`, `"inf"` -> `n_inf`, both excluded from the median), and writes three artefacts: `outputs/tables/table2_dl_vs_gt.tex` (self-contained `\begin{tabular}` block ready to `\input{}` from `paper/main.tex`; optional `Per-cell denominator` footnote when any cell drops samples), `outputs/tables/table2_dl_vs_gt.csv` (long-format per-subject audit trail), `outputs/tables/table2_dl_vs_gt.json` (per-cell summary + list of `(subject_id, reason)` problems). New `make table2-dl-vs-gt` wrapper. 33 isolated tests in `tests/test_build_table2_dl_vs_gt.py`. Sanity-checked end-to-end on the locked cohort YAML before any PR-7d sidecars exist on disk: producer correctly warns on every subject, writes a placeholder TeX table with `--` cells, and exits 1. | 120 | low | R2.2, R2.10 |
| `scripts/build_table3_segmenter_agreement.py` --- **DONE (PR-7h)**: reads the three PR-7d/PR-7e paired-metrics sidecar families per subject (`metrics/sub-XXXX_metrics_dl_vs_{raidionics,segmentglioma,gt}.json`), validates each against the schema (`schema_version=1.0`, expected `comparison` per pair, all WT/TC/ET × {dice, hd95_mm, abs_volumetric_error_mm3, sensitivity, specificity} present, forward-compat about extra keys), aggregates per-(pair, compartment, metric) median [Q1, Q3] with the same NaN/Inf-safe denominator rule as PR-7g (`null` -> `n_missing`, `"inf"` -> `n_inf`, both excluded from the median), and writes three artefacts: `outputs/tables/table3_segmenter_agreement.tex` (self-contained four-column `\begin{tabular}{l c c c}` block grouped by metric --- one `\multicolumn{4}{l}{\textit{...}}` header per metric followed by three `\quad WT|TC|ET` rows, columns are `Raidionics | segment\_glioma | BraTS21-manual` with the BraTS21-manual column reproducing Table 2 by construction; optional `Per-cell denominator` footnote when any (pair, cell) drops samples), `outputs/tables/table3_segmenter_agreement.csv` (long-format per-subject audit trail with `subject_id, pair_id, compartment, metric, value, status, raw`), `outputs/tables/table3_segmenter_agreement.json` (per-(pair, compartment, metric) summary + `n_with_sidecar_per_pair` + `n_problems_per_pair` + `(subject_id, reason)` per pair; reserves a `ccc_per_cell: null` slot for the PR-7i Hit-Plot agreement panel). New `make table3-segmenter-agreement` wrapper (overrides: `TABLE3_DERIV_ROOT`, `TABLE3_OUT_TEX`, `TABLE3_OUT_CSV`, `TABLE3_OUT_JSON`, `TABLE3_STRICT`). 46 isolated tests in `tests/test_build_table3_segmenter_agreement.py` (including a cross-table consistency test that asserts the PR-7h `dl_vs_gt` block aggregates byte-equal to the PR-7g `summarise_per_cell` on the same fixture, and a regression guard that a swapped-comparison sidecar served from the wrong on-disk path is rejected with a descriptive `comparison` mismatch). | 180 | medium | R2.5, R2.10 |
| `scripts/build_hitplot_agreement_panel.py` --- **DONE (PR-7i)**: consumes the per-subject deterministic Hit-Plot CSVs PR-7f writes (`hitplot/sub-XXXX_hitplot_{prediction,reference}.csv`; default pair `dl` / `gt`), validates each against the PR-7f schema (rejects `hitplot_dl_prob.csv` by hard-failing on the missing `overlap_voxels` column, since the panel needs hard counts to be Bland–Altman-comparable), inner-joins on `(label, compartment)` per subject, drops NaN-in-either-pct rows defensively, and aggregates per-compartment {Lin's CCC, mean / std of (pred - ref) on `pct_of_parcel`, 1.96-SD limits-of-agreement, cohort-mean per-cell `|dl_overlap_volume_mm3 - gt_overlap_volume_mm3|`, n_pairs / n_subjects / n_regions} into three artefacts: `outputs/figures/fig_hitplot_agreement_dl_vs_gt.pdf` (1 × 3 Bland–Altman panel grid; CCC + n_pairs in each panel title; mean-diff + LoA reference lines + inline LoA annotation; "no data" placeholder if a compartment is empty), `outputs/figures/fig_hitplot_agreement_dl_vs_gt.json` (per-compartment summary the figure renders + `(subject_id, reason)` problems list, `schema_version=1.0`, `comparison={prediction, reference}`), `outputs/figures/fig_hitplot_agreement_dl_vs_gt.csv` (long-format per-(subject, label, compartment) audit trail with `pct_of_parcel_{pred,ref}`, `overlap_volume_mm3_{pred,ref}`, `diff_pct_of_parcel`, `mean_pct_of_parcel`, `abs_diff_volume_mm3` --- reviewers can recompute every panel and JSON cell from this CSV alone). Default comparison is `dl` vs `gt` (the headline figure for R2.8); `--prediction` / `--reference` accept any pair from the accepted-source set `{dl, gt, raidionics, segmentglioma}` so the same producer powers any pairwise sensitivity panel without code changes. New `make agreement-panel` wrapper with env knobs (`AGREEMENT_HITPLOT_ROOT`, `AGREEMENT_PREDICTION`, `AGREEMENT_REFERENCE`, `AGREEMENT_OUT_PDF`, `AGREEMENT_OUT_JSON`, `AGREEMENT_OUT_CSV`, `AGREEMENT_STRICT`). Exit codes 0 (clean), 1 (panel emitted but at least one compartment has n_pairs == 0), 2 (cohort YAML / hitplot root unreachable / pred==ref), 3 (`--strict` and at least one CSV missing / malformed). 43 isolated tests in `tests/test_build_hitplot_agreement_panel.py`. | 180 | medium | R2.8 |
| Tests: `tests/test_parcellate_all_cohort50_runner.py` (PR-7c; 39 tests on the dummy backend — see the PR-7c row above for the full coverage matrix), `tests/test_metrics_hd95.py`, `tests/test_metrics_ccc.py`, `tests/test_hitplot_matrix.py`, `tests/test_hitplot_probabilistic.py`, `tests/test_segment_all_cohort50_runner.py` (the dryrun-named placeholder was promoted to the broader `_runner` suite by PR-7d, covering both `--dry-run` enumeration and the full `_real_process_subject` end-to-end with the dummy backend), `tests/test_build_table2_dl_vs_gt.py` (PR-7g; 33 tests covering NaN/Inf-safe value classification, sidecar schema validation including forward-compat tolerance, sidecar loader happy-path / strict-mode / missing-file / malformed-JSON, hand-checked median/Q1/Q3 against a 5-sample fixture, exclusion of `null` and `"inf"` from the median, all-missing -> JSON-safe `None`, long-audit-frame schema, LaTeX table structure / median [IQR] formatting / `--` placeholders / footnote emission, CLI orchestration return codes 0/1/2/3), `tests/test_build_table3_segmenter_agreement.py` (PR-7h; 46 tests mirroring the PR-7g coverage but parameterised by the three pair families, plus a per-pair swapped-comparison rejection guard, a per-pair distinct-cell ordering check on the LaTeX output, an empty-pair fallback test on `summarise_all_pairs`, and the cross-table consistency assertion that the PR-7h `dl_vs_gt` summary block equals the PR-7g `summarise_per_cell` output on the same fixture), `tests/test_compute_hitplot_cohort50_runner.py` (PR-7f; 50 tests covering the Section-4 artefact contract, subject/source CLI expansion including the `all` alias and whitespace-packed tokens, the wmparc LUT loader (string-keyed JSON + empty + non-integer + non-dict rejection), the `probability_channel_order` sidecar reader with three fallback paths, synthetic-cohort end-to-end for each of the 5 sources, the cross-check that hard-{0,1} probs reproduce the deterministic counts bit-for-bit, missing-input + shape-mismatch + malformed-JSON failure paths, `--skip-existing` short-circuit with sentinel-preservation, CLI arg-parsing defaults, and `main()` exit codes 0 / 1 / 2), `tests/test_build_hitplot_agreement_panel.py` (PR-7i; 43 tests: per-compartment Bland–Altman + CCC summary on hand-checked fixtures, identity / mirror-image / orthogonal pair invariants (CCC = 1 / -1 / penalty), constant-bias-penalty regression, hand-checked Bland–Altman mean / std (ddof=1) / 1.96-SD LoA against numpy closed form, cross-check that the producer's CCC matches `hpgs.metrics.concordance_correlation_coefficient` on the same fixture, mean-`|dV|` against closed form, NaN/Inf-safe JSON serialisation, singleton-pair-keeps-mean-drops-std, `n_pairs / n_subjects / n_regions` distinct counters, CSV loader (happy-path / strict-mode / missing-file / malformed / probabilistic-CSV-rejection / inner-join-on-shared-labels-only / multi-subject concatenation / NaN-row drop), PDF artefact (non-empty file / parent-dir creation / empty-compartment placeholder), and CLI orchestration with `main()` exit codes 0 / 1 / 2 / 3 + alternative `--prediction`/`--reference` (e.g. `dl` vs `raidionics`)) | 350 | low | --- |
| `Makefile` --- **`parcellate-all` (DONE in PR-7c; calls `scripts/parcellate_all_cohort50.py --no-dry-run`; honours `PARCELLATE_BACKEND`, `PARCELLATE_INPUT_CHANNEL`, `PARCELLATE_FS_WORK_DIR`, `PARCELLATE_THREADS`, `PARCELLATE_SUBJECTS`, `PARCELLATE_SKIP_EXISTING`)**, **`segment-all` (DONE in PR-7d; calls `scripts/segment_all_cohort50.py --no-dry-run`; honours `SEGMENT_DEVICE`, `SEGMENT_SUBJECTS`, `SEGMENT_SKIP_EXISTING`)**, **`hitplot-all` (DONE in PR-7f; calls `scripts/compute_hitplot_cohort50.py --no-dry-run`; honours `HITPLOT_COHORT_ROOT`, `HITPLOT_DERIV_ROOT`, `HITPLOT_SOURCES`, `HITPLOT_SUBJECTS`, `HITPLOT_SKIP_EXISTING`, `HITPLOT_REQUIRE_PARCELLATION`)**, **`table2-dl-vs-gt` (DONE in PR-7g; calls `scripts/build_table2_dl_vs_gt.py`; honours `TABLE2_DERIV_ROOT`, `TABLE2_OUT_TEX`, `TABLE2_OUT_CSV`, `TABLE2_OUT_JSON`, `TABLE2_STRICT`)**, **`table3-segmenter-agreement` (DONE in PR-7h; calls `scripts/build_table3_segmenter_agreement.py`; honours `TABLE3_DERIV_ROOT`, `TABLE3_OUT_TEX`, `TABLE3_OUT_CSV`, `TABLE3_OUT_JSON`, `TABLE3_STRICT`)**, **`agreement-panel` (DONE in PR-7i; calls `scripts/build_hitplot_agreement_panel.py`; honours `AGREEMENT_HITPLOT_ROOT`, `AGREEMENT_PREDICTION`, `AGREEMENT_REFERENCE`, `AGREEMENT_OUT_PDF`, `AGREEMENT_OUT_JSON`, `AGREEMENT_OUT_CSV`, `AGREEMENT_STRICT`)** | --- | trivial | --- |
| `paper/main.tex` — Tables 2 & 3 captions in `\rev{...}`; Hit-Plot agreement panel figure in `\rev{...}` | --- | trivial | R2.2, R2.5, R2.8 |

**Engineering effort estimate (focused work):** 5--8 working days.

---

## 6. Acceptance criteria

PR-7 is mergeable when *all* of the following hold:

1. `make segment-all` succeeds on the locked cohort YAML and writes
   the per-subject artefacts in Section 4 for all 50 subjects (or
   exits with a clear per-subject error and a summary report; partial
   failure is acceptable provided the failed subject is logged).
2. `pytest -q -m "not slow and not gpu"` is green, including the new
   tests in Section 5 (the fast ones; full-cohort numerical
   regression tests are marked `slow` and not run in CI).
3. `make tables` produces `outputs/tables/table2_dl_vs_gt.tex` and
   `outputs/tables/table3_segmenter_agreement.tex`; both are valid
   LaTeX (compile cleanly inside `paper/main.tex`).
4. `make agreement-panel` produces
   `outputs/figures/fig_hitplot_agreement_dl_vs_gt.pdf`.
5. `make sync && make paper` rebuilds `paper/main.pdf` with
   the new tables and figure visibly inserted in `\rev{...}` blocks.
   (commit hash + summary stats).

A "headline numbers OK" sanity check on Table 2 is: cohort mean
WT-Dice $\geq 0.85$, TC-Dice $\geq 0.80$, ET-Dice $\geq 0.65$. These
are the bundle's own published BraTS-2018 validation operating point;
substantially lower numbers across the cohort would be a red flag of
the same kind we caught and fixed in PR-8 (channel-order bug).

---

## 7. Wall-clock and dependency budget (your hardware, not the agent)

| Step | Per subject | n=50 total | Notes |
|---|---:|---:|---|
| MONAI Bundle inference (CPU) | 5--10 min | 4--8 h | overnight on a Mac CPU; `make smoke-segment SMOKE_DEVICE=cpu` is the n=3 calibration |
| MONAI Bundle inference (MPS) | ~2 min | ~100 min | Apple-silicon path; PR-8 confirmed it loads |
| MONAI Bundle inference (CUDA) | ~5 s | ~5 min | only if a CUDA box is available |
| FreeSurfer 8.2.0 `recon-all-clinical` | 30 min--2 h | many h | trivially parallel; **biggest unknown — needs to be run before PR-7 can compute Hit-Plots** |
| `Raidionics` baseline | ~3 min | ~3 h | docker container, CPU |
| `segment_glioma` baseline | ~3 min | ~3 h | already wired for legacy-5 |

The right strategy is: while PR-7 code is being written, kick off
`make parcellate-all` (which PR-7 will define; today the dry-run
script just enumerates the subjects and prints which parcellations
would be needed) on your hardware in the background. Parcellation is
the long pole.

---

## 8. Risks (known unknowns)

- **R-1 — FreeSurfer parcellation quality on tumor-bearing brains.**
  `recon-all-clinical` + SynthSeg are robust to lesions but not
  perfect. We may need to fall back to `KUL_VBG` virtual brain
  grafting on a small subset; the design note already discusses this.
- **R-2 — Reference-mask edge cases.** A handful of UCSF-PDGM subjects
  have reference masks with empty ET; HD95 and Dice are then
  ill-defined. The metrics module already returns NaN; the table
  producer must handle NaN explicitly (median across non-NaN, plus
  per-compartment "n with non-empty ET" footnote).
- **R-3 — Bundle channel order.** Locked in PR-8 to
  `(T1c, T1, T2, FLAIR)`; any regression is caught by
  `tests/test_segment_unified.py::test_channel_order_matches_monai_bundle_metadata`.
- **R-4 — Raidionics / `segment_glioma` reproducibility.** Pin Docker
  digests in `pyproject.toml` / `environment.yml`; the design doc
  already commits to MONAI Bundle as canonical and Raidionics /
  `segment_glioma` only as comparators.

---

## 9. What ships in *this* commit (deliberately small)

This commit is the scoping commit, not PR-7 itself. It contains:

1. **`docs/scope_pr7.md`** — this file.
2. **`scripts/segment_all_cohort50.py`** — dry-run stub. Reads the
   cohort YAML and the metadata CSV, joins them, and for each of the
   50 subjects prints the artefact paths from Section 4 that the real
   fan-out runner will produce; checks that every input NIfTI exists
   (so the operator immediately sees if a subject is missing data).
   Real inference (`--no-dry-run`) raises
   `NotImplementedError("PR-7 not yet implemented; see docs/scope_pr7.md")`.
3. **`make scope-pr7`** — Make target that runs the dry-run on the
   locked cohort YAML and exits non-zero if any input is missing.
4. **`paper/revision_tracker.md`** — new "PR-7 (cohort fan-out +
   Tables 2/3 + Hit-Plot agreement panel)" stub row pointing at this
   document.

PR-7 itself is the next chunk of work.
