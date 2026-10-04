---
title: "Future-work note --- methodology-matched comparator benchmark (Raidionics, segment_glioma)"
subtitle: "HPGS revision (JMET)"
author: "Arvid Lundervold and the HPGS team"
date: "2026-04-18"
toc: true
toc-depth: 2
numbersections: true
geometry: margin=2.5cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

# Purpose

A quantitative comparison of HPGS against established comparator pipelines
(Raidionics and `segment_glioma`) is **not** reported on the same $n=50$
UCSF-PDGM evaluation cohort, for two methodological reasons that are
documented in `paper/main.tex` Discussion (the
*substitution-to-MONAI-Bundle* paragraph) and in Limitations item~(v).
This note specifies the standing harness for picking the comparison up
in a follow-up paper, and the methodological constraints the follow-up
must respect to produce a fair comparison.

This note records why a same-cohort head-to-head against Raidionics and
`segment_glioma` is a separate, methodology-matched study and not a table
in this paper.

# Why the head-to-head is not in PR-16

Two structural asymmetries between HPGS and the comparators determine
the outcome of any single-cohort paired comparison, independently of
segmenter quality.

## A1 --- Spatial frame of reference (Raidionics)

Raidionics is an **atlas-warped** pipeline: every input is registered
to a standardised atlas in MNI space, segmented there, and inverse-warped
back if subject-native output is requested. HPGS evaluates everything in
**native subject space** (this is the reason the new Methods §
*Evaluation framework* block names the per-subject sidecar contract).

A same-cohort, same-table comparison forces one of two choices, neither
of them fair:

- **Option A1a --- evaluate in native space.** Raidionics output must
  be inverse-warped back to native, which interpolates a discrete
  label map across a non-rigid transform. Interpolation across discrete
  labels biases small compartments downward (the enhancing tumour is
  most affected), which makes Raidionics look worse than it is.
- **Option A1b --- evaluate in MNI space.** HPGS predictions must be
  forward-warped to MNI, which is the architecture HPGS argues
  *against* in Methods § *Evaluation framework* and in the
  Discussion's contrast with atlas-warped pipelines. It also requires
  warping the UCSF-PDGM reference masks themselves, which were
  supplied in native space and which would acquire the same
  interpolation artefacts.

The methodology-matched alternative is to compare on a cohort where
one spatial frame is genuinely shared (e.g.\ a public BraTS subset
with both a native-space reference *and* an MNI-space one), or to
report the comparison in a representation that is invariant to the
warp (e.g.\ regional tumour-burden profiles aggregated into anatomical
parcels) rather than at the voxel level. Both designs are explicitly
out of scope for the present revision.

## A2 --- Preprocessing and operating-space contract (`segment_glioma`)

The preoperative `segment_glioma` / PICTURE nnU-Net model is trained
and deployed on a **multiparametric** input stack {T1, T1c, T2, FLAIR}
after the PICTURE preprocessing contract: rigid registration to T1c,
N4 bias correction, HD-BET skull stripping, and affine registration
to the SRI-24 atlas (Pemberton et al. 2023). HPGS evaluates on the
**dataset-provided native co-registered UCSF-PDGM volumes** without
re-running that SRI-24 pipeline. On the same $n=50$ cohort, a paired
Dice / HD$_{95}$ / $|\mathrm{VE}|$ table would therefore confound
segmenter identity with preprocessing and operating-space differences,
not isolate backbone quality.

The methodology-matched alternative is either (i) run both segmenters
through the same preprocessing and spatial-frame contract before
scoring (e.g. the full PICTURE stack for `segment_glioma` and a
deliberately matched native-space or SRI-24-space HPGS run), or (ii)
compare at a warp-invariant representation (e.g. regional tumour-burden
profiles in anatomical parcels) rather than at mismatched voxel grids.
Neither design is a same-cohort drop-in for the present Table~3 slot.

# What is already in place (the standing harness)

The PR-7e1 work landed the full producer chain for the Table~3 slot
that PR-16 is no longer populating:

| File | State | Purpose |
|------|-------|---------|
| `src/hpgs/segment/baselines.py` | EXISTS | Adapter for `raidionics` and `segmentglioma` with `dummy` and `docker` backends. The `docker` backend currently raises `NotImplementedError("PR-7e2 will land the real glue")`. |
| `scripts/run_baseline_segmenters_cohort50.py` | EXISTS | Cohort runner that loops `predict_baseline` over the n=50 subjects and writes per-subject sidecars under `data/derivatives_cohort50/sub-XXXX/seg_baselines/`. |
| `scripts/build_table3_segmenter_agreement.py` | EXISTS, 714 lines | Table~3 producer. Reads per-subject `metrics_dl_vs_{raidionics,segmentglioma,gt}.json` sidecars, emits four-column LaTeX table (Raidionics / segment_glioma / BraTS21-manual). Tolerates missing sidecars (auto-generates `--` cells with footnote-friendly per-cell denominators). |
| `tests/test_build_table3_segmenter_agreement.py` | EXISTS, 999 lines | Producer-side tests; pass on the empty-sidecar path. |
| `Makefile` targets `baselines-all`, `table3-segmenter-agreement` | EXIST | One-line invocation to (re-)run the chain end-to-end. |

What is **missing** for the chain to actually produce numbers:

1. `configs/baselines.yaml` --- pinned upstream Docker image digests
   for Raidionics and `segment_glioma` (PR-7e2 deliverable).
2. The real Docker glue inside `src/hpgs/segment/baselines.py`
   (currently raises `NotImplementedError`; PR-7e2 deliverable).
3. A methodology-matched cohort design that addresses A1 and A2 above
   (this is the *follow-up paper* deliverable, not a code task).
4. The actual per-subject `metrics_dl_vs_{raidionics,segmentglioma}.json`
   sidecars from running the chain on the cohort designed in (3).

# Recommended follow-up scope

A reasonable follow-up paper specifies:

1. **Cohort design** that respects A1 + A2 (e.g.\ a multi-cohort
   design where the spatial-frame comparison and the modality
   comparison are factored out into separate sub-tables, rather than
   conflated in a single $n=50$ row).
2. **Per-cohort evaluation strategy** documented in a new
   `\rev{}` Methods block analogous to PR-14's Evaluation framework
   block, identifying the spatial frame and the modality contract
   used for each comparator.
3. **Three sub-tables** (one per spatial-frame / modality regime)
   rather than a single conflated table, each with the
   methodology-matched cohort and metric set declared in its caption.
4. **No re-implementation of the harness** --- the PR-7e1 producer
   chain above is the starting point. Only the missing items
   (`configs/baselines.yaml` + Docker glue + cohort design) need to be
   landed.

# Citation hooks for the follow-up

Once minted, the follow-up paper should cite:

- This revision's Methods § *Evaluation framework* block (PR-14) as
  the principle behind the methodology-matched cohort design.
- This revision's Limitations item (v) (PR-15 + PR-16) as the
  explicit pre-registration of the comparator benchmark as future
  work.
- The Zenodo DOI of `v2.0-revised` (when minted) as the artefact-of-
  record for the unified-engine numbers (Table~2 + Table~4) the
  comparator benchmark is being run *against*.
