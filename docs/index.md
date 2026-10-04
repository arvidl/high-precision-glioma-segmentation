# HPGS — documentation

This is the documentation home for `high-precision-glioma-segmentation`. Build with
`mkdocs serve` once the toolchain is added (Cursor TODO).

## Contents

- [Quickstart](../README.md)
- [Software installation and reproduction guide](software_installation.md)
- [TumorSynth comparator workflow](tumorsynth_comparator_workflow.md)
- [TumorSynth Appendix 5 reproducibility runbook](tumorsynth_appendix5_reproducibility.md)
- [**Build system — `make` targets**](build.md)
- [Cohort definition](../configs/cohort_ucsfpdgm_n50.yaml)
- [Pipeline configuration](../configs/default.yaml)
- [Paper subfolder and how to build main.tex](../paper/README.md)
- [Agent / AI rules](../AGENTS.md)

### Design notes

- [Future-work note on methodology-matched comparator benchmarks](future_work_segmenter_comparators.md)

PDF renderings of both notes are built alongside the Markdown files (same basename, `.pdf` extension) — see `make docs` (or run `pandoc` directly as documented in [build.md](build.md)).

## Overview

The pipeline takes a multiparametric MRI exam and produces:

1. A native-space brain parcellation (FreeSurfer 8.2.0 SynthSeg).
2. A multi-class tumor segmentation (BraTS labels) from fastMONAI / nnU-Net v2.
3. Standard segmentation metrics against the dataset's expert masks.
4. A regional tumor-burden table and a Hit-Plot visualisation.

All stages are configurable from `configs/default.yaml` and runnable from the CLI.
