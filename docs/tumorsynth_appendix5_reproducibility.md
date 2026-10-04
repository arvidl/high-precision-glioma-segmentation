# TumorSynth Appendix 5 — reproducibility runbook

This note explains how to regenerate **Appendix 5** in the JMET revision manuscript
from first principles: an audited inner-label mapping, a full `n=50` TumorSynth
comparator run, downstream Hit-Plot artefacts, synced `paper/figs/` assets, and
updated response prose.

The canonical segmentation backbone remains the MONAI Bundle BraTS
three-compartment SegResNet. TumorSynth is framed as a **bounded
segmentation-backbone sensitivity analysis** on the same staged mpMRI inputs, not
as a benchmark under TumorSynth's ideal SRI-24 preprocessing regime.

Related docs:

- [`tumorsynth_comparator_workflow.md`](tumorsynth_comparator_workflow.md) — adapter
  contract, Make targets, Ubuntu CUDA setup
- [`tumorsynth_smoke_run_notes.md`](tumorsynth_smoke_run_notes.md) — early smoke-run
  timings and wrapper patches
- [`../paper/figs/README.md`](../paper/figs/README.md) — manuscript asset inventory
- [`../paper/README.md`](../paper/README.md) — how to build `paper/main.tex`

## What Appendix 5 consumes

| Manuscript asset | Source under `outputs/` | Producer |
|---|---|---|
| `paper/figs/table_tumorsynth_agreement.tex` | `outputs/tables/table_tumorsynth_agreement.tex` | `make table-tumorsynth-agreement` |
| `paper/figs/fig_hitplot_agreement_dl_vs_tumorsynth.pdf` | `outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.pdf` | `make agreement-panel` (`AGREEMENT_REFERENCE=tumorsynth`) |
| `paper/figs/fig_functional_anatomy_hitplot_tumorsynth.pdf` | `outputs/figures/fig_functional_anatomy_hitplot_tumorsynth.pdf` | `make functional-anatomy-hitplot FUNCTIONAL_HITPLOT_SOURCE=tumorsynth` |

Prose in `paper/main.tex` cites median Dice / CCC numbers that must match
the synced table and agreement-panel JSON after regeneration.

## Audited inner-label mapping (locked default)

A five-subject raw-label audit (`0005`, `0012`, `0018`, `0026`, `0035`) compared
raw TumorSynth inner labels `1`, `2`, and `3` against both the MONAI Bundle DL
masks and the UCSF-PDGM reference masks. The audit script is
`scripts/audit_tumorsynth_raw_labels.py`; its summary is written to
`outputs/tables/tumorsynth_raw_label_crosswalk.{csv,json}`.

Evidence across all five subjects:

| Raw inner label | Best match (DL and reference) |
|---|---|
| `1` | ED |
| `2` | ET |
| `3` | NCR |

The HPGS adapter therefore defaults to:

```text
TUMORSYNTH_WHOLE_LABEL  = 18
TUMORSYNTH_NCR_LABELS   = 3
TUMORSYNTH_ED_LABELS    = 1
TUMORSYNTH_ET_LABELS    = 2
```

These defaults are wired in `src/hpgs/segment/baselines.py` and the `Makefile`.
Each successful run records the mapping in
`data/derivatives_cohort50/sub-XXXX/seg_tumorsynth/sub-XXXX_seg_tumorsynth.json`
under the `tumorsynth.inner_*_labels` fields.

**Do not use compartment metrics from earlier smoke runs** that assumed
`1 -> NCR; 2 -> ED; 3 -> ET` or `1,3 -> NCR; 2 -> ET`. Those mappings inflate or
deflate TC/ET agreement artefactually.

## Why reproducibility can drift

Two practical facts explain common “paper says n=50 but my laptop only has five
subjects” confusion:

1. **`data/derivatives_cohort50/` is gitignored.** The committed manuscript assets
   live under `paper/figs/`, mirrored from `outputs/` by `make sync`. A fresh clone
   does not contain per-subject TumorSynth sidecars unless you rerun the pipeline
   or copy derivatives from the machine that produced them.

2. **TumorSynth is slow on CPU.** A full `n=50` pass on Apple Silicon CPU mode
   can take on the order of **1–2 days** wall clock. The documented production
   run used **Ubuntu 24.04 + NVIDIA CUDA**; see
   [`tumorsynth_comparator_workflow.md`](tumorsynth_comparator_workflow.md).

## Step 0 — choose the execution machine

| Machine | When to use |
|---|---|
| **Ubuntu + CUDA** (documented RTX A5000 setup) | Preferred for the full `n=50` TumorSynth fan-out |
| **macOS CPU** | Smoke subsets only (`0005`, `0026`, …), not full-cohort regeneration before submission |

Use the HPGS repo `.venv` for Python (`uv run` / `make PY=...`). TumorSynth itself
runs from a separate legacy nnU-Net v1.7 environment that supplies `nnUNet_predict`
on `PATH`.

## Step 1 — verify existing artefacts before re-running

On the machine you intend to treat as source of truth:

```bash
find data/derivatives_cohort50 -name '*metrics_dl_vs_tumorsynth.json' | wc -l
# expect: 50 for a complete Appendix 5 regeneration
```

Check that sidecars use the audited mapping:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/derivatives_cohort50")
paths = sorted(root.glob("sub-*/metrics/*_metrics_dl_vs_tumorsynth.json"))
stale = []
for metrics_path in paths:
    sidecar_path = metrics_path.parent.parent / "seg_tumorsynth" / (
        metrics_path.name.replace("_metrics_dl_vs_tumorsynth.json", "_seg_tumorsynth.json")
    )
    ts = json.loads(sidecar_path.read_text())["tumorsynth"]
    if (
        ts["inner_ncr_labels"] != [3]
        or ts["inner_ed_labels"] != [1]
        or ts["inner_et_labels"] != [2]
    ):
        stale.append(sidecar_path.parent.name)

print(f"n_sidecars={len(paths)}")
print(f"stale_mapping={stale}")
PY
```

Interpretation:

- **`n_sidecars=50` and `stale_mapping=[]`** — skip Step 2; proceed to Step 4
  (downstream rebuild only).
- **Fewer than 50 sidecars, or any stale mapping** — run Step 2 and Step 3.

Optional mapping audit on the five-subject evidence set:

```bash
uv run python scripts/audit_tumorsynth_raw_labels.py \
  --subject 0005 --subject 0012 --subject 0018 \
  --subject 0026 --subject 0035
```

## Step 2 — invalidate stale TumorSynth outputs

Do **not** use `BASELINE_SKIP_EXISTING=1` until every subject on disk carries the
audited mapping.

Remove stale outputs for known smoke subjects (extend the list if needed):

```bash
for sid in 0005 0012 0018 0026 0035; do
  rm -rf "data/derivatives_cohort50/sub-${sid}/seg_tumorsynth"
  rm -f  "data/derivatives_cohort50/sub-${sid}/metrics/sub-${sid}_metrics_dl_vs_tumorsynth.json"
  rm -f  "data/derivatives_cohort50/sub-${sid}/hitplot/sub-${sid}_hitplot_tumorsynth.csv"
done
```

For a full refresh, delete `seg_tumorsynth/` and `metrics/*_metrics_dl_vs_tumorsynth.json`
for all cohort subjects instead of only the smoke subset.

## Step 3 — run TumorSynth for all 50 subjects

### Ubuntu + CUDA (recommended)

See [`tumorsynth_comparator_workflow.md`](tumorsynth_comparator_workflow.md) for
environment creation, model-tree paths, and the patched `mri_tumorsynth_hpgs`
wrapper.

```bash
source .venv-nnunet-v17-cuda/bin/activate
export PATH="$PWD/.venv-nnunet-v17-cuda/bin:$PATH"
export NNUNET_ENV_DIR=/path/to/nnUNet
export nnUNet_raw_data_base=/path/to/nnUNet/nnUNet_v1.7/nnUNet_raw_data_base
export nnUNet_preprocessed=/path/to/nnUNet/nnUNet_v1.7/nnUNet_preprocessed
export RESULTS_FOLDER=/path/to/nnUNet/nnUNet_v1.7/nnUNet_trained_models

make tumorsynth-all \
  PY="$PWD/.venv/bin/python" \
  TUMORSYNTH_COMMAND=/path/to/mri_tumorsynth_hpgs \
  TUMORSYNTH_NNUNET_DIR=/path/to/nnUNet \
  TUMORSYNTH_THREADS=16 \
  BASELINE_SKIP_EXISTING=1
```

Use `BASELINE_SKIP_EXISTING=1` only after Step 2 has cleared stale subjects.
Omit it (or set to empty) when forcing a full rewrite.

### macOS CPU (smoke / subset only)

```bash
conda run -n nnUNet_v1.7 bash -lc '
  export PATH=/opt/anaconda3/envs/nnUNet_v1.7/bin:$PATH
  export NNUNET_ENV_DIR=/Users/arvid/nnUNet
  export nnUNet_raw_data_base=/Users/arvid/nnUNet/nnUNet_raw_data_base
  export nnUNet_preprocessed=/Users/arvid/nnUNet/nnUNet_preprocessed
  export RESULTS_FOLDER=/Users/arvid/nnUNet
  make tumorsynth-all \
    TUMORSYNTH_COMMAND=/Users/arvid/nnUNet/mri_TumorSynth_hpgs \
    TUMORSYNTH_NNUNET_DIR=/Users/arvid/nnUNet \
    TUMORSYNTH_CPU=1 \
    TUMORSYNTH_THREADS=16 \
    BASELINE_SUBJECTS="0005 0026"
'
```

**Gate after Step 3:**

- `find ... metrics_dl_vs_tumorsynth.json | wc -l` equals **50**
- every sidecar has `inner_ncr_labels: [3]`, `inner_ed_labels: [1]`,
  `inner_et_labels: [2]`
- raw whole/inner NIfTIs exist under each `seg_tumorsynth/` tree when preservation
  is enabled (see sidecar `outputs.tumorsynth_*_raw` paths)

## Step 4 — rebuild Appendix 5 downstream artefacts

Prerequisites from the main revision pipeline (should already exist on disk):

- `data/derivatives_cohort50/sub-XXXX/seg_dl/` — MONAI Bundle masks
- `data/derivatives_cohort50/sub-XXXX/parcellation/` — FreeSurfer wmparc fan-out
- `data/derivatives_cohort50/sub-XXXX/hitplot/sub-XXXX_hitplot_dl.csv` (and GT)

Run the TumorSynth downstream chain:

```bash
# Per-subject TumorSynth Hit-Plot CSVs
make hitplot-all HITPLOT_SOURCES=tumorsynth HITPLOT_SKIP_EXISTING=1

# Appendix 5 voxel-space agreement table
make table-tumorsynth-agreement TUMORSYNTH_TABLE_STRICT=1

# Appendix 5 Hit-Plot agreement panel (DL vs TumorSynth)
make agreement-panel \
  AGREEMENT_PREDICTION=dl \
  AGREEMENT_REFERENCE=tumorsynth \
  AGREEMENT_STRICT=1

# Appendix 5 functional-anatomy aggregation
make functional-anatomy-hitplot FUNCTIONAL_HITPLOT_SOURCE=tumorsynth
```

**Gate after Step 4:**

- `outputs/tables/table_tumorsynth_agreement.json` reports complete cohort coverage
  (header comment in the `.tex` file should read `n_with_sidecar=50`)
- `outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.json` exists and lists
  per-compartment CCC summaries
- `outputs/figures/fig_functional_anatomy_hitplot_tumorsynth.pdf` exists

## Step 5 — sync into the manuscript and rebuild PDFs

```bash
make sync
make paper
```

`make sync` copies generated assets from `outputs/figures/` and `outputs/tables/`
into `paper/figs/`. Do not hand-edit mirrored files; regenerate via Make targets.

## Step 6 — reconcile prose numbers

After regeneration, verify that hard-coded numbers in the TeX sources match the
new artefacts:

| Location | What to check |
|---|---|
| `paper/main.tex` Appendix 5 | table caption; interpretation paragraph |

Read the canonical values from:

```bash
python3 - <<'PY'
import json
from pathlib import Path

table = json.loads(Path("outputs/tables/table_tumorsynth_agreement.json").read_text())
panel = json.loads(
    Path("outputs/figures/fig_hitplot_agreement_dl_vs_tumorsynth.json").read_text()
)
print("table summary keys:", sorted(table.keys())[:8], "...")
print("panel compartments:", sorted(panel.get("summary", panel).keys()))
PY
```

Update TeX sentences if medians or CCC shift. The manuscript numbers must
agree with `paper/figs/table_tumorsynth_agreement.tex`.

## Step 7 — rebuild the author manuscript

```bash
uv run pytest -q tests/test_build_table_tumorsynth_agreement.py
cd paper && latexmk -pdf main.tex
```

Manual PDF checks:

- Appendix 5 table renders with `n=50` footnotes as expected
- Hit-Plot agreement panel and functional-anatomy panel are present
- bounded-comparator caveat language is intact

## Decision table

| Situation | Recommended action |
|---|---|
| Ubuntu box has 50 sidecars with audited mapping | Run Steps 4–7 only on that machine |
| Fresh clone; only 5 local sidecars with old mapping | Do **not** treat `paper/figs/` as proof of local reproducibility; run Steps 2–7 on Ubuntu |
| Full n=50 rerun infeasible before deadline | Last resort: remove or downgrade Appendix 5 claims in manuscript + responses (requires explicit editorial decision) |

## End-to-end command summary

```bash
# 1) verify / invalidate (see Steps 1–2 above)

# 2) segment
make tumorsynth-all ... BASELINE_SKIP_EXISTING=1

# 3) downstream Appendix 5
make hitplot-all HITPLOT_SOURCES=tumorsynth HITPLOT_SKIP_EXISTING=1
make table-tumorsynth-agreement TUMORSYNTH_TABLE_STRICT=1
make agreement-panel AGREEMENT_PREDICTION=dl AGREEMENT_REFERENCE=tumorsynth AGREEMENT_STRICT=1
make functional-anatomy-hitplot FUNCTIONAL_HITPLOT_SOURCE=tumorsynth

# 4) manuscript
make sync
make paper
make check-drift
```
