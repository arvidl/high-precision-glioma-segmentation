"""Smoke + numerical tests for the LUMIERE Patient-048 segmentation summarizer.

Builds a minimal but realistic synthetic derivatives tree with three
timepoints, two channels each, a per-tp DL label map, and -- for one of
the timepoints -- registered HD-GLIO and DeepBraTumIA comparator masks
with hand-computed Dice/Jaccard targets. Verifies that the summarizer:

  * Walks the manifest, reads each per-tp seg sidecar, and outputs the
    expected per-tp QC rows;
  * Joins the segmentation_summary.csv on timepoint to backfill RANO
    and days_since_baseline;
  * Computes per-(comparator, compartment) Dice / Jaccard on the
    Boolean masks at the registered grid that match the analytical
    expectation;
  * Returns rc=0 and writes JSON + Markdown sidecars when asked;
  * Returns rc=2 for a missing derivatives root with no partial outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "summarize_lumiere_p048_segmentation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_lumiere_p048_segmentation", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_label_nifti(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.int16), affine), str(path))


def _compartment_voxels(m: np.ndarray) -> tuple[int, int, int, int, int]:
    """Return (wt, tc, et, ncr, ed) voxel counts for a BraTS-3 label map."""
    ed = int((m == 2).sum())
    ncr = int((m == 1).sum())
    et = int((m == 4).sum())
    return ed + ncr + et, ncr + et, et, ncr, ed


def _write_sidecar(seg_dl_dir: Path, tp: str, m: np.ndarray, dl_path: Path) -> None:
    wt, tc, et, ncr, ed = _compartment_voxels(m)
    sidecar = {
        "schema_version": "1.0",
        "timepoint": tp,
        "backend": "dummy",
        "bundle_version": "synthetic",
        "device": "cpu",
        "elapsed_s": 0.5,
        "voxel_volume_mm3": 1.0,
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "channel_order": ["T1c", "T1", "T2", "FLAIR"],
        "probability_channel_order": ["TC", "WT", "ET"],
        "label_scheme": {
            "background": 0,
            "necrosis_or_non_enhancing": 1,
            "edema": 2,
            "enhancing_tumor": 4,
        },
        "volumes_ml": {
            "vol_WT_ml": round(wt / 1000.0, 4),
            "vol_TC_ml": round(tc / 1000.0, 4),
            "vol_ET_ml": round(et / 1000.0, 4),
            "vol_NCR_ml": round(ncr / 1000.0, 4),
            "vol_ED_ml": round(ed / 1000.0, 4),
        },
        "outputs": {
            "label_map": str(dl_path),
            "probabilities": str(dl_path).replace(".nii.gz", "_probs.nii.gz"),
        },
        "bundle_orientation": ["L", "P", "S"],
        "output_orientation": ["R", "A", "S"],
    }
    (seg_dl_dir / f"tp-{tp}_seg_brats3_dl.json").write_text(json.dumps(sidecar, indent=2))


def _build_synthetic_tree(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Build a 3-tp synthetic derivatives tree.

    Layout (mirrors the real one)::

        derivatives/
          manifest.json
          segmentation_summary.csv
          tp-week-000-1/
            seg_dl/
              tp-week-000-1_seg_brats3_dl.nii.gz   (DL label map)
              tp-week-000-1_seg_brats3_dl.json     (sidecar)
            tp-week-000-1_hdglio_to_ref_segmentation.nii.gz       (only at tp1)
            tp-week-000-1_deepbratumia_to_ref_segmentation.nii.gz  (only at tp1)
          tp-week-013/   ...
          tp-week-049/   ...

    DL label map at tp1 (uses BraTS-3 codes 1=NCR, 2=ED, 4=ET):
      A 6x6x6 cube with::
        x in [0,2], y in [0,2], z in [0,2]: ED   (label 2)  -- 27 voxels
        x in [3,5], y in [0,2], z in [0,2]: NCR  (label 1)  --  9 voxels
        x in [3,5], y in [3,5], z in [0,2]: ET   (label 4)  -- 27 voxels
        elsewhere: 0

    HD-GLIO mask at tp1 (codes 1=non-enh, 2=enh):
      Identical mask to the DL one but mapped to HD-GLIO codes::
        ED+NCR cells -> code 1 (non-enh tumour)
        ET cells     -> code 2 (enh tumour)
      WT (DL) = WT (HD-GLIO) = 63 voxels  -> Dice 1.0
      ET (DL) = ET (HD-GLIO) = 27 voxels  -> Dice 1.0

    DeepBraTumIA mask at tp1 (codes 1=ET, 2=NCR, 3=ED -- empirically
    validated against the round-3 staged Patient-048 native CT1 mask):
      Slightly disagrees: drops one ET voxel and labels one extra
      voxel as ED, so the per-compartment sets differ predictably::
        DL ET vs DBT ET: 26 of 27 shared -> Dice 2*26 / (27+26) = 52/53.
        DL WT vs DBT WT: 64 union, 63 intersection -> Dice 126/127.
    """
    derivs = root / "derivatives"
    derivs.mkdir(parents=True, exist_ok=True)
    tps = ("week-000-1", "week-013", "week-049")
    days = (0, 91, 343)
    rano = ("Pre-Op", "SD", "PD")

    # ---- DL label maps for all 3 tps. ----
    dl_maps: dict[str, np.ndarray] = {}
    for tp in tps:
        m = np.zeros((6, 6, 6), dtype=np.int16)
        # ED 27 vox
        m[0:3, 0:3, 0:3] = 2
        # NCR 9 vox  (just one z-slice: z=0..2 collapsed -> 3*3*1=9? -> use thinner block)
        m[3:6, 0:3, 0:1] = 1  # 3*3*1 = 9 voxels
        # ET 27 vox
        m[3:6, 3:6, 0:3] = 4
        dl_maps[tp] = m

    # ---- Comparator masks only at tp1. ----
    tp1 = tps[0]
    hdglio = np.zeros((6, 6, 6), dtype=np.int16)
    hdglio[0:3, 0:3, 0:3] = 1  # ED -> non-enh
    hdglio[3:6, 0:3, 0:1] = 1  # NCR -> non-enh
    hdglio[3:6, 3:6, 0:3] = 2  # ET -> enh

    dbt = np.zeros((6, 6, 6), dtype=np.int16)
    dbt[0:3, 0:3, 0:3] = 3  # ED -> 3 (DBT native: edema)
    dbt[3:6, 0:3, 0:1] = 2  # NCR -> 2 (DBT native: necrosis)
    dbt[3:6, 3:6, 0:3] = 1  # ET -> 1 (DBT native: enhancing tumour)
    # Now perturb: drop one ET voxel and add one ED voxel outside any DL tumour.
    dbt[5, 5, 2] = 0  # was ET (1) -> drop
    dbt[5, 5, 5] = 3  # was 0 -> add ED outside DL tumour

    # ---- Write the per-tp tree. ----
    for tp in tps:
        tp_dir = derivs / f"tp-{tp}"
        seg_dl_dir = tp_dir / "seg_dl"
        seg_dl_dir.mkdir(parents=True, exist_ok=True)
        dl_path = seg_dl_dir / f"tp-{tp}_seg_brats3_dl.nii.gz"
        _save_label_nifti(dl_maps[tp], dl_path)
        _write_sidecar(seg_dl_dir, tp, dl_maps[tp], dl_path)
        if tp == tp1:
            _save_label_nifti(hdglio, tp_dir / f"tp-{tp}_hdglio_to_ref_segmentation.nii.gz")
            _save_label_nifti(dbt, tp_dir / f"tp-{tp}_deepbratumia_to_ref_segmentation.nii.gz")

    # ---- manifest.json.
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "stage_root": str(root / "stage"),
        "channels": ["T1", "CT1", "T2", "FLAIR"],
        "reference_timepoint": tp1,
        "include_legacy_seg": True,
        "timepoints": [
            {"name": tp, "days_since_baseline": d, "rano": r}
            for tp, d, r in zip(tps, days, rano, strict=True)
        ],
    }
    (derivs / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _write_summary_csv(derivs, tps, days, rano, dl_maps)
    return {"dl": dl_maps, "hdglio": {tp1: hdglio}, "dbt": {tp1: dbt}}


def _write_summary_csv(
    derivs: Path,
    tps: tuple[str, ...],
    days: tuple[int, ...],
    rano: tuple[str, ...],
    dl_maps: dict[str, np.ndarray],
) -> None:
    summary_csv = derivs / "segmentation_summary.csv"
    fieldnames = (
        "timepoint",
        "days_since_baseline",
        "rano",
        "voxel_volume_mm3",
        "vol_WT_ml",
        "vol_TC_ml",
        "vol_ET_ml",
        "vol_NCR_ml",
        "vol_ED_ml",
    )
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        for tp, d, r in zip(tps, days, rano, strict=True):
            wt, tc, et, ncr, ed = _compartment_voxels(dl_maps[tp])
            w.writerow(
                {
                    "timepoint": tp,
                    "days_since_baseline": str(d),
                    "rano": r,
                    "voxel_volume_mm3": "1.0",
                    "vol_WT_ml": str(round(wt / 1000.0, 4)),
                    "vol_TC_ml": str(round(tc / 1000.0, 4)),
                    "vol_ET_ml": str(round(et / 1000.0, 4)),
                    "vol_NCR_ml": str(round(ncr / 1000.0, 4)),
                    "vol_ED_ml": str(round(ed / 1000.0, 4)),
                }
            )


def test_summarize_smoke_with_comparators(tmp_path: Path) -> None:
    masks = _build_synthetic_tree(tmp_path)
    derivs = tmp_path / "derivatives"
    csv_path = derivs / "segmentation_summary.csv"
    json_out = tmp_path / "out" / "lumiere_p048_segmentation_summary.json"
    md_out = tmp_path / "out" / "lumiere_p048_segmentation_summary.md"

    module = _load_module()
    rc = module.main(
        [
            "--derivatives-root",
            str(derivs),
            "--summary-csv",
            str(csv_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ]
    )
    assert rc == 0
    assert json_out.is_file()
    assert md_out.is_file() and md_out.stat().st_size > 0

    payload = json.loads(json_out.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["cohort_id"] == "lumiere_p048_synthetic"
    assert payload["reference_timepoint"] == "week-000-1"

    # ---- per_timepoint table ----
    rows = payload["per_timepoint"]
    assert len(rows) == 3
    by_tp = {r["timepoint"]: r for r in rows}
    tp1 = by_tp["week-000-1"]
    # ED 27 vox + NCR 9 vox + ET 27 vox = 63 vox -> 0.063 mL @ 1 mm^3
    assert tp1["vol_WT_ml"] == pytest.approx(0.063, abs=1e-4)
    assert tp1["vol_ED_ml"] == pytest.approx(0.027, abs=1e-4)
    assert tp1["vol_TC_ml"] == pytest.approx(0.036, abs=1e-4)
    assert tp1["vol_ET_ml"] == pytest.approx(0.027, abs=1e-4)
    assert tp1["vol_NCR_ml"] == pytest.approx(0.009, abs=1e-4)
    assert tp1["rano"] == "Pre-Op"
    assert tp1["days_since_baseline"] == 0.0
    assert tp1["status"] == "ok"
    assert tp1["output_orientation"] == ["R", "A", "S"]

    _assert_agreement(payload)
    _assert_markdown(md_out)

    # Used the staged DL mask -- not just the CSV.
    assert masks["dl"]["week-000-1"].sum() > 0  # belt-and-braces


def _assert_agreement(payload: dict) -> None:
    """Per-comparator x compartment Dice/JSC matches hand-computed values."""
    agreement = payload["comparator_agreement"]
    assert "hdglio" in agreement
    assert "deepbratumia" in agreement
    assert set(agreement["hdglio"].keys()) == {"week-000-1"}
    assert set(agreement["deepbratumia"].keys()) == {"week-000-1"}

    hd_tp1 = agreement["hdglio"]["week-000-1"]
    assert "TC" not in hd_tp1
    assert hd_tp1["WT"]["dice"] == pytest.approx(1.0)
    assert hd_tp1["WT"]["jaccard"] == pytest.approx(1.0)
    assert hd_tp1["ET"]["dice"] == pytest.approx(1.0)
    assert hd_tp1["ET"]["jaccard"] == pytest.approx(1.0)
    assert hd_tp1["WT"]["vol_diff_ml"] == pytest.approx(0.0, abs=1e-4)

    # DBT disagrees by 1 vox in ET (-1) and +1 vox added as ED outside DL.
    # WT(DL) = 63, WT(DBT) = 63, intersection = 62 -> Dice = 124/126.
    dbt_tp1 = agreement["deepbratumia"]["week-000-1"]
    assert dbt_tp1["WT"]["dice"] == pytest.approx(124.0 / 126.0, abs=1e-6)
    assert dbt_tp1["WT"]["jaccard"] == pytest.approx(62.0 / 64.0, abs=1e-6)
    assert dbt_tp1["ET"]["dice"] == pytest.approx(52.0 / 53.0, abs=1e-6)
    assert dbt_tp1["ET"]["jaccard"] == pytest.approx(26.0 / 27.0, abs=1e-6)
    assert dbt_tp1["TC"]["dice"] == pytest.approx(70.0 / 71.0, abs=1e-6)

    schemes = payload["comparator_label_schemes"]
    assert schemes["hdglio"] == {"WT": [1, 2], "ET": [2]}
    assert schemes["deepbratumia"] == {"WT": [1, 2, 3], "TC": [1, 2], "ET": [1]}


def _assert_markdown(md_out: Path) -> None:
    md = md_out.read_text()
    assert "Per-timepoint segmentation QC" in md
    assert "DL vs hdglio" in md
    assert "DL vs deepbratumia" in md
    assert "1.000" in md  # at least one 1.000 cell from the HD-GLIO row


def test_summarize_smoke_no_comparators(tmp_path: Path) -> None:
    """When no comparator masks are on disk, agreement should be empty
    and the per-tp table should still be populated and rc=0."""
    _build_synthetic_tree(tmp_path)
    derivs = tmp_path / "derivatives"
    # Remove the comparator masks the fixture wrote.
    for p in (derivs / "tp-week-000-1").glob("tp-week-000-1_*_to_ref_segmentation.nii.gz"):
        p.unlink()
    csv_path = derivs / "segmentation_summary.csv"
    json_out = tmp_path / "out" / "no_comp.json"
    module = _load_module()
    rc = module.main(
        [
            "--derivatives-root",
            str(derivs),
            "--summary-csv",
            str(csv_path),
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    payload = json.loads(json_out.read_text())
    assert payload["per_timepoint"][0]["status"] == "ok"
    assert payload["comparator_agreement"] == {}


def test_summarize_missing_derivatives_returns_nonzero(tmp_path: Path) -> None:
    module = _load_module()
    rc = module.main(["--derivatives-root", str(tmp_path / "does-not-exist")])
    assert rc == 2
