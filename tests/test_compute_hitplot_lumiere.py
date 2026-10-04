"""Integration smoke tests for the LUMIERE Patient-048 Hit-Plot runner.

Builds a tiny synthetic derivatives tree (two timepoints, 12^3 grid) with:
    * a real wmparc + LUT (5 region labels)
    * a real DL hard label map (BraTS-3) + sigmoid probs (4-D)
    * a real HD-GLIO 2-class comparator NIfTI
    * a real DeepBraTumIA 3-class native comparator NIfTI

Exercises the script end-to-end:
    * file-IO contract for every output (per-tp CSVs, cohort summary CSV+JSON)
    * source filtering (--source dl --source hdglio)
    * timepoint filtering (--timepoint week-013)
    * --skip-existing idempotence
    * graceful degradation when a comparator NIfTI is missing
    * comparator -> BraTS-3 remapping correctness (HD-GLIO emits WT+ET only;
      DBT emits WT+TC+ET; both produce non-trivial overlap rows)

The dummy data are deliberately small so each test runs in <1s and we never
shell out to any external tool.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_hitplot_lumiere_p048.py"

# ---------------------------------------------------------------------------
# Synthetic-tree builders.
# ---------------------------------------------------------------------------

_SHAPE = (12, 12, 12)
_AFFINE = np.eye(4, dtype=np.float64)
_TPS = ("week-000-1", "week-013")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("compute_hitplot_lumiere_p048", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, _AFFINE), str(path))


def _make_wmparc(derivatives_root: Path, tp: str) -> None:
    """Five-region wmparc occupying the central 8^3 cube; LUT covers all of them."""
    wm = np.zeros(_SHAPE, dtype=np.int16)
    wm[2:6, 2:10, 2:10] = 2  # Left-Cerebral-WM (FS code 2)
    wm[6:10, 2:10, 2:10] = 41  # Right-Cerebral-WM (FS code 41)
    wm[2:10, 2:10, 2:4] = 3  # Left-Cerebral-Cortex (FS code 3) -- overlaps WM, ok
    wm[2:10, 2:10, 8:10] = 42  # Right-Cerebral-Cortex (FS code 42)
    wm[5:7, 5:7, 5:7] = 16  # Brain-Stem (FS code 16) (small central region)
    parc_dir = derivatives_root / f"tp-{tp}" / "parcellation"
    _save(wm, parc_dir / f"tp-{tp}_wmparc_native.nii.gz")
    lut = {
        "0": "Unknown",
        "2": "Left-Cerebral-White-Matter",
        "3": "Left-Cerebral-Cortex",
        "16": "Brain-Stem",
        "41": "Right-Cerebral-White-Matter",
        "42": "Right-Cerebral-Cortex",
    }
    (parc_dir / f"tp-{tp}_wmparc_lut.json").write_text(json.dumps(lut, indent=2))


def _make_seg_dl(derivatives_root: Path, tp: str) -> None:
    """BraTS-3 hard label map + 4-D sigmoid probs in (X,Y,Z,C=3) layout (TC,WT,ET)."""
    seg_dir = derivatives_root / f"tp-{tp}" / "seg_dl"
    base = f"tp-{tp}_seg_brats3_dl"
    # Hard mask: a 4^3 NCR core with a 6^3 ET ring around it on the right side.
    label_map = np.zeros(_SHAPE, dtype=np.int16)
    label_map[6:10, 4:8, 4:8] = 2  # ED
    label_map[7:9, 5:7, 5:7] = 4  # ET inside the ED
    label_map[7:9, 5:7, 5:6] = 1  # NCR slice inside the ET
    _save(label_map, seg_dir / f"{base}.nii.gz")

    probs = np.zeros((*_SHAPE, 3), dtype=np.float32)  # (X,Y,Z, C={TC,WT,ET})
    probs[..., 1][label_map > 0] = 0.95  # WT prob inside any tumour
    probs[..., 0][np.isin(label_map, (1, 4))] = 0.85  # TC prob inside NCR+ET
    probs[..., 2][label_map == 4] = 0.75  # ET prob inside ET
    _save(probs, seg_dir / f"{base}_probs.nii.gz")
    sidecar = {
        "schema_version": "1.0",
        "voxel_volume_mm3": 1.0,
        "elapsed_s": 0.0,
        "output_orientation": ["R", "A", "S"],
        "device": "cpu",
        "backend": "synthetic",
        "bundle_version": "test-stub",
        "probability_channel_order": ["TC", "WT", "ET"],
    }
    (seg_dir / f"{base}.json").write_text(json.dumps(sidecar, indent=2))


def _make_hdglio(derivatives_root: Path, tp: str) -> None:
    """HD-GLIO 2-class mask: 1 = T2-hyperintense (BraTS ED), 2 = enhancing (BraTS ET)."""
    arr = np.zeros(_SHAPE, dtype=np.int16)
    arr[6:10, 4:8, 4:8] = 1
    arr[7:9, 5:7, 5:7] = 2
    _save(arr, derivatives_root / f"tp-{tp}" / f"tp-{tp}_hdglio_to_ref_segmentation.nii.gz")


def _make_dbt(derivatives_root: Path, tp: str) -> None:
    """DeepBraTumIA native 3-class: 1 = enhancing, 2 = necrosis, 3 = edema."""
    arr = np.zeros(_SHAPE, dtype=np.int16)
    arr[6:10, 4:8, 4:8] = 3  # edema
    arr[7:9, 5:7, 5:7] = 1  # enhancing
    arr[7:9, 5:7, 5:6] = 2  # necrosis
    _save(arr, derivatives_root / f"tp-{tp}" / f"tp-{tp}_deepbratumia_to_ref_segmentation.nii.gz")


def _build_synthetic_tree(
    derivatives_root: Path,
    *,
    skip_dbt: bool = False,
) -> None:
    for tp in _TPS:
        _make_wmparc(derivatives_root, tp)
        _make_seg_dl(derivatives_root, tp)
        _make_hdglio(derivatives_root, tp)
        if not skip_dbt:
            _make_dbt(derivatives_root, tp)


def _write_manifest(path: Path) -> None:
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "reference_timepoint": "week-000-1",
        "channels": ["T1", "CT1", "T2", "FLAIR"],
        "timepoints": [
            {"name": "week-000-1", "days_since_baseline": 0, "rano": "Pre-Op"},
            {"name": "week-013", "days_since_baseline": 91, "rano": "SD"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_compute_hitplot_lumiere_full_smoke(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_tree(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
        ]
    )
    assert rc == 0, "all 8 (tp, source) pairs must succeed"

    # Per-(tp, source) CSVs all present with the documented schema.
    for tp in _TPS:
        for source in ("dl", "dl_prob", "hdglio", "dbt"):
            csv_path = derivatives_root / f"tp-{tp}" / "hitplot" / f"tp-{tp}_hitplot_{source}.csv"
            assert csv_path.is_file(), f"missing {csv_path}"
            with csv_path.open() as fh:
                rows = list(csv.DictReader(fh))
            assert rows, f"empty CSV: {csv_path}"
            cols = set(rows[0].keys())
            # Common columns across deterministic + probabilistic schemata.
            for required in ("label", "name", "compartment", "parcel_voxels"):
                assert required in cols, f"{csv_path}: missing column {required}"
            comps = {r["compartment"] for r in rows}
            if source == "hdglio":
                assert comps == {"WT", "ET"}, f"HD-GLIO must skip TC, got {comps}"
            else:
                assert comps == {"WT", "TC", "ET"}, f"{source}: bad compartments {comps}"

    # Cohort summary CSV + JSON.
    csv_path = derivatives_root / "hitplot_summary.csv"
    json_path = derivatives_root / "hitplot_summary.json"
    assert csv_path.is_file() and json_path.is_file()
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    # Two timepoints x four sources = eight rows.
    assert len(rows) == 8
    assert {r["status"] for r in rows} == {"ok"}
    assert {r["source"] for r in rows} == {"dl", "dl_prob", "hdglio", "dbt"}
    assert {r["timepoint"] for r in rows} == set(_TPS)

    payload = json.loads(json_path.read_text())
    assert payload["cohort_id"] == "lumiere_p048_synthetic"
    assert payload["patient_id"] == "048"
    assert payload["sources"] == ["dl", "dl_prob", "hdglio", "dbt"]
    # Provenance: comparator remap dictionaries are explicit in the JSON.
    assert payload["comparator_label_remap"]["hdglio"] == {"1": 2, "2": 4}
    assert payload["comparator_label_remap"]["dbt"] == {"1": 4, "2": 1, "3": 2}


def test_compute_hitplot_lumiere_source_and_timepoint_filter(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_tree(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--timepoint",
            "week-013",
            "--source",
            "dl",
            "--source",
            "hdglio",
        ]
    )
    assert rc == 0
    csv_path = derivatives_root / "hitplot_summary.csv"
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["timepoint"] for r in rows} == {"week-013"}
    assert {r["source"] for r in rows} == {"dl", "hdglio"}
    assert (derivatives_root / "tp-week-013" / "hitplot" / "tp-week-013_hitplot_dl.csv").is_file()
    assert not (
        derivatives_root / "tp-week-000-1" / "hitplot" / "tp-week-000-1_hitplot_dl.csv"
    ).is_file()


def test_compute_hitplot_lumiere_skip_existing_is_idempotent(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_tree(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    args = [
        "--manifest",
        str(manifest_path),
        "--derivatives-root",
        str(derivatives_root),
    ]
    assert module.main(args) == 0
    csv_path = derivatives_root / "tp-week-013" / "hitplot" / "tp-week-013_hitplot_dl.csv"
    first_mtime = csv_path.stat().st_mtime

    # Re-run with --skip-existing: per-tp CSVs must NOT be rewritten.
    assert module.main([*args, "--skip-existing"]) == 0
    assert csv_path.stat().st_mtime == first_mtime

    # Cohort summary should still be regenerated and reflect the skip.
    summary = derivatives_root / "hitplot_summary.csv"
    with summary.open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["status"] for r in rows} == {"skipped"}


def test_compute_hitplot_lumiere_missing_comparator_is_graceful(tmp_path: Path) -> None:
    """When a comparator NIfTI is absent the runner records ``missing_input``
    for that one (tp, source) pair and continues with the rest; exit code 0
    because ``missing_input`` is not an ``error``.
    """
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_tree(derivatives_root, skip_dbt=True)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
        ]
    )
    assert rc == 0
    summary = derivatives_root / "hitplot_summary.csv"
    with summary.open() as fh:
        rows = list(csv.DictReader(fh))
    statuses = {(r["timepoint"], r["source"]): r["status"] for r in rows}
    for tp in _TPS:
        assert statuses[(tp, "dl")] == "ok"
        assert statuses[(tp, "dl_prob")] == "ok"
        assert statuses[(tp, "hdglio")] == "ok"
        assert statuses[(tp, "dbt")] == "missing_input"


def test_compute_hitplot_lumiere_remap_comparator_to_brats() -> None:
    """Unit test the comparator -> BraTS-3 remapping in isolation."""
    module = _load_script_module()

    hdglio = np.array([[0, 1], [2, 1]], dtype=np.int32)
    out = module._remap_comparator_to_brats(hdglio, "hdglio")
    np.testing.assert_array_equal(out, np.array([[0, 2], [4, 2]], dtype=np.int32))

    dbt = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int32)
    out = module._remap_comparator_to_brats(dbt, "dbt")
    np.testing.assert_array_equal(out, np.array([[0, 4, 1], [2, 4, 1]], dtype=np.int32))

    # Unknown source raises.
    import pytest

    with pytest.raises(ValueError, match="No comparator remap"):
        module._remap_comparator_to_brats(hdglio, "raidionics")
