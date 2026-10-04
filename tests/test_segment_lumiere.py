"""Integration smoke test for the LUMIERE Patient-048 segmentation runner.

Uses the deterministic ``dummy`` backend of ``hpgs.segment.unified.predict_brats3``
so the test does not need MONAI weights, GPU/MPS, or network access. The point
is to exercise the script's *file-IO contract* end-to-end (channel paths,
output paths, sidecar JSON shape, cohort CSV columns, BraTS label scheme),
not the segmentation accuracy of the dummy field.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "segment_lumiere_p048.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("segment_lumiere_p048", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _gaussian_blob(shape: tuple[int, int, int], centre: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float32)
    cz, cy, cx = centre
    r2 = (grid[0] - cz) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cx) ** 2
    return np.exp(-r2 / (2 * 4.0**2)).astype(np.float32)


def _build_synthetic_derivatives(derivatives_root: Path, channels: list[str]) -> None:
    """Two timepoints, four channels each, all 24^3 RAS NIfTIs."""
    shape = (24, 24, 24)
    affine = np.eye(4)
    for tp_name, centre in (("week-000-1", (12, 12, 12)), ("week-013", (13, 12, 11))):
        for ch in channels:
            _save_nifti(
                _gaussian_blob(shape, centre),
                affine,
                derivatives_root / f"tp-{tp_name}" / f"tp-{tp_name}_{ch}_to_ref.nii.gz",
            )


def _write_manifest(path: Path, channels: list[str]) -> None:
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "reference_timepoint": "week-000-1",
        "channels": channels,
        "timepoints": [
            {"name": "week-000-1", "days_since_baseline": 0, "rano": "Pre-Op"},
            {"name": "week-013", "days_since_baseline": 91, "rano": "SD"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh)


CHANNELS = ["T1", "CT1", "T2", "FLAIR"]


def test_segment_lumiere_smoke_dummy_backend(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"

    _build_synthetic_derivatives(derivatives_root, CHANNELS)
    _write_manifest(manifest_path, CHANNELS)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "dummy",
        ]
    )
    assert rc == 0

    # Per-tp artefacts.
    for tp in ("week-000-1", "week-013"):
        seg_dir = derivatives_root / f"tp-{tp}" / "seg_dl"
        label_map = seg_dir / f"tp-{tp}_seg_brats3_dl.nii.gz"
        probs = seg_dir / f"tp-{tp}_seg_brats3_dl_probs.nii.gz"
        sidecar_path = seg_dir / f"tp-{tp}_seg_brats3_dl.json"
        assert label_map.is_file()
        assert probs.is_file()
        assert sidecar_path.is_file()

        lm_img = nib.load(str(label_map))
        lm = np.asarray(lm_img.dataobj)
        assert lm.dtype == np.uint8
        assert lm.shape == (24, 24, 24)
        # The dummy backend is centred at the volume midpoint, so every BraTS
        # class (NCR=1, ED=2, ET=4) should be present.
        labels = set(int(v) for v in np.unique(lm))
        assert {0, 1, 2, 4}.issubset(labels), f"unexpected label set: {labels}"

        probs_img = nib.load(str(probs))
        probs_arr = np.asarray(probs_img.dataobj)
        assert probs_arr.shape == (24, 24, 24, 3)

        sc = json.loads(sidecar_path.read_text())
        assert sc["timepoint"] == tp
        assert sc["backend"] == "dummy"
        assert sc["channel_order"] == ["T1c", "T1", "T2", "FLAIR"]
        assert sc["probability_channel_order"] == ["TC", "WT", "ET"]
        for k in ("vol_WT_ml", "vol_TC_ml", "vol_ET_ml", "vol_NCR_ml", "vol_ED_ml"):
            assert k in sc["volumes_ml"]
        # Sub-region monotonicity: ET subset-of TC subset-of WT.
        v = sc["volumes_ml"]
        assert v["vol_ET_ml"] <= v["vol_TC_ml"] + 1e-6
        assert v["vol_TC_ml"] <= v["vol_WT_ml"] + 1e-6

    # Cohort summary CSV + JSON.
    csv_path = derivatives_root / "segmentation_summary.csv"
    json_path = derivatives_root / "segmentation_summary.json"
    assert csv_path.is_file()
    assert json_path.is_file()
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["timepoint"] for r in rows] == ["week-000-1", "week-013"]
    assert rows[0]["rano"] == "Pre-Op"
    assert rows[1]["rano"] == "SD"
    for r in rows:
        for k in ("vol_WT_ml", "vol_TC_ml", "vol_ET_ml"):
            assert float(r[k]) > 0.0, f"expected non-zero {k} from dummy backend"


def test_segment_lumiere_skip_existing_is_idempotent(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root, CHANNELS)
    _write_manifest(manifest_path, CHANNELS)

    module = _load_script_module()
    args = [
        "--manifest",
        str(manifest_path),
        "--derivatives-root",
        str(derivatives_root),
        "--backend",
        "dummy",
    ]
    assert module.main(args) == 0
    first_csv = (derivatives_root / "segmentation_summary.csv").read_text()
    first_label_mtime = (
        (derivatives_root / "tp-week-013" / "seg_dl" / "tp-week-013_seg_brats3_dl.nii.gz")
        .stat()
        .st_mtime
    )

    assert module.main([*args, "--skip-existing"]) == 0
    second_csv = (derivatives_root / "segmentation_summary.csv").read_text()
    second_label_mtime = (
        (derivatives_root / "tp-week-013" / "seg_dl" / "tp-week-013_seg_brats3_dl.nii.gz")
        .stat()
        .st_mtime
    )

    assert first_csv == second_csv
    assert (
        first_label_mtime == second_label_mtime
    ), "with --skip-existing the label map should not have been rewritten"


def test_segment_lumiere_missing_channel_is_reported_not_raised(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root, CHANNELS)
    _write_manifest(manifest_path, CHANNELS)

    bad_channel = derivatives_root / "tp-week-013" / "tp-week-013_FLAIR_to_ref.nii.gz"
    bad_channel.unlink()

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "dummy",
        ]
    )
    assert rc == 1, "missing channel for one tp must surface as a non-zero exit code"

    # The healthy timepoint should still have produced its outputs.
    assert (
        derivatives_root / "tp-week-000-1" / "seg_dl" / "tp-week-000-1_seg_brats3_dl.nii.gz"
    ).is_file()
    assert not (
        derivatives_root / "tp-week-013" / "seg_dl" / "tp-week-013_seg_brats3_dl.nii.gz"
    ).exists()


def test_segment_lumiere_timepoint_filter(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root, CHANNELS)
    _write_manifest(manifest_path, CHANNELS)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "dummy",
            "--timepoint",
            "week-013",
        ]
    )
    assert rc == 0
    assert (
        derivatives_root / "tp-week-013" / "seg_dl" / "tp-week-013_seg_brats3_dl.nii.gz"
    ).is_file()
    assert not (
        derivatives_root / "tp-week-000-1" / "seg_dl" / "tp-week-000-1_seg_brats3_dl.nii.gz"
    ).exists()


def test_unknown_timepoint_raises(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root, CHANNELS)
    _write_manifest(manifest_path, CHANNELS)

    module = _load_script_module()
    with pytest.raises(SystemExit):
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--derivatives-root",
                str(derivatives_root),
                "--backend",
                "dummy",
                "--timepoint",
                "week-999",
            ]
        )
