"""Integration smoke test for the LUMIERE Patient-048 registration orchestrator.

Builds a tiny synthetic stage tree (2 timepoints x 4 channels, 16^3 voxels)
matching the layout produced by ``scripts/extract_lumiere_p048.py``, runs
``register_subject_longitudinal``, and verifies that:

* Every expected output file is written.
* The reference timepoint's outputs share a voxel grid with the reference T1.
* The non-reference timepoint's outputs share a voxel grid with the reference T1.
* ``registration_qc.json`` and ``manifest.json`` exist and contain the expected keys.
* A second run without ``overwrite=True`` is idempotent (status keys flip to ``skipped_existing``).

We do NOT test registration accuracy here -- that is the job of
``tests/test_register_core.py`` on synthetic Gaussian-blob volumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from hpgs.register.lumiere import register_subject_longitudinal

CHANNELS = ["T1", "CT1", "T2", "FLAIR"]


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _gaussian_blob(shape: tuple[int, int, int], centre: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float32)
    cz, cy, cx = centre
    r2 = (grid[0] - cz) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cx) ** 2
    return np.exp(-r2 / (2 * 3.0**2)).astype(np.float32)


def _build_synthetic_stage(stage_root: Path) -> None:
    """Two timepoints, four channels each, all 16^3 RAS NIfTIs."""
    shape = (16, 16, 16)
    affine = np.eye(4)
    ref_centre = (8, 8, 8)
    moving_centre = (8 + 1, 8, 8 - 1)

    for ch in CHANNELS:
        _save_nifti(
            _gaussian_blob(shape, ref_centre),
            affine,
            stage_root / "tp-week-000-1" / f"tp-week-000-1_{ch}.nii.gz",
        )
        _save_nifti(
            _gaussian_blob(shape, moving_centre),
            affine,
            stage_root / "tp-week-013" / f"tp-week-013_{ch}.nii.gz",
        )


def _write_manifest(path: Path) -> None:
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "reference_timepoint": "week-000-1",
        "channels": CHANNELS,
        "timepoints": [
            {"name": "week-000-1", "days_since_baseline": 0, "rano": "Pre-Op"},
            {"name": "week-013", "days_since_baseline": 91, "rano": "SD"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh)


def test_register_subject_longitudinal_smoke(tmp_path: Path) -> None:
    pytest.importorskip("ants")
    stage_root = tmp_path / "lumiere_p048"
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"

    _build_synthetic_stage(stage_root)
    _write_manifest(manifest_path)

    provenance = register_subject_longitudinal(
        manifest_path=manifest_path,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
    )

    # Provenance shape.
    assert provenance["reference_timepoint"] == "week-000-1"
    assert provenance["channels"] == CHANNELS
    assert (derivatives_root / "manifest.json").is_file()
    assert len(provenance["timepoints"]) == 2

    ref_record = next(tp for tp in provenance["timepoints"] if tp["is_reference"])
    moving_record = next(tp for tp in provenance["timepoints"] if not tp["is_reference"])

    assert ref_record["name"] == "week-000-1"
    assert ref_record["moving_channel"] is None
    assert moving_record["name"] == "week-013"
    # Override map only kicks in for week-023; here we should default to T1.
    assert moving_record["moving_channel"] == "T1"
    assert moving_record["transform_paths"], "expected at least one ANTs transform file"
    assert all(Path(p).is_file() for p in moving_record["transform_paths"])

    # Per-timepoint outputs and QC files.
    for tp_name in ("week-000-1", "week-013"):
        tp_dir = derivatives_root / f"tp-{tp_name}"
        for ch in CHANNELS:
            out_file = tp_dir / f"tp-{tp_name}_{ch}_to_ref.nii.gz"
            assert out_file.is_file(), f"missing output: {out_file}"
        assert (tp_dir / "registration_qc.json").is_file()

    # Grid match: every output must have the same shape/zooms/axcodes as the
    # reference T1 (which is 16^3, identity-affine -> RAS).
    ref_shape = (16, 16, 16)
    for tp_name in ("week-000-1", "week-013"):
        for ch in CHANNELS:
            img = nib.load(
                str(derivatives_root / f"tp-{tp_name}" / f"tp-{tp_name}_{ch}_to_ref.nii.gz")
            )
            assert img.shape[:3] == ref_shape
            assert "".join(nib.aff2axcodes(img.affine)) == "RAS"

    # registration_qc.json content for the moving timepoint.
    qc = json.loads((derivatives_root / "tp-week-013" / "registration_qc.json").read_text())
    assert qc["moving_channel"] == "T1"
    assert qc["qc"]["channels"]["T1"]["grid_matches_reference"] is True
    assert qc["qc"]["channels"]["FLAIR"]["grid_matches_reference"] is True
    assert isinstance(qc["qc"]["moving_to_ref_correlation"], float)


def test_register_subject_longitudinal_is_idempotent(tmp_path: Path) -> None:
    pytest.importorskip("ants")
    stage_root = tmp_path / "lumiere_p048"
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"

    _build_synthetic_stage(stage_root)
    _write_manifest(manifest_path)

    register_subject_longitudinal(
        manifest_path=manifest_path,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
    )
    provenance = register_subject_longitudinal(
        manifest_path=manifest_path,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
    )
    moving_record = next(tp for tp in provenance["timepoints"] if not tp["is_reference"])
    for ch in CHANNELS:
        assert moving_record["qc"]["channels"][ch]["status"] == "skipped_existing"


def test_week_023_default_override_uses_ct1(tmp_path: Path) -> None:
    """When week-023 is in the manifest, the default override uses CT1 as moving."""
    pytest.importorskip("ants")
    stage_root = tmp_path / "lumiere_p048"
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"

    shape = (16, 16, 16)
    affine = np.eye(4)
    ref_centre = (8, 8, 8)
    moving_centre = (9, 8, 7)
    for ch in CHANNELS:
        _save_nifti(
            _gaussian_blob(shape, ref_centre),
            affine,
            stage_root / "tp-week-000-1" / f"tp-week-000-1_{ch}.nii.gz",
        )
        _save_nifti(
            _gaussian_blob(shape, moving_centre),
            affine,
            stage_root / "tp-week-023" / f"tp-week-023_{ch}.nii.gz",
        )

    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "reference_timepoint": "week-000-1",
        "channels": CHANNELS,
        "timepoints": [
            {"name": "week-000-1", "days_since_baseline": 0, "rano": "Pre-Op"},
            {"name": "week-023", "days_since_baseline": 161, "rano": "CR"},
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        yaml.safe_dump(manifest, fh)

    provenance = register_subject_longitudinal(
        manifest_path=manifest_path,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
    )
    week_023 = next(tp for tp in provenance["timepoints"] if tp["name"] == "week-023")
    assert week_023["moving_channel"] == "CT1"
