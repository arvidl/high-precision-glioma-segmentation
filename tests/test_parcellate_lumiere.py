"""Integration smoke test for the LUMIERE Patient-048 parcellation runner.

Uses the deterministic ``dummy`` backend of ``hpgs.parcellate.predict_wmparc``
so the test does not need FreeSurfer 8.2.0 on the host. The point is to
exercise the script's *file-IO contract* end-to-end (manifest parsing,
per-tp ``parcellation/`` tree, LUT JSON shape, sidecar shape, cohort summary
columns, RAS-grid alignment with the seed channel), not the anatomical
correctness of the dummy field.
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "parcellate_lumiere_p048.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("parcellate_lumiere_p048", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _build_synthetic_derivatives(derivatives_root: Path) -> None:
    """Two timepoints with just the T1 seed channel parcellation needs."""
    shape = (24, 24, 24)
    affine = np.eye(4)
    for tp_name in ("week-000-1", "week-013"):
        # Only T1_to_ref is required for parcellation.
        _save_nifti(
            np.zeros(shape, dtype=np.float32),
            affine,
            derivatives_root / f"tp-{tp_name}" / f"tp-{tp_name}_T1_to_ref.nii.gz",
        )


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


def test_parcellate_lumiere_smoke_dummy_backend(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "dummy",
            "--no-dry-run",
        ]
    )
    assert rc == 0

    # Per-tp artefacts.
    for tp in ("week-000-1", "week-013"):
        parc_dir = derivatives_root / f"tp-{tp}" / "parcellation"
        wmparc = parc_dir / f"tp-{tp}_wmparc_native.nii.gz"
        lut = parc_dir / f"tp-{tp}_wmparc_lut.json"
        sidecar_path = parc_dir / f"tp-{tp}_wmparc.json"
        assert wmparc.is_file()
        assert lut.is_file()
        assert sidecar_path.is_file()

        wmparc_img = nib.load(str(wmparc))
        wm = np.asarray(wmparc_img.dataobj)
        assert wm.dtype == np.int16
        assert wm.shape == (24, 24, 24)
        # Dummy backend writes the canonical 9-region FreeSurfer subset; at
        # minimum the cortex + WM + brain-stem labels should be present.
        labels = set(int(v) for v in np.unique(wm))
        for required in (0, 2, 41, 16):
            assert required in labels, f"missing FS label {required} in {labels}"

        # Grid alignment: wmparc_native must sit on the seed-channel grid.
        seed = nib.load(str(derivatives_root / f"tp-{tp}" / f"tp-{tp}_T1_to_ref.nii.gz"))
        assert wmparc_img.shape == seed.shape
        np.testing.assert_allclose(wmparc_img.affine, seed.affine, atol=1e-6)

        sc = json.loads(sidecar_path.read_text())
        assert sc["timepoint"] == tp
        assert sc["patient_id"] == "048"
        assert sc["subject_id"] == f"p048_{tp}"
        assert sc["backend"] == "dummy"
        assert sc["input_channel"] == "T1_to_ref"
        assert sc["schema_version"] == "1.0"
        assert sc["n_labels_present"] == len(json.loads(lut.read_text()))
        # LUT JSON keys are stringified ints.
        for key in json.loads(lut.read_text()):
            int(key)  # raises ValueError if not int-stringable

    # Cohort summary CSV + JSON.
    csv_path = derivatives_root / "parcellation_summary.csv"
    json_path = derivatives_root / "parcellation_summary.json"
    assert csv_path.is_file()
    assert json_path.is_file()
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["timepoint"] for r in rows] == ["week-000-1", "week-013"]
    assert rows[0]["rano"] == "Pre-Op"
    assert rows[1]["rano"] == "SD"
    for r in rows:
        assert r["backend"] == "dummy"
        assert int(r["n_labels_present"]) >= 8

    payload = json.loads(json_path.read_text())
    assert payload["cohort_id"] == "lumiere_p048_synthetic"
    assert payload["patient_id"] == "048"
    assert payload["backend"] == "dummy"
    assert len(payload["rows"]) == 2


def test_parcellate_lumiere_skip_existing_is_idempotent(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    args = [
        "--manifest",
        str(manifest_path),
        "--derivatives-root",
        str(derivatives_root),
        "--backend",
        "dummy",
        "--no-dry-run",
    ]
    assert module.main(args) == 0
    wmparc = derivatives_root / "tp-week-013" / "parcellation" / "tp-week-013_wmparc_native.nii.gz"
    first_mtime = wmparc.stat().st_mtime
    first_csv = (derivatives_root / "parcellation_summary.csv").read_text()

    assert module.main([*args, "--skip-existing"]) == 0
    second_mtime = wmparc.stat().st_mtime
    second_csv = (derivatives_root / "parcellation_summary.csv").read_text()

    assert first_csv == second_csv
    assert (
        first_mtime == second_mtime
    ), "with --skip-existing the wmparc should not have been rewritten"


def test_parcellate_lumiere_missing_seed_is_reported_not_raised(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root)
    _write_manifest(manifest_path)

    bad_seed = derivatives_root / "tp-week-013" / "tp-week-013_T1_to_ref.nii.gz"
    bad_seed.unlink()

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "dummy",
            "--no-dry-run",
        ]
    )
    # Real run with one missing seed: returns 1, but does not raise.
    assert rc == 1
    summary = derivatives_root / "parcellation_summary.csv"
    assert summary.is_file()
    with summary.open() as fh:
        rows = list(csv.DictReader(fh))
    # week-013 is missing its seed -> only week-000-1 lands in the summary.
    assert [r["timepoint"] for r in rows] == ["week-000-1"]


def test_parcellate_lumiere_dry_run_default(tmp_path: Path, capsys) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root)
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
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "WOULD WRITE" in out
    # Default is --backend freesurfer in the dry-run banner.
    assert "backend           = freesurfer" in out
    # Crucially: no parcellation/ directories should have been created.
    assert not (derivatives_root / "tp-week-000-1" / "parcellation").exists()
    assert not (derivatives_root / "tp-week-013" / "parcellation").exists()


def test_parcellate_lumiere_freesurfer_requires_fs_work_dir(tmp_path: Path) -> None:
    derivatives_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_derivatives(derivatives_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--derivatives-root",
            str(derivatives_root),
            "--backend",
            "freesurfer",
            "--no-dry-run",
        ]
    )
    # Real freesurfer fan-out without a scratch SUBJECTS_DIR is a config error.
    assert rc == 2
