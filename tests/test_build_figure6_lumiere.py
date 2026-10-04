"""Smoke + summary tests for the LUMIERE Patient-048 Figure-6 builder.

Verifies that the producer:
  * accepts a tiny synthetic CSV + manifest pair and writes PDF/PNG/JSON;
  * computes the caption-cited landmark numbers (peak WT, post-op ET drop,
    recurrence ET surge) from the synthetic data correctly;
  * is robust to malformed CSVs (returns non-zero, no partial outputs).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_figure6_lumiere_p048_volumes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_figure6_lumiere_p048_volumes", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# A trajectory close in shape to the real Patient-048 numbers, but at half
# the magnitude so the test never collides with the real on-disk file.
SYNTHETIC_ROWS: tuple[dict[str, str], ...] = (
    {
        "timepoint": "week-000-1",
        "days_since_baseline": "0",
        "rano": "Pre-Op",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "70.0",
        "vol_TC_ml": "17.0",
        "vol_ET_ml": "12.0",
        "vol_NCR_ml": "5.0",
        "vol_ED_ml": "53.0",
    },
    {
        "timepoint": "week-000-2",
        "days_since_baseline": "3",
        "rano": "Post-Op",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "56.0",
        "vol_TC_ml": "22.0",
        "vol_ET_ml": "1.8",
        "vol_NCR_ml": "20.0",
        "vol_ED_ml": "34.0",
    },
    {
        "timepoint": "week-013",
        "days_since_baseline": "91",
        "rano": "SD",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "16.0",
        "vol_TC_ml": "9.0",
        "vol_ET_ml": "0.4",
        "vol_NCR_ml": "8.6",
        "vol_ED_ml": "7.0",
    },
    {
        "timepoint": "week-023",
        "days_since_baseline": "161",
        "rano": "CR",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "16.5",
        "vol_TC_ml": "8.0",
        "vol_ET_ml": "0.5",
        "vol_NCR_ml": "7.5",
        "vol_ED_ml": "8.0",
    },
    {
        "timepoint": "week-045",
        "days_since_baseline": "315",
        "rano": "PD",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "14.0",
        "vol_TC_ml": "7.5",
        "vol_ET_ml": "1.8",
        "vol_NCR_ml": "5.7",
        "vol_ED_ml": "6.2",
    },
    {
        "timepoint": "week-049",
        "days_since_baseline": "343",
        "rano": "PD",
        "voxel_volume_mm3": "1.0",
        "vol_WT_ml": "50.0",
        "vol_TC_ml": "17.5",
        "vol_ET_ml": "11.4",
        "vol_NCR_ml": "6.1",
        "vol_ED_ml": "32.5",
    },
)

CSV_FIELDS = (
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


def _write_synthetic_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in SYNTHETIC_ROWS:
            writer.writerow(row)


def _write_synthetic_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "time_axis": {
            "unit": "days_since_baseline",
            "baseline": "week-000-1",
            "caption_disclosure": "Synthetic test fixture; no real anonymisation applies.",
        },
        "timepoints": [
            {
                "name": r["timepoint"],
                "days_since_baseline": int(r["days_since_baseline"]),
                "rano": r["rano"],
                "rano_rationale": "",
            }
            for r in SYNTHETIC_ROWS
        ],
    }
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh)


def test_build_figure6_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "segmentation_summary.csv"
    manifest_path = tmp_path / "lumiere_p048_timepoints.yaml"
    out_dir = tmp_path / "outputs" / "figures"
    _write_synthetic_csv(csv_path)
    _write_synthetic_manifest(manifest_path)

    module = _load_module()
    rc = module.main(
        [
            "--summary-csv",
            str(csv_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    pdf = out_dir / "fig6_lumiere_p048_volumes.pdf"
    png = out_dir / "fig6_lumiere_p048_volumes.png"
    sidecar = out_dir / "fig6_lumiere_p048_volumes.json"
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert png.is_file() and png.stat().st_size > 0
    assert sidecar.is_file()

    payload = json.loads(sidecar.read_text())
    assert payload["figure_id"] == "fig6_lumiere_p048_volumes"
    assert payload["cohort_id"] == "lumiere_p048_synthetic"
    assert payload["patient_id"] == "048"
    summary = payload["summary"]
    assert summary["n_timepoints"] == 6
    assert summary["day_axis"] == {"min": 0.0, "max": 343.0}

    # Peak WT is at Pre-Op (70 mL); peak ET is also at Pre-Op (12 mL).
    assert summary["peaks"]["WT"]["timepoint"] == "week-000-1"
    assert summary["peaks"]["WT"]["value_ml"] == pytest.approx(70.0, rel=1e-3)
    assert summary["peaks"]["ET"]["timepoint"] == "week-000-1"
    assert summary["peaks"]["ET"]["value_ml"] == pytest.approx(12.0, rel=1e-3)

    # Post-op ET drop = 12.0 - 1.8 = 10.2 mL.
    assert summary["events"]["postop_ET_drop_ml"] == pytest.approx(10.2, rel=1e-3)

    # Recurrence trough is the lowest ET seen between Post-Op and the first
    # PD; from the synthetic table that is week-013 at 0.4 mL.
    assert summary["events"]["recurrence_ET_trough_timepoint"] == "week-013"
    assert summary["events"]["recurrence_ET_trough_ml"] == pytest.approx(0.4, rel=1e-3)
    # ET at the last PD = 11.4 mL.
    assert summary["events"]["recurrence_ET_last_PD_timepoint"] == "week-049"
    assert summary["events"]["recurrence_ET_at_last_PD_ml"] == pytest.approx(11.4, rel=1e-3)

    # The caption-disclosure sentence is propagated verbatim.
    assert "Synthetic test fixture" in payload["caption_disclosure"]


def test_manifest_overrides_csv_rano(tmp_path: Path) -> None:
    """The manifest is the single source of truth for RANO labels."""
    csv_path = tmp_path / "summary.csv"
    manifest_path = tmp_path / "manifest.yaml"
    out_dir = tmp_path / "out"

    bad_rows = [{**dict(row), "rano": "WRONG"} for row in SYNTHETIC_ROWS]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for r in bad_rows:
            writer.writerow(r)
    _write_synthetic_manifest(manifest_path)

    module = _load_module()
    assert (
        module.main(
            [
                "--summary-csv",
                str(csv_path),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    sidecar = out_dir / "fig6_lumiere_p048_volumes.json"
    payload = json.loads(sidecar.read_text())
    rano_seen = [r["rano"] for r in payload["table"]]
    assert "WRONG" not in rano_seen
    assert rano_seen[0] == "Pre-Op"
    assert rano_seen[-1] == "PD"


def test_missing_csv_returns_nonzero(tmp_path: Path) -> None:
    manifest_path = tmp_path / "m.yaml"
    _write_synthetic_manifest(manifest_path)
    module = _load_module()
    rc = module.main(
        [
            "--summary-csv",
            str(tmp_path / "does-not-exist.csv"),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "out" / "fig6_lumiere_p048_volumes.pdf").exists()


def test_csv_missing_required_column_is_classified_as_malformed(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    manifest_path = tmp_path / "m.yaml"
    out_dir = tmp_path / "out"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timepoint", "days_since_baseline", "rano"])
        writer.writeheader()
        writer.writerow({"timepoint": "week-000-1", "days_since_baseline": "0", "rano": "Pre-Op"})
    _write_synthetic_manifest(manifest_path)
    module = _load_module()
    rc = module.main(
        [
            "--summary-csv",
            str(csv_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 2
    assert not (out_dir / "fig6_lumiere_p048_volumes.pdf").exists()


def test_empty_csv_returns_malformed(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    manifest_path = tmp_path / "m.yaml"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()  # header only, zero data rows
    _write_synthetic_manifest(manifest_path)
    module = _load_module()
    rc = module.main(
        [
            "--summary-csv",
            str(csv_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
