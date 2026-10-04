"""Smoke tests for the Figure 11 (LUMIERE Patient-048 longitudinal Hit-Plot) builder.

Builds a synthetic per-tp Hit-Plot tree (six timepoints, three regions per
compartment) and runs the figure builder end-to-end. Verifies file-IO contract
(PDF/PNG/JSON outputs), JSON sidecar shape (region order, matrix shape,
per-tp totals), and that the "Unknown" parcel is correctly excluded from
the heatmap.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_figure11_lumiere_p048_hitplot.py"

_TPS = (
    ("week-000-1", 0, "Pre-Op"),
    ("week-000-2", 3, "Post-Op"),
    ("week-013", 91, "SD"),
    ("week-023", 161, "CR"),
    ("week-045", 315, "PD"),
    ("week-049", 343, "PD"),
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_figure11_lumiere_p048_hitplot", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path) -> None:
    manifest = {
        "schema_version": "1.0",
        "cohort_id": "lumiere_p048_synthetic",
        "patient_id": "048",
        "reference_timepoint": "week-000-1",
        "channels": ["T1", "CT1", "T2", "FLAIR"],
        "timepoints": [
            {"name": tp, "days_since_baseline": days, "rano": rano} for tp, days, rano in _TPS
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(manifest, fh)


# Per-tp WT volumes (mm^3) for three named regions plus the Unknown bucket.
# Trajectory mimics the round-3 real-data shape (Pre-Op high, Post-Op drop,
# stable mid, recurrence at the end).
_WT_TRAJ = {
    "Right-Frontal-WM": (15000, 8000, 2000, 2000, 3000, 9000),
    "Right-Frontal-Cortex": (10000, 6000, 1000, 1000, 1500, 6000),
    "Right-Cerebellum-Cortex": (0, 0, 4000, 4000, 0, 0),
    "Unknown": (35000, 20000, 10000, 10000, 12000, 25000),
}
# Per-tp ET volumes -- same regions, smaller magnitudes, stronger PD recurrence.
_ET_TRAJ = {
    "Right-Frontal-WM": (1500, 200, 0, 0, 200, 2500),
    "Right-Frontal-Cortex": (1000, 100, 0, 0, 100, 1500),
    "Right-Cerebellum-Cortex": (0, 0, 0, 0, 0, 0),
    "Unknown": (4000, 100, 0, 0, 200, 1000),
}


def _write_csv(path: Path, tp_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    label_map = {
        "Right-Frontal-WM": 41,
        "Right-Frontal-Cortex": 42,
        "Right-Cerebellum-Cortex": 47,
        "Unknown": 0,
    }
    for compartment, traj in (("WT", _WT_TRAJ), ("ET", _ET_TRAJ)):
        for region, vols in traj.items():
            v = vols[tp_index]
            if v == 0 and compartment == "ET":
                continue
            rows.append(
                {
                    "label": label_map[region],
                    "name": region,
                    "compartment": compartment,
                    "parcel_voxels": 1000 if region != "Unknown" else 100000,
                    "overlap_voxels": v,
                    "overlap_volume_mm3": float(v),
                    "pct_of_parcel": 0.0,
                    "pct_of_compartment": 0.0,
                }
            )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "name",
                "compartment",
                "parcel_voxels",
                "overlap_voxels",
                "overlap_volume_mm3",
                "pct_of_parcel",
                "pct_of_compartment",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_synthetic_hitplot_tree(hitplot_root: Path) -> None:
    for i, (tp, _, _) in enumerate(_TPS):
        _write_csv(hitplot_root / f"tp-{tp}" / "hitplot" / f"tp-{tp}_hitplot_dl.csv", i)


def test_build_figure11_smoke(tmp_path: Path) -> None:
    hitplot_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    out_dir = tmp_path / "out_figs"
    _build_synthetic_hitplot_tree(hitplot_root)
    _write_manifest(manifest_path)

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--hitplot-root",
            str(hitplot_root),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "3",
        ]
    )
    assert rc == 0, "builder must succeed on a clean synthetic input"

    pdf = out_dir / "fig11_lumiere_p048_hitplot.pdf"
    png = out_dir / "fig11_lumiere_p048_hitplot.png"
    json_path = out_dir / "fig11_lumiere_p048_hitplot.json"
    assert pdf.is_file() and pdf.stat().st_size > 1024
    assert png.is_file() and png.stat().st_size > 1024
    assert json_path.is_file()

    payload = json.loads(json_path.read_text())
    assert payload["figure_id"] == "fig11_lumiere_p048_hitplot"
    assert payload["cohort_id"] == "lumiere_p048_synthetic"
    assert payload["patient_id"] == "048"
    assert payload["top_k"] == 3
    assert "caption_disclosure" in payload

    panels = payload["panels"]
    assert set(panels) == {"a_wt", "b_et"}

    wt = panels["a_wt"]
    # Unknown must be excluded; we have only 3 named WT regions in the synth.
    assert wt["compartment"] == "WT"
    assert "Unknown" not in wt["regions_top_to_bottom"]
    assert wt["n_regions"] == 3
    assert len(wt["matrix_overlap_ml"]) == 3
    for row in wt["matrix_overlap_ml"]:
        assert len(row) == len(_TPS)  # 6 columns
    # Right-Frontal-WM must come first (highest max across tp).
    assert wt["regions_top_to_bottom"][0] == "Right-Frontal-WM"
    # Per-tp WT total (excluding Unknown) must match the synthetic sum.
    expected_wt_pre = (15000 + 10000 + 0) / 1000.0  # = 25.0 mL
    assert abs(wt["per_tp_total_ml_excl_unknown"]["week-000-1"] - expected_wt_pre) < 1e-6

    et = panels["b_et"]
    assert et["compartment"] == "ET"
    assert "Unknown" not in et["regions_top_to_bottom"]
    # Right-Cerebellum-Cortex has zero ET in every tp -> it should NOT appear
    # in the ET top-3 (the matrix builder ranks by max overlap; zeros lose).
    assert "Right-Cerebellum-Cortex" not in et["regions_top_to_bottom"]
    # Per-tp ET totals (excl Unknown) should be (1500+1000)/1000=2.5 at Pre-Op.
    assert abs(et["per_tp_total_ml_excl_unknown"]["week-000-1"] - 2.5) < 1e-6

    # tp_order_left_to_right is the manifest order.
    assert wt["tp_order_left_to_right"] == [tp for tp, _, _ in _TPS]


def test_build_figure11_missing_csv_returns_1(tmp_path: Path) -> None:
    """Missing per-tp CSV must produce a clean exit code 1, not a traceback."""
    hitplot_root = tmp_path / "derivatives"
    manifest_path = tmp_path / "configs" / "lumiere_p048_timepoints.yaml"
    _build_synthetic_hitplot_tree(hitplot_root)
    _write_manifest(manifest_path)
    # Nuke one CSV.
    target = hitplot_root / "tp-week-013" / "hitplot" / "tp-week-013_hitplot_dl.csv"
    assert target.is_file()
    target.unlink()

    module = _load_script_module()
    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--hitplot-root",
            str(hitplot_root),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
