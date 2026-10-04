"""Tests for ``scripts/build_table_tumorsynth_agreement.py``.

The TumorSynth table is a standalone optional-comparator producer. These tests
exercise its path contract, comparison validation, aggregation, renderers, and
CLI orchestration on synthetic sidecars only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "build_table_tumorsynth_agreement.py"
    spec = importlib.util.spec_from_file_location("build_table_tumorsynth_agreement", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


def _make_sidecar(
    subject_id: str,
    *,
    comparison: str = "dl_vs_tumorsynth",
    wt_dice: float = 0.90,
    tc_dice: float = 0.80,
    et_dice: float = 0.70,
    wt_hd95: float = 4.0,
    tc_hd95: float = 5.0,
    et_hd95: float = 6.0,
    wt_ve: float = 1000.0,
    tc_ve: float = 800.0,
    et_ve: float = 600.0,
    wt_sens: float = 0.92,
    tc_sens: float = 0.88,
    et_sens: float = 0.84,
    wt_spec: float = 0.999,
    tc_spec: float = 0.998,
    et_spec: float = 0.997,
) -> dict:
    return {
        "schema_version": "1.0",
        "subject_id": subject_id,
        "comparison": comparison,
        "voxel_volume_mm3": 1.0,
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "metrics": {
            "WT": {
                "dice": wt_dice,
                "hd95_mm": wt_hd95,
                "abs_volumetric_error_mm3": wt_ve,
                "sensitivity": wt_sens,
                "specificity": wt_spec,
            },
            "TC": {
                "dice": tc_dice,
                "hd95_mm": tc_hd95,
                "abs_volumetric_error_mm3": tc_ve,
                "sensitivity": tc_sens,
                "specificity": tc_spec,
            },
            "ET": {
                "dice": et_dice,
                "hd95_mm": et_hd95,
                "abs_volumetric_error_mm3": et_ve,
                "sensitivity": et_sens,
                "specificity": et_spec,
            },
        },
    }


def _write_sidecar_to_disk(metrics_root: Path, doc: dict) -> Path:
    path = mod._metrics_path(metrics_root, doc["subject_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def test_metrics_path_targets_tumorsynth_sidecar(tmp_path: Path) -> None:
    path = mod._metrics_path(tmp_path, "sub-0005")
    assert path.name == "sub-0005_metrics_dl_vs_tumorsynth.json"
    assert path.parent.name == "metrics"
    assert path.parent.parent.name == "sub-0005"


def test_validate_accepts_well_formed_tumorsynth_sidecar() -> None:
    mod._validate_sidecar(_make_sidecar("0005"))


def test_validate_rejects_wrong_comparison() -> None:
    with pytest.raises(mod.SidecarError, match="comparison"):
        mod._validate_sidecar(_make_sidecar("0005", comparison="dl_vs_gt"))


def test_load_sidecars_happy_path(tmp_path: Path) -> None:
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0005", wt_dice=0.8))
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0026", wt_dice=0.9))
    records, problems = mod.load_metrics_sidecars(tmp_path, ["0005", "0026"])
    assert problems == []
    assert [r["subject_id"] for r in records] == ["0005", "0026"]


def test_load_sidecars_records_missing_files(tmp_path: Path) -> None:
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0005"))
    records, problems = mod.load_metrics_sidecars(tmp_path, ["0005", "0068"])
    assert [r["subject_id"] for r in records] == ["0005"]
    assert len(problems) == 1
    assert problems[0][0] == "0068"
    assert "not found" in problems[0][1]


def test_load_sidecars_strict_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_metrics_sidecars(tmp_path, ["0005"], strict=True)


def test_summarise_known_tumorsynth_values() -> None:
    records = [
        _make_sidecar("0001", wt_dice=0.50),
        _make_sidecar("0002", wt_dice=0.60),
        _make_sidecar("0003", wt_dice=0.70),
    ]
    summary = mod.summarise_per_cell(records)
    cell = summary["WT"]["dice"]
    assert cell["median"] == pytest.approx(0.60)
    assert cell["n_used"] == 3


def test_long_audit_frame_includes_comparison_column() -> None:
    df = mod.long_audit_frame([_make_sidecar("0005")])
    assert set(df["comparison"]) == {"dl_vs_tumorsynth"}
    assert len(df) == len(mod.COMPARTMENTS) * len(mod._METRIC_KEYS)


def test_format_table_tex_mentions_tumorsynth_source() -> None:
    summary = mod.summarise_per_cell([_make_sidecar("0005", wt_dice=0.841)])
    tex = mod.format_table_tex(summary, n_total=50, n_used_cohort=1)
    assert "build_table_tumorsynth_agreement.py" in tex
    assert "sub-XXXX_metrics_dl_vs_tumorsynth.json" in tex
    assert "Dice & 0.841" in tex
    assert "n=1/1" not in tex


def test_main_writes_outputs_with_synthetic_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_root = tmp_path / "derivatives"
    out_tex = tmp_path / "table.tex"
    out_csv = tmp_path / "table.csv"
    out_json = tmp_path / "table.json"
    _write_sidecar_to_disk(metrics_root, _make_sidecar("0005", wt_dice=0.8))
    _write_sidecar_to_disk(metrics_root, _make_sidecar("0026", wt_dice=0.9))

    class _FakeFrame:
        def __init__(self) -> None:
            self.index = ["0005", "0026"]

    class _FakeCohort:
        n = 2
        df = _FakeFrame()

    monkeypatch.setattr(mod, "load_cohort_metadata", lambda *_args: _FakeCohort())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table_tumorsynth_agreement.py",
            "--cohort-yaml",
            str(tmp_path / "cohort.yaml"),
            "--metadata-csv",
            str(tmp_path / "metadata.csv"),
            "--metrics-root",
            str(metrics_root),
            "--out-tex",
            str(out_tex),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--strict",
        ],
    )

    assert mod.main() == 0
    assert out_tex.is_file()
    assert out_csv.is_file()
    payload = json.loads(out_json.read_text())
    assert payload["comparison"] == "dl_vs_tumorsynth"
    assert payload["n_with_sidecar"] == 2
    assert payload["summary"]["WT"]["dice"]["median"] == pytest.approx(0.85)
