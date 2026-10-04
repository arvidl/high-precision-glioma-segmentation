"""Unit tests for ``scripts/build_table2_dl_vs_gt.py`` (PR-7g).

We import the script as a module via ``importlib`` (the ``scripts/``
directory is intentionally not on ``sys.path``; the script is a CLI
shim). Helpers, schema validation, NaN/Inf-safe aggregation, and the
LaTeX/CSV/JSON outputs are covered here on synthetic fixtures so the
test does not depend on torch / monai / FreeSurfer / network access
or on real PR-7d sidecars on disk.

``paper/main.tex`` (R2.2) and for the cross-cell agreement
summary that PR-7h will derive on top (R2.5 / R2.10).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_table2_dl_vs_gt.py"
    spec = importlib.util.spec_from_file_location("build_table2_dl_vs_gt", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# Synthetic sidecar fixtures
# ---------------------------------------------------------------------------


def _make_sidecar(
    subject_id: str,
    *,
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
    """Build a fully-populated PR-7d-shaped sidecar dict."""
    return {
        "schema_version": "1.0",
        "subject_id": subject_id,
        "comparison": "dl_vs_gt",
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
    sid = doc["subject_id"]
    p = mod._metrics_path(metrics_root, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


# ---------------------------------------------------------------------------
# _classify_value: NaN / Inf / wrong type / ok
# ---------------------------------------------------------------------------


def test_classify_finite_number() -> None:
    assert mod._classify_value(0.5) == ("ok", 0.5)
    assert mod._classify_value(0) == ("ok", 0.0)
    assert mod._classify_value(-3.14) == ("ok", -3.14)


def test_classify_json_null_is_missing() -> None:
    assert mod._classify_value(None) == ("missing", None)


def test_classify_string_inf_is_inf() -> None:
    assert mod._classify_value("inf") == ("inf", None)
    assert mod._classify_value("+inf") == ("inf", None)
    assert mod._classify_value("-inf") == ("inf", None)


def test_classify_python_inf_is_inf() -> None:
    assert mod._classify_value(float("inf")) == ("inf", None)
    assert mod._classify_value(-float("inf")) == ("inf", None)


def test_classify_python_nan_is_missing() -> None:
    assert mod._classify_value(float("nan")) == ("missing", None)


def test_classify_unknown_string_is_missing() -> None:
    assert mod._classify_value("oops") == ("missing", None)


def test_classify_bool_is_missing() -> None:
    # Guard against pandas/json quirk where True/False can sneak into
    # numeric fields; bool is a subclass of int but is not a metric value.
    assert mod._classify_value(True) == ("missing", None)
    assert mod._classify_value(False) == ("missing", None)


# ---------------------------------------------------------------------------
# _validate_sidecar
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed_sidecar() -> None:
    mod._validate_sidecar(_make_sidecar("0005"))


def test_validate_rejects_unknown_schema_version() -> None:
    bad = _make_sidecar("0005")
    bad["schema_version"] = "9.9"
    with pytest.raises(mod.SidecarError, match="schema_version"):
        mod._validate_sidecar(bad)


def test_validate_rejects_wrong_comparison() -> None:
    bad = _make_sidecar("0005")
    bad["comparison"] = "dl_vs_raidionics"
    with pytest.raises(mod.SidecarError, match="comparison"):
        mod._validate_sidecar(bad)


def test_validate_rejects_missing_compartment() -> None:
    bad = _make_sidecar("0005")
    del bad["metrics"]["ET"]
    with pytest.raises(mod.SidecarError, match="ET"):
        mod._validate_sidecar(bad)


def test_validate_rejects_missing_metric() -> None:
    bad = _make_sidecar("0005")
    del bad["metrics"]["WT"]["hd95_mm"]
    with pytest.raises(mod.SidecarError, match="hd95_mm"):
        mod._validate_sidecar(bad)


def test_validate_tolerates_extra_keys() -> None:
    """Forward-compat: extra unknown keys must not break the loader."""
    extra = _make_sidecar("0005")
    extra["future_key"] = "ignored"
    extra["metrics"]["WT"]["future_metric"] = 0.5
    mod._validate_sidecar(extra)  # must not raise


# ---------------------------------------------------------------------------
# load_metrics_sidecars
# ---------------------------------------------------------------------------


def test_load_sidecars_happy_path(tmp_path: Path) -> None:
    for sid, dice in (("0005", 0.9), ("0026", 0.85), ("0068", 0.88)):
        _write_sidecar_to_disk(tmp_path, _make_sidecar(sid, wt_dice=dice))
    records, problems = mod.load_metrics_sidecars(tmp_path, ["0005", "0026", "0068"])
    assert problems == []
    assert [r["subject_id"] for r in records] == ["0005", "0026", "0068"]


def test_load_sidecars_normalises_subject_prefix(tmp_path: Path) -> None:
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0005"))
    records, problems = mod.load_metrics_sidecars(tmp_path, ["sub-0005"])
    assert problems == []
    assert records[0]["subject_id"] == "0005"


def test_load_sidecars_records_missing_files(tmp_path: Path) -> None:
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0005"))
    records, problems = mod.load_metrics_sidecars(tmp_path, ["0005", "0026"])
    assert [r["subject_id"] for r in records] == ["0005"]
    assert len(problems) == 1
    assert problems[0][0] == "0026"
    assert "not found" in problems[0][1]


def test_load_sidecars_records_malformed_json(tmp_path: Path) -> None:
    _write_sidecar_to_disk(tmp_path, _make_sidecar("0005"))
    bad_path = mod._metrics_path(tmp_path, "0026")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not json")
    records, problems = mod.load_metrics_sidecars(tmp_path, ["0005", "0026"])
    assert [r["subject_id"] for r in records] == ["0005"]
    assert problems[0][0] == "0026"
    assert "JSON" in problems[0][1]


def test_load_sidecars_strict_raises_on_first_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_metrics_sidecars(tmp_path, ["0005"], strict=True)


def test_load_sidecars_strict_raises_on_schema_violation(tmp_path: Path) -> None:
    bad = _make_sidecar("0005")
    bad["schema_version"] = "9.9"
    _write_sidecar_to_disk(tmp_path, bad)
    with pytest.raises(mod.SidecarError):
        mod.load_metrics_sidecars(tmp_path, ["0005"], strict=True)


# ---------------------------------------------------------------------------
# summarise_per_cell
# ---------------------------------------------------------------------------


def test_summarise_known_dice_values() -> None:
    """Hand-checked median/Q1/Q3 against a small known sample."""
    records = [
        _make_sidecar("0001", wt_dice=0.50),
        _make_sidecar("0002", wt_dice=0.60),
        _make_sidecar("0003", wt_dice=0.70),
        _make_sidecar("0004", wt_dice=0.80),
        _make_sidecar("0005", wt_dice=0.90),
    ]
    summary = mod.summarise_per_cell(records)
    cell = summary["WT"]["dice"]
    assert cell["median"] == pytest.approx(0.70)
    assert cell["q1"] == pytest.approx(0.60)  # numpy's linear interpolation
    assert cell["q3"] == pytest.approx(0.80)
    assert cell["n_used"] == 5
    assert cell["n_missing"] == 0
    assert cell["n_inf"] == 0


def test_summarise_excludes_null_and_inf() -> None:
    """null -> n_missing; "inf" -> n_inf; both excluded from the median."""
    records = [
        _make_sidecar("0001", wt_dice=0.5),
        _make_sidecar("0002", wt_dice=0.7),
        _make_sidecar("0003", wt_dice=0.9),
    ]
    # Replace HD95 of WT with one null and one "inf"; one finite.
    records[0]["metrics"]["WT"]["hd95_mm"] = None
    records[1]["metrics"]["WT"]["hd95_mm"] = "inf"
    records[2]["metrics"]["WT"]["hd95_mm"] = 5.0
    summary = mod.summarise_per_cell(records)
    cell = summary["WT"]["hd95_mm"]
    assert cell["median"] == pytest.approx(5.0)
    assert cell["n_used"] == 1
    assert cell["n_missing"] == 1
    assert cell["n_inf"] == 1
    # The dice cell is unaffected and should still see all 3 samples.
    assert summary["WT"]["dice"]["n_used"] == 3


def test_summarise_all_missing_returns_none(tmp_path: Path) -> None:
    """Cell with zero finite samples: median/Q1/Q3 all None, JSON-safe."""
    records = [_make_sidecar("0001"), _make_sidecar("0002")]
    for rec in records:
        rec["metrics"]["ET"]["sensitivity"] = None
    summary = mod.summarise_per_cell(records)
    cell = summary["ET"]["sensitivity"]
    assert cell["median"] is None
    assert cell["q1"] is None
    assert cell["q3"] is None
    assert cell["n_used"] == 0
    assert cell["n_missing"] == 2
    # The whole summary must be JSON-serialisable as-is.
    json.dumps(summary)


def test_summarise_shape_matches_compartments_and_metrics() -> None:
    summary = mod.summarise_per_cell([_make_sidecar("0001")])
    assert set(summary) == set(mod.COMPARTMENTS)
    for comp in mod.COMPARTMENTS:
        assert set(summary[comp]) == set(mod._METRIC_KEYS)


# ---------------------------------------------------------------------------
# long_audit_frame
# ---------------------------------------------------------------------------


def test_audit_frame_columns_and_row_count() -> None:
    records = [_make_sidecar("0001"), _make_sidecar("0002")]
    df = mod.long_audit_frame(records)
    assert list(df.columns) == [
        "subject_id",
        "compartment",
        "metric",
        "value",
        "status",
        "raw",
    ]
    # Two subjects * 3 compartments * 5 metrics = 30 rows.
    assert len(df) == 2 * 3 * 5
    # No status other than 'ok' on a clean fixture.
    assert (df["status"] == "ok").all()


def test_audit_frame_marks_status_for_null_and_inf() -> None:
    records = [_make_sidecar("0001")]
    records[0]["metrics"]["WT"]["dice"] = None
    records[0]["metrics"]["WT"]["hd95_mm"] = "inf"
    df = mod.long_audit_frame(records)
    wt_dice_status = df.query("compartment == 'WT' and metric == 'dice'")["status"].iloc[0]
    wt_hd95_status = df.query("compartment == 'WT' and metric == 'hd95_mm'")["status"].iloc[0]
    assert wt_dice_status == "missing"
    assert wt_hd95_status == "inf"
    # The numeric value column is NaN for non-ok cells.
    wt_dice_value = df.query("compartment == 'WT' and metric == 'dice'")["value"].iloc[0]
    assert np.isnan(wt_dice_value)


# ---------------------------------------------------------------------------
# format_table_tex
# ---------------------------------------------------------------------------


def test_table_tex_contains_required_structure() -> None:
    summary = mod.summarise_per_cell([_make_sidecar(f"{i:04d}") for i in range(1, 6)])
    tex = mod.format_table_tex(summary, n_total=50, n_used_cohort=5)
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    # All five metric labels appear.
    for _, label, _ in mod.METRICS:
        assert label in tex
    # All three compartment headers appear.
    for comp in mod.COMPARTMENTS:
        assert comp in tex


def test_table_tex_renders_median_iqr_per_cell() -> None:
    """Hand-checked: 5 identical Dice=0.80 -> '0.800 [0.800, 0.800]'."""
    records = [_make_sidecar(f"{i:04d}", wt_dice=0.80) for i in range(1, 6)]
    summary = mod.summarise_per_cell(records)
    tex = mod.format_table_tex(summary, n_total=5, n_used_cohort=5)
    assert "0.800 [0.800, 0.800]" in tex


def test_table_tex_dashes_empty_cells() -> None:
    records = [_make_sidecar("0001")]
    records[0]["metrics"]["ET"]["sensitivity"] = None
    summary = mod.summarise_per_cell(records)
    tex = mod.format_table_tex(summary, n_total=1, n_used_cohort=1)
    # Each metric row is one line; find the Sensitivity line and check
    # that its ET column is '--'.
    sens_line = next(line for line in tex.splitlines() if "Sensitivity" in line)
    assert sens_line.rstrip().endswith("-- \\\\")


def test_table_tex_includes_denominator_footnote_when_n_used_differs() -> None:
    """When some cell has n_used < n_total, a footnote line is emitted."""
    records = [_make_sidecar("0001"), _make_sidecar("0002")]
    records[0]["metrics"]["WT"]["hd95_mm"] = "inf"
    summary = mod.summarise_per_cell(records)
    tex = mod.format_table_tex(summary, n_total=2, n_used_cohort=2)
    assert "Per-cell denominator" in tex
    assert "WT" in tex


# ---------------------------------------------------------------------------
# main(): full CLI orchestration on synthetic data
# ---------------------------------------------------------------------------


def _write_minimal_cohort_yaml(path: Path, subject_ids: list[str]) -> None:
    rows = "\n".join(f"  - {sid}" for sid in subject_ids)
    path.write_text(
        "seed: 20260415\n"
        f"n: {len(subject_ids)}\n"
        "stratification:\n"
        "  os_tertile_cuts_days: [300, 700]\n"
        f"subjects:\n{rows}\n",
    )


def _write_minimal_metadata_csv(path: Path, subject_ids: list[str]) -> None:
    df = pd.DataFrame(
        {
            "ID": [f"UCSF-PDGM-{sid}" for sid in subject_ids],
            "Sex": ["F"] * len(subject_ids),
            "Age at MRI": [60.0] * len(subject_ids),
            "OS": [365] * len(subject_ids),
            "1-dead 0-alive": [1] * len(subject_ids),
            "WHO CNS Grade": [4] * len(subject_ids),
            "Final pathologic diagnosis (WHO 2021)": ["GBM"] * len(subject_ids),
        },
    )
    df.to_csv(path, index=False)


def test_main_writes_three_artefacts(tmp_path: Path, monkeypatch, capsys) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    metrics_root = tmp_path / "deriv"
    out_tex = tmp_path / "out" / "table2.tex"
    out_csv = tmp_path / "out" / "table2.csv"
    out_json = tmp_path / "out" / "table2.json"

    sids = ["0001", "0002", "0003"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    for sid in sids:
        _write_sidecar_to_disk(metrics_root, _make_sidecar(sid))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table2_dl_vs_gt.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--metrics-root",
            str(metrics_root),
            "--out-tex",
            str(out_tex),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )
    rc = mod.main()
    capsys.readouterr()  # silence the printed report
    assert rc == 0
    assert out_tex.is_file()
    assert out_csv.is_file()
    assert out_json.is_file()
    summary_doc = json.loads(out_json.read_text())
    assert summary_doc["n_total"] == 3
    assert summary_doc["n_with_sidecar"] == 3
    assert summary_doc["n_problems"] == 0
    assert set(summary_doc["compartments"]) == set(mod.COMPARTMENTS)


def test_main_returns_2_on_missing_metrics_root(tmp_path: Path, monkeypatch, capsys) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    sids = ["0001"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table2_dl_vs_gt.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--metrics-root",
            str(tmp_path / "no-such-dir"),
        ],
    )
    rc = mod.main()
    capsys.readouterr()
    assert rc == 2


def test_main_returns_1_when_a_cell_has_no_finite_samples(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    metrics_root = tmp_path / "deriv"
    out_tex = tmp_path / "out" / "table2.tex"
    out_csv = tmp_path / "out" / "table2.csv"
    out_json = tmp_path / "out" / "table2.json"

    sids = ["0001", "0002"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    for sid in sids:
        doc = _make_sidecar(sid)
        # Wipe ET sensitivity for both subjects -> n_used = 0 for that cell.
        doc["metrics"]["ET"]["sensitivity"] = None
        _write_sidecar_to_disk(metrics_root, doc)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table2_dl_vs_gt.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--metrics-root",
            str(metrics_root),
            "--out-tex",
            str(out_tex),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )
    rc = mod.main()
    capsys.readouterr()
    assert rc == 1
    summary_doc = json.loads(out_json.read_text())
    assert summary_doc["summary"]["ET"]["sensitivity"]["n_used"] == 0


def test_main_strict_returns_3_on_missing_sidecar(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    metrics_root = tmp_path / "deriv"
    metrics_root.mkdir()  # exists but empty

    sids = ["0001"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table2_dl_vs_gt.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--metrics-root",
            str(metrics_root),
            "--strict",
        ],
    )
    rc = mod.main()
    capsys.readouterr()
    assert rc == 3
