"""Unit tests for ``scripts/build_table3_segmenter_agreement.py`` (PR-7h).

We import the script as a module via ``importlib`` (the ``scripts/``
directory is intentionally not on ``sys.path``; the script is a CLI
shim). Helpers, schema validation, NaN/Inf-safe aggregation, and the
LaTeX/CSV/JSON outputs are covered here on synthetic fixtures so the
test does not depend on torch / monai / FreeSurfer / Docker / network
access or on real PR-7d / PR-7e sidecars on disk.

``paper/main.tex`` (R2.5 / R2.10) and for the cross-cell
consistency check (the ``BraTS21-manual`` column reproduces Table 2).
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
    script_path = repo_root / "scripts" / "build_table3_segmenter_agreement.py"
    spec = importlib.util.spec_from_file_location(
        "build_table3_segmenter_agreement",
        script_path,
    )
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
    comparison: str,
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
    """Build a fully-populated PR-7d/PR-7e-shaped paired-metrics sidecar dict.

    The ``comparison`` field selects which of the three Table 3
    families the sidecar belongs to (must be one of
    ``dl_vs_raidionics`` / ``dl_vs_segmentglioma`` / ``dl_vs_gt``).
    """
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


def _write_sidecar_to_disk(metrics_root: Path, doc: dict, *, pair_id: str) -> Path:
    sid = doc["subject_id"]
    p = mod._metrics_path(metrics_root, sid, pair_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def _write_full_subject(
    metrics_root: Path,
    sid: str,
    *,
    raid_dice: float = 0.78,
    sg_dice: float = 0.82,
    gt_dice: float = 0.90,
) -> None:
    """Write all three PR-7h sidecar families for a single subject.

    Lets us test the cross-pair behaviour with different magnitudes
    per pair --- useful to assert that columns are not silently
    swapped at the loader level.
    """
    _write_sidecar_to_disk(
        metrics_root,
        _make_sidecar(sid, comparison="dl_vs_raidionics", wt_dice=raid_dice),
        pair_id="dl_vs_raidionics",
    )
    _write_sidecar_to_disk(
        metrics_root,
        _make_sidecar(sid, comparison="dl_vs_segmentglioma", wt_dice=sg_dice),
        pair_id="dl_vs_segmentglioma",
    )
    _write_sidecar_to_disk(
        metrics_root,
        _make_sidecar(sid, comparison="dl_vs_gt", wt_dice=gt_dice),
        pair_id="dl_vs_gt",
    )


# ---------------------------------------------------------------------------
# Module-level constants and tables
# ---------------------------------------------------------------------------


def test_pair_ids_are_the_three_expected_families() -> None:
    assert mod._PAIR_IDS == ("dl_vs_raidionics", "dl_vs_segmentglioma", "dl_vs_gt")


def test_compartments_match_table_2() -> None:
    assert mod.COMPARTMENTS == ("WT", "TC", "ET")


def test_metric_keys_match_q2c_bundle() -> None:
    assert mod._METRIC_KEYS == (
        "dice",
        "hd95_mm",
        "abs_volumetric_error_mm3",
        "sensitivity",
        "specificity",
    )


def test_metrics_path_uses_pair_suffix(tmp_path: Path) -> None:
    p = mod._metrics_path(tmp_path, "0005", "dl_vs_raidionics")
    assert p.name == "sub-0005_metrics_dl_vs_raidionics.json"
    assert p.parent.name == "metrics"
    assert p.parent.parent.name == "sub-0005"


def test_metrics_path_normalises_subject_prefix(tmp_path: Path) -> None:
    p = mod._metrics_path(tmp_path, "sub-0005", "dl_vs_gt")
    assert p.name == "sub-0005_metrics_dl_vs_gt.json"


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
    assert mod._classify_value(True) == ("missing", None)
    assert mod._classify_value(False) == ("missing", None)


# ---------------------------------------------------------------------------
# _validate_sidecar
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed_sidecar_for_each_pair() -> None:
    for cmp in ("dl_vs_raidionics", "dl_vs_segmentglioma", "dl_vs_gt"):
        mod._validate_sidecar(
            _make_sidecar("0005", comparison=cmp),
            expected_comparison=cmp,
        )


def test_validate_rejects_unknown_schema_version() -> None:
    bad = _make_sidecar("0005", comparison="dl_vs_raidionics")
    bad["schema_version"] = "9.9"
    with pytest.raises(mod.SidecarError, match="schema_version"):
        mod._validate_sidecar(bad, expected_comparison="dl_vs_raidionics")


def test_validate_rejects_wrong_comparison() -> None:
    """Cross-pair confusion (raidionics sidecar served as gt) must trip the validator."""
    bad = _make_sidecar("0005", comparison="dl_vs_raidionics")
    with pytest.raises(mod.SidecarError, match="comparison"):
        mod._validate_sidecar(bad, expected_comparison="dl_vs_gt")


def test_validate_rejects_missing_compartment() -> None:
    bad = _make_sidecar("0005", comparison="dl_vs_gt")
    del bad["metrics"]["ET"]
    with pytest.raises(mod.SidecarError, match="ET"):
        mod._validate_sidecar(bad, expected_comparison="dl_vs_gt")


def test_validate_rejects_missing_metric() -> None:
    bad = _make_sidecar("0005", comparison="dl_vs_gt")
    del bad["metrics"]["WT"]["hd95_mm"]
    with pytest.raises(mod.SidecarError, match="hd95_mm"):
        mod._validate_sidecar(bad, expected_comparison="dl_vs_gt")


def test_validate_tolerates_extra_keys() -> None:
    """Forward-compat: extra unknown keys must not break the loader."""
    extra = _make_sidecar("0005", comparison="dl_vs_gt")
    extra["future_key"] = "ignored"
    extra["metrics"]["WT"]["future_metric"] = 0.5
    mod._validate_sidecar(extra, expected_comparison="dl_vs_gt")  # must not raise


# ---------------------------------------------------------------------------
# load_metrics_sidecars
# ---------------------------------------------------------------------------


def test_load_sidecars_happy_path_per_pair(tmp_path: Path) -> None:
    for sid in ("0005", "0026", "0068"):
        _write_full_subject(tmp_path, sid)
    for pair_id in mod._PAIR_IDS:
        records, problems = mod.load_metrics_sidecars(
            tmp_path,
            ["0005", "0026", "0068"],
            pair_id=pair_id,
        )
        assert problems == []
        assert [r["subject_id"] for r in records] == ["0005", "0026", "0068"]


def test_load_sidecars_normalises_subject_prefix(tmp_path: Path) -> None:
    _write_sidecar_to_disk(
        tmp_path,
        _make_sidecar("0005", comparison="dl_vs_raidionics"),
        pair_id="dl_vs_raidionics",
    )
    records, problems = mod.load_metrics_sidecars(
        tmp_path,
        ["sub-0005"],
        pair_id="dl_vs_raidionics",
    )
    assert problems == []
    assert records[0]["subject_id"] == "0005"


def test_load_sidecars_records_missing_files(tmp_path: Path) -> None:
    _write_sidecar_to_disk(
        tmp_path,
        _make_sidecar("0005", comparison="dl_vs_gt"),
        pair_id="dl_vs_gt",
    )
    records, problems = mod.load_metrics_sidecars(
        tmp_path,
        ["0005", "0026"],
        pair_id="dl_vs_gt",
    )
    assert [r["subject_id"] for r in records] == ["0005"]
    assert len(problems) == 1
    assert problems[0][0] == "0026"
    assert "not found" in problems[0][1]


def test_load_sidecars_records_malformed_json(tmp_path: Path) -> None:
    _write_sidecar_to_disk(
        tmp_path,
        _make_sidecar("0005", comparison="dl_vs_segmentglioma"),
        pair_id="dl_vs_segmentglioma",
    )
    bad_path = mod._metrics_path(tmp_path, "0026", "dl_vs_segmentglioma")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not json")
    records, problems = mod.load_metrics_sidecars(
        tmp_path,
        ["0005", "0026"],
        pair_id="dl_vs_segmentglioma",
    )
    assert [r["subject_id"] for r in records] == ["0005"]
    assert problems[0][0] == "0026"
    assert "JSON" in problems[0][1]


def test_load_sidecars_strict_raises_on_first_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_metrics_sidecars(
            tmp_path,
            ["0005"],
            pair_id="dl_vs_raidionics",
            strict=True,
        )


def test_load_sidecars_strict_raises_on_schema_violation(tmp_path: Path) -> None:
    bad = _make_sidecar("0005", comparison="dl_vs_gt")
    bad["schema_version"] = "9.9"
    _write_sidecar_to_disk(tmp_path, bad, pair_id="dl_vs_gt")
    with pytest.raises(mod.SidecarError):
        mod.load_metrics_sidecars(
            tmp_path,
            ["0005"],
            pair_id="dl_vs_gt",
            strict=True,
        )


def test_load_sidecars_rejects_swapped_comparison(tmp_path: Path) -> None:
    """A raidionics sidecar served from the segmentglioma path is a problem.

    Regression guard: pair_id selects the on-disk filename *and* the
    expected ``comparison`` field; so if a sidecar is mislabeled
    upstream, the loader must surface it instead of silently feeding
    the wrong cells to ``summarise_per_cell``.
    """
    swap = _make_sidecar("0005", comparison="dl_vs_raidionics")
    # Park it where the loader looks for the segmentglioma sidecar.
    p = mod._metrics_path(tmp_path, "0005", "dl_vs_segmentglioma")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(swap))
    records, problems = mod.load_metrics_sidecars(
        tmp_path,
        ["0005"],
        pair_id="dl_vs_segmentglioma",
    )
    assert records == []
    assert len(problems) == 1
    assert "comparison" in problems[0][1]


# ---------------------------------------------------------------------------
# summarise_per_cell + summarise_all_pairs
# ---------------------------------------------------------------------------


def test_summarise_known_dice_values() -> None:
    """Hand-checked median/Q1/Q3 against a small known sample."""
    records = [
        _make_sidecar("0001", comparison="dl_vs_raidionics", wt_dice=0.50),
        _make_sidecar("0002", comparison="dl_vs_raidionics", wt_dice=0.60),
        _make_sidecar("0003", comparison="dl_vs_raidionics", wt_dice=0.70),
        _make_sidecar("0004", comparison="dl_vs_raidionics", wt_dice=0.80),
        _make_sidecar("0005", comparison="dl_vs_raidionics", wt_dice=0.90),
    ]
    summary = mod.summarise_per_cell(records)
    cell = summary["WT"]["dice"]
    assert cell["median"] == pytest.approx(0.70)
    assert cell["q1"] == pytest.approx(0.60)
    assert cell["q3"] == pytest.approx(0.80)
    assert cell["n_used"] == 5
    assert cell["n_missing"] == 0
    assert cell["n_inf"] == 0


def test_summarise_excludes_null_and_inf() -> None:
    """null -> n_missing; "inf" -> n_inf; both excluded from the median."""
    records = [
        _make_sidecar("0001", comparison="dl_vs_gt", wt_dice=0.5),
        _make_sidecar("0002", comparison="dl_vs_gt", wt_dice=0.7),
        _make_sidecar("0003", comparison="dl_vs_gt", wt_dice=0.9),
    ]
    records[0]["metrics"]["WT"]["hd95_mm"] = None
    records[1]["metrics"]["WT"]["hd95_mm"] = "inf"
    records[2]["metrics"]["WT"]["hd95_mm"] = 5.0
    summary = mod.summarise_per_cell(records)
    cell = summary["WT"]["hd95_mm"]
    assert cell["median"] == pytest.approx(5.0)
    assert cell["n_used"] == 1
    assert cell["n_missing"] == 1
    assert cell["n_inf"] == 1
    assert summary["WT"]["dice"]["n_used"] == 3


def test_summarise_all_missing_returns_none() -> None:
    """Cell with zero finite samples: median/Q1/Q3 all None, JSON-safe."""
    records = [
        _make_sidecar("0001", comparison="dl_vs_segmentglioma"),
        _make_sidecar("0002", comparison="dl_vs_segmentglioma"),
    ]
    for rec in records:
        rec["metrics"]["ET"]["sensitivity"] = None
    summary = mod.summarise_per_cell(records)
    cell = summary["ET"]["sensitivity"]
    assert cell["median"] is None
    assert cell["q1"] is None
    assert cell["q3"] is None
    assert cell["n_used"] == 0
    assert cell["n_missing"] == 2
    json.dumps(summary)


def test_summarise_shape_matches_compartments_and_metrics() -> None:
    summary = mod.summarise_per_cell(
        [_make_sidecar("0001", comparison="dl_vs_raidionics")],
    )
    assert set(summary) == set(mod.COMPARTMENTS)
    for comp in mod.COMPARTMENTS:
        assert set(summary[comp]) == set(mod._METRIC_KEYS)


def test_summarise_all_pairs_returns_one_block_per_pair() -> None:
    records_per_pair = {
        "dl_vs_raidionics": [
            _make_sidecar("0001", comparison="dl_vs_raidionics", wt_dice=0.50),
        ],
        "dl_vs_segmentglioma": [
            _make_sidecar("0001", comparison="dl_vs_segmentglioma", wt_dice=0.60),
        ],
        "dl_vs_gt": [
            _make_sidecar("0001", comparison="dl_vs_gt", wt_dice=0.90),
        ],
    }
    summary = mod.summarise_all_pairs(records_per_pair)
    assert set(summary) == set(mod._PAIR_IDS)
    # Per-pair median is the sole sample (n=1).
    assert summary["dl_vs_raidionics"]["WT"]["dice"]["median"] == pytest.approx(0.50)
    assert summary["dl_vs_segmentglioma"]["WT"]["dice"]["median"] == pytest.approx(0.60)
    assert summary["dl_vs_gt"]["WT"]["dice"]["median"] == pytest.approx(0.90)


def test_summarise_all_pairs_handles_empty_pair() -> None:
    """If one pair's records are absent (e.g. baseline never ran),
    its block must still be present with all-empty cells."""
    records_per_pair = {
        "dl_vs_raidionics": [],
        "dl_vs_segmentglioma": [],
        "dl_vs_gt": [_make_sidecar("0001", comparison="dl_vs_gt")],
    }
    summary = mod.summarise_all_pairs(records_per_pair)
    assert summary["dl_vs_raidionics"]["WT"]["dice"]["n_used"] == 0
    assert summary["dl_vs_raidionics"]["WT"]["dice"]["median"] is None
    # Sanity: the populated pair still aggregates correctly.
    assert summary["dl_vs_gt"]["WT"]["dice"]["n_used"] == 1


# ---------------------------------------------------------------------------
# long_audit_frame
# ---------------------------------------------------------------------------


def test_audit_frame_columns_and_row_count() -> None:
    records_per_pair = {
        pid: [
            _make_sidecar("0001", comparison=mod._PAIR_COMPARISON[pid]),
            _make_sidecar("0002", comparison=mod._PAIR_COMPARISON[pid]),
        ]
        for pid in mod._PAIR_IDS
    }
    df = mod.long_audit_frame(records_per_pair)
    assert list(df.columns) == [
        "subject_id",
        "pair_id",
        "compartment",
        "metric",
        "value",
        "status",
        "raw",
    ]
    # 3 pairs * 2 subjects * 3 compartments * 5 metrics = 90 rows.
    assert len(df) == 3 * 2 * 3 * 5
    assert (df["status"] == "ok").all()
    assert set(df["pair_id"]) == set(mod._PAIR_IDS)


def test_audit_frame_marks_status_for_null_and_inf() -> None:
    records_per_pair = {
        "dl_vs_raidionics": [_make_sidecar("0001", comparison="dl_vs_raidionics")],
        "dl_vs_segmentglioma": [],
        "dl_vs_gt": [],
    }
    records_per_pair["dl_vs_raidionics"][0]["metrics"]["WT"]["dice"] = None
    records_per_pair["dl_vs_raidionics"][0]["metrics"]["WT"]["hd95_mm"] = "inf"
    df = mod.long_audit_frame(records_per_pair)
    wt_dice = df.query(
        "pair_id == 'dl_vs_raidionics' and compartment == 'WT' and metric == 'dice'",
    )
    wt_hd95 = df.query(
        "pair_id == 'dl_vs_raidionics' and compartment == 'WT' and metric == 'hd95_mm'",
    )
    assert wt_dice["status"].iloc[0] == "missing"
    assert wt_hd95["status"].iloc[0] == "inf"
    assert np.isnan(wt_dice["value"].iloc[0])


# ---------------------------------------------------------------------------
# format_table_tex
# ---------------------------------------------------------------------------


def _tiny_summary_per_pair() -> dict:
    """Build a one-subject summary for each pair, with distinct WT-Dice values."""
    return {
        "dl_vs_raidionics": mod.summarise_per_cell(
            [_make_sidecar("0001", comparison="dl_vs_raidionics", wt_dice=0.78)],
        ),
        "dl_vs_segmentglioma": mod.summarise_per_cell(
            [_make_sidecar("0001", comparison="dl_vs_segmentglioma", wt_dice=0.82)],
        ),
        "dl_vs_gt": mod.summarise_per_cell(
            [_make_sidecar("0001", comparison="dl_vs_gt", wt_dice=0.90)],
        ),
    }


def test_table_tex_contains_required_structure() -> None:
    summary = _tiny_summary_per_pair()
    tex = mod.format_table_tex(
        summary,
        n_total=1,
        n_used_per_pair={pid: 1 for pid in mod._PAIR_IDS},
    )
    assert r"\begin{tabular}{l c c c}" in tex
    assert r"\end{tabular}" in tex
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    # All five metric block headers appear.
    for _, label, _ in mod.METRICS:
        assert label in tex
    # All three compartment row labels appear (as ``\quad WT`` etc.).
    for comp in mod.COMPARTMENTS:
        assert rf"\quad {comp}" in tex
    # All three column headers appear.
    for pid in mod._PAIR_IDS:
        assert mod._PAIR_LATEX[pid] in tex


def test_table_tex_renders_distinct_per_pair_cells() -> None:
    """Per-pair WT-Dice values must show up in the rendered table; they
    must not be silently averaged or column-swapped."""
    summary = _tiny_summary_per_pair()
    tex = mod.format_table_tex(
        summary,
        n_total=1,
        n_used_per_pair={pid: 1 for pid in mod._PAIR_IDS},
    )
    # n=1 so median = q1 = q3 = the input value.
    # 0.78 -> raidionics, 0.82 -> segmentglioma, 0.90 -> BraTS21 (DL-vs-gt).
    assert "0.780 [0.780, 0.780]" in tex
    assert "0.820 [0.820, 0.820]" in tex
    assert "0.900 [0.900, 0.900]" in tex
    # The WT-Dice row should have those three values in column order.
    wt_dice_line = next(
        line
        for line in tex.splitlines()
        if line.startswith(r"\quad WT") and "0.780" in line and "0.820" in line and "0.900" in line
    )
    # Column order: raidionics, segmentglioma, gt.
    pos_r = wt_dice_line.find("0.780")
    pos_s = wt_dice_line.find("0.820")
    pos_g = wt_dice_line.find("0.900")
    assert pos_r < pos_s < pos_g


def test_table_tex_dashes_empty_cells() -> None:
    """An empty cell in one pair must render as ``--`` while leaving
    the other two columns intact."""
    records = [_make_sidecar("0001", comparison="dl_vs_raidionics")]
    records[0]["metrics"]["ET"]["sensitivity"] = None
    summary = {
        "dl_vs_raidionics": mod.summarise_per_cell(records),
        "dl_vs_segmentglioma": mod.summarise_per_cell(
            [_make_sidecar("0001", comparison="dl_vs_segmentglioma")],
        ),
        "dl_vs_gt": mod.summarise_per_cell(
            [_make_sidecar("0001", comparison="dl_vs_gt")],
        ),
    }
    tex = mod.format_table_tex(
        summary,
        n_total=1,
        n_used_per_pair={pid: 1 for pid in mod._PAIR_IDS},
    )
    et_lines = [line for line in tex.splitlines() if line.startswith(r"\quad ET")]
    # ET sensitivity is the row whose first numeric column is '--'.
    sens_line = next(line for line in et_lines if r"-- &" in line)
    # The other two columns are populated (default 0.84 / 0.84) and the
    # last column should not be '--'.
    assert not sens_line.rstrip().endswith(r"-- \\")


def test_table_tex_includes_denominator_footnote_when_n_used_differs() -> None:
    summary = _tiny_summary_per_pair()
    # n_total=10 but n_used_per_pair=1 -> footnote should fire for every cell.
    tex = mod.format_table_tex(
        summary,
        n_total=10,
        n_used_per_pair={pid: 1 for pid in mod._PAIR_IDS},
    )
    assert "Per-cell denominator" in tex
    # The footnote must qualify by pair (PR-7h-specific extension).
    for plain in ("Raidionics", "segment_glioma", "BraTS21-manual"):
        assert plain in tex


def test_table_tex_no_footnote_when_all_cells_full() -> None:
    summary = _tiny_summary_per_pair()
    tex = mod.format_table_tex(
        summary,
        n_total=1,
        n_used_per_pair={pid: 1 for pid in mod._PAIR_IDS},
    )
    assert "Per-cell denominator" not in tex


# ---------------------------------------------------------------------------
# Cross-table consistency: BraTS21-manual column reproduces Table 2
# ---------------------------------------------------------------------------


def test_brats21_column_reproduces_table2_dl_vs_gt_input() -> None:
    """The PR-7h ``dl_vs_gt`` block must aggregate the same numbers as
    the PR-7g (Table 2) producer would on the same sidecars.

    We import the PR-7g module directly and compare the per-cell
    summaries; they must be byte-equal under the same fixture, since
    the aggregation rules and metric/compartment keys are identical.
    """
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "build_table2_dl_vs_gt",
        repo_root / "scripts" / "build_table2_dl_vs_gt.py",
    )
    assert spec is not None and spec.loader is not None
    table2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = table2
    spec.loader.exec_module(table2)

    fixture = [
        _make_sidecar("0001", comparison="dl_vs_gt", wt_dice=0.85, tc_dice=0.78),
        _make_sidecar("0002", comparison="dl_vs_gt", wt_dice=0.91, tc_dice=0.82),
        _make_sidecar("0003", comparison="dl_vs_gt", wt_dice=0.88, tc_dice=0.79),
    ]
    table2_summary = table2.summarise_per_cell(fixture)
    table3_summary = mod.summarise_per_cell(fixture)
    assert table2_summary == table3_summary


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
    out_tex = tmp_path / "out" / "table3.tex"
    out_csv = tmp_path / "out" / "table3.csv"
    out_json = tmp_path / "out" / "table3.json"

    sids = ["0001", "0002", "0003"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    for sid in sids:
        _write_full_subject(metrics_root, sid)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
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
    assert rc == 0
    assert out_tex.is_file()
    assert out_csv.is_file()
    assert out_json.is_file()
    summary_doc = json.loads(out_json.read_text())
    assert summary_doc["n_total"] == 3
    assert summary_doc["n_with_sidecar_per_pair"] == {pid: 3 for pid in mod._PAIR_IDS}
    assert summary_doc["n_problems_per_pair"] == {pid: 0 for pid in mod._PAIR_IDS}
    assert set(summary_doc["compartments"]) == set(mod.COMPARTMENTS)
    assert set(summary_doc["comparisons"]) == set(mod._PAIR_IDS)
    # CCC slot is reserved for PR-7i; explicit None is documentation.
    assert summary_doc["ccc_per_cell"] is None


def test_main_writes_csv_with_pair_id_column(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    metrics_root = tmp_path / "deriv"
    out_tex = tmp_path / "out" / "table3.tex"
    out_csv = tmp_path / "out" / "table3.csv"
    out_json = tmp_path / "out" / "table3.json"

    sids = ["0001"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    _write_full_subject(metrics_root, "0001")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
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
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert "pair_id" in df.columns
    assert set(df["pair_id"]) == set(mod._PAIR_IDS)


def test_main_returns_2_on_missing_metrics_root(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    sids = ["0001"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
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


def test_main_returns_2_on_missing_cohort_yaml(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
            "--cohort-yaml",
            str(tmp_path / "no-such.yaml"),
            "--metadata-csv",
            str(tmp_path / "no-such.csv"),
            "--metrics-root",
            str(tmp_path),
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
    out_tex = tmp_path / "out" / "table3.tex"
    out_csv = tmp_path / "out" / "table3.csv"
    out_json = tmp_path / "out" / "table3.json"

    sids = ["0001", "0002"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    for sid in sids:
        _write_full_subject(metrics_root, sid)
        # Wipe ET sensitivity in the raidionics sidecar for both subjects ->
        # n_used = 0 for the (raidionics, ET, sensitivity) cell only.
        path = mod._metrics_path(metrics_root, sid, "dl_vs_raidionics")
        doc = json.loads(path.read_text())
        doc["metrics"]["ET"]["sensitivity"] = None
        path.write_text(json.dumps(doc))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
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
    assert summary_doc["summary_per_pair"]["dl_vs_raidionics"]["ET"]["sensitivity"]["n_used"] == 0
    # The other two pairs' cells must still be populated.
    assert (
        summary_doc["summary_per_pair"]["dl_vs_segmentglioma"]["ET"]["sensitivity"]["n_used"] == 2
    )
    assert summary_doc["summary_per_pair"]["dl_vs_gt"]["ET"]["sensitivity"]["n_used"] == 2


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
            "build_table3_segmenter_agreement.py",
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


def test_main_warns_about_problems_without_strict(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Without --strict, missing sidecars are warnings on stderr, not errors."""
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "meta.csv"
    metrics_root = tmp_path / "deriv"
    out_tex = tmp_path / "out" / "table3.tex"
    out_csv = tmp_path / "out" / "table3.csv"
    out_json = tmp_path / "out" / "table3.json"

    sids = ["0001", "0002"]
    _write_minimal_cohort_yaml(cohort_yaml, sids)
    _write_minimal_metadata_csv(metadata_csv, sids)
    # Only write 0001's three families; 0002's are missing.
    _write_full_subject(metrics_root, "0001")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_table3_segmenter_agreement.py",
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
    captured = capsys.readouterr()
    # rc=0 because every cell still has n_used >= 1 (from 0001).
    assert rc == 0
    assert "warning: sub-0002" in captured.err
    summary_doc = json.loads(out_json.read_text())
    assert summary_doc["n_problems_per_pair"] == {pid: 1 for pid in mod._PAIR_IDS}
