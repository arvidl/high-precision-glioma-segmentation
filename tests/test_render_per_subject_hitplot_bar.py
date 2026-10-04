"""Unit tests for ``scripts/render_per_subject_hitplot_bar.py``.

Loaded via ``importlib`` (``scripts/`` is intentionally not on
``sys.path``). Coverage rubric: hierarchical-to-disjoint translation,
schema detection, top-K aggregation (drops label 0 + zero-volume
rows), and the matplotlib renderer's basic output contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "render_per_subject_hitplot_bar.py"
    spec = importlib.util.spec_from_file_location(
        "render_per_subject_hitplot_bar_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# _hierarchical_to_disjoint
# ---------------------------------------------------------------------------


def test_hierarchical_to_disjoint_basic() -> None:
    out = mod._hierarchical_to_disjoint({"WT": 100.0, "TC": 30.0, "ET": 10.0})
    assert out == {"NCR": 20.0, "ED": 70.0, "ET": 10.0}
    # Sum must equal WT (the whole tumor) by construction.
    assert sum(out.values()) == pytest.approx(100.0)


def test_hierarchical_to_disjoint_all_zero() -> None:
    out = mod._hierarchical_to_disjoint({"WT": 0.0, "TC": 0.0, "ET": 0.0})
    assert out == {"NCR": 0.0, "ED": 0.0, "ET": 0.0}


def test_hierarchical_to_disjoint_clips_negative_diffs() -> None:
    """Numerical noise in upstream voxel sums could violate WT >= TC."""
    out = mod._hierarchical_to_disjoint({"WT": 5.0, "TC": 6.0, "ET": 1.0})
    # ED = WT - TC = -1.0 -> clipped to 0; NCR = TC - ET = 5.0 (kept)
    assert out["ED"] == 0.0
    assert out["NCR"] == 5.0


def test_hierarchical_to_disjoint_only_et() -> None:
    out = mod._hierarchical_to_disjoint({"WT": 4.0, "TC": 4.0, "ET": 4.0})
    assert out == {"NCR": 0.0, "ED": 0.0, "ET": 4.0}


# ---------------------------------------------------------------------------
# _detect_compartment_schema
# ---------------------------------------------------------------------------


def test_detect_schema_hierarchical() -> None:
    df = pd.DataFrame({"compartment": ["WT", "TC", "ET", "WT", "TC", "ET"]})
    assert mod._detect_compartment_schema(df) == mod.HIERARCHICAL_COMPARTMENT_ORDER


def test_detect_schema_disjoint() -> None:
    df = pd.DataFrame({"compartment": ["NCR", "ED", "ET", "NCR", "ED", "ET"]})
    assert mod._detect_compartment_schema(df) == mod.CANONICAL_COMPARTMENT_ORDER


def test_detect_schema_unknown_raises() -> None:
    df = pd.DataFrame({"compartment": ["foo", "bar"]})
    with pytest.raises(ValueError):
        mod._detect_compartment_schema(df)


# ---------------------------------------------------------------------------
# aggregate_top_k
# ---------------------------------------------------------------------------


def _toy_hierarchical_df() -> pd.DataFrame:
    """Build a tiny WT/TC/ET CSV-shaped frame for testing.

    Three regions:
      - 1031 ctx-lh-supramarginal:  ED-dominant, total WT 100.
      - 0    Unknown:               very large WT 999 -- must be dropped.
      - 17   Left-Hippocampus:      zero everywhere -- must be dropped.
      - 8    Left-Cerebellum-Cortex: tiny ET-only, total WT 5.
    """
    rows = []
    for label, name, vols in [
        (1031, "ctx-lh-supramarginal", {"WT": 100.0, "TC": 30.0, "ET": 10.0}),
        (0, "Unknown", {"WT": 999.0, "TC": 100.0, "ET": 10.0}),
        (17, "Left-Hippocampus", {"WT": 0.0, "TC": 0.0, "ET": 0.0}),
        (8, "Left-Cerebellum-Cortex", {"WT": 5.0, "TC": 5.0, "ET": 5.0}),
    ]:
        for comp, vol in vols.items():
            rows.append(
                {"label": label, "name": name, "compartment": comp, "overlap_volume_mm3": vol},
            )
    return pd.DataFrame(rows)


def test_aggregate_top_k_drops_label_zero() -> None:
    df = _toy_hierarchical_df()
    out = mod.aggregate_top_k(df, k=10)
    assert "Unknown" not in out.index


def test_aggregate_top_k_drops_zero_volume_rows() -> None:
    df = _toy_hierarchical_df()
    out = mod.aggregate_top_k(df, k=10)
    assert "Left-Hippocampus" not in out.index


def test_aggregate_top_k_translates_to_disjoint() -> None:
    df = _toy_hierarchical_df()
    out = mod.aggregate_top_k(df, k=10)
    # Columns are NCR/ED/ET (not WT/TC/ET) regardless of input schema.
    assert list(out.columns) == list(mod.CANONICAL_COMPARTMENT_ORDER)
    # Per-region check on supramarginal:
    row = out.loc["ctx-lh-supramarginal"]
    assert row["NCR"] == pytest.approx(20.0)
    assert row["ED"] == pytest.approx(70.0)
    assert row["ET"] == pytest.approx(10.0)
    # Sum matches WT.
    assert row.sum() == pytest.approx(100.0)


def test_aggregate_top_k_orders_by_total_volume_desc() -> None:
    df = _toy_hierarchical_df()
    out = mod.aggregate_top_k(df, k=10)
    # ctx-lh-supramarginal (100) > Left-Cerebellum-Cortex (5)
    assert list(out.index) == ["ctx-lh-supramarginal", "Left-Cerebellum-Cortex"]


def test_aggregate_top_k_truncates_to_k() -> None:
    df = _toy_hierarchical_df()
    out = mod.aggregate_top_k(df, k=1)
    assert len(out) == 1
    assert list(out.index) == ["ctx-lh-supramarginal"]


def test_aggregate_top_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        mod.aggregate_top_k(_toy_hierarchical_df(), k=0)


def test_aggregate_top_k_rejects_missing_columns() -> None:
    df = pd.DataFrame({"compartment": ["WT"], "overlap_volume_mm3": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        mod.aggregate_top_k(df, k=1)


def test_aggregate_top_k_handles_disjoint_input() -> None:
    """A future caller may pre-translate to NCR/ED/ET and pass it directly."""
    df = pd.DataFrame(
        [
            {
                "label": 1031,
                "name": "ctx-lh-supramarginal",
                "compartment": "NCR",
                "overlap_volume_mm3": 20.0,
            },
            {
                "label": 1031,
                "name": "ctx-lh-supramarginal",
                "compartment": "ED",
                "overlap_volume_mm3": 70.0,
            },
            {
                "label": 1031,
                "name": "ctx-lh-supramarginal",
                "compartment": "ET",
                "overlap_volume_mm3": 10.0,
            },
        ],
    )
    out = mod.aggregate_top_k(df, k=5)
    assert list(out.columns) == list(mod.CANONICAL_COMPARTMENT_ORDER)
    assert out.loc["ctx-lh-supramarginal"].sum() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# render_bar_chart
# ---------------------------------------------------------------------------


def test_render_bar_chart_returns_figure() -> None:
    aggregated = pd.DataFrame(
        {
            "NCR": [20.0, 0.0],
            "ED": [70.0, 0.0],
            "ET": [10.0, 5.0],
        },
        index=["ctx-lh-supramarginal", "Left-Cerebellum-Cortex"],
    )
    fig = mod.render_bar_chart(
        aggregated,
        subject_id="0020",
        source="gt",
        title="test",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_render_bar_chart_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no non-zero region rows"):
        mod.render_bar_chart(
            pd.DataFrame(columns=["NCR", "ED", "ET"]),
            subject_id="0020",
            source="gt",
            title=None,
        )


# ---------------------------------------------------------------------------
# main(): end-to-end via argv override
# ---------------------------------------------------------------------------


def test_main_writes_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "hp.csv"
    _toy_hierarchical_df().to_csv(csv_path, index=False)
    out_pdf = tmp_path / "out.pdf"
    out_png = tmp_path / "out.png"
    rc = mod.main(
        [
            "--csv",
            str(csv_path),
            "--subject",
            "0020",
            "--source",
            "gt",
            "--top-k",
            "5",
            "--out-pdf",
            str(out_pdf),
            "--out-png",
            str(out_png),
        ],
    )
    assert rc == 0
    assert out_pdf.is_file()
    assert out_png.is_file()
