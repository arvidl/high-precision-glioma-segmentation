"""Unit tests for ``scripts/build_legacy5_dlvsgt_panel.py``.

Loaded via ``importlib`` (``scripts/`` is intentionally not on
``sys.path``). Coverage rubric: deterministic S-D subject pick,
per-subject DL/GT alignment, end-to-end main() that writes PDF +
PNG + JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_legacy5_dlvsgt_panel.py"
    spec = importlib.util.spec_from_file_location(
        "build_legacy5_dlvsgt_panel_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# select_representative_subjects (the S-D rule)
# ---------------------------------------------------------------------------


def _toy_agreement_df() -> pd.DataFrame:
    """One subject per row of |residual| (cells collapsed for brevity)."""
    rows = []
    # subject 0001 mean = 10, 0002 = 20, 0003 = 30, 0004 = 40, 0005 = 50,
    # 0006 = 60, 0007 = 100. Cohort mean = (10+20+30+40+50+60+100)/7 = 44.29.
    for sid, value in [
        ("0001", 10.0),
        ("0002", 20.0),
        ("0003", 30.0),
        ("0004", 40.0),
        ("0005", 50.0),
        ("0006", 60.0),
        ("0007", 100.0),
    ]:
        # Three cells per subject so groupby().mean() == value.
        for _ in range(3):
            rows.append({"subject_id": sid, "abs_diff_volume_mm3": value})
    return pd.DataFrame(rows)


def test_select_picks_n_closest_to_cohort_mean() -> None:
    pick = mod.select_representative_subjects(_toy_agreement_df(), n_pick=5)
    assert len(pick.selected) == 5
    assert pick.cohort_mean_abs_residual == pytest.approx(44.285714, abs=1e-4)
    # Distances from 44.29: 0001=34.3, 0002=24.3, 0003=14.3, 0004=4.3,
    # 0005=5.7, 0006=15.7, 0007=55.7
    # Top-5 closest: 0004 (4.3), 0005 (5.7), 0003 (14.3), 0006 (15.7), 0002 (24.3)
    assert pick.selected == ("0004", "0005", "0003", "0006", "0002")


def test_select_is_deterministic_under_id_tiebreak() -> None:
    """Two subjects with identical distance should resolve by subject_id ascending."""
    df = pd.DataFrame(
        [
            # cohort mean = (5+15)/2 = 10; both are 5 away.
            {"subject_id": "0099", "abs_diff_volume_mm3": 5.0},
            {"subject_id": "0001", "abs_diff_volume_mm3": 15.0},
        ],
    )
    pick = mod.select_representative_subjects(df, n_pick=2)
    # Both at distance 5; ascending id wins.
    assert pick.selected == ("0001", "0099")


def test_select_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        mod.select_representative_subjects(pd.DataFrame({"foo": [1]}), n_pick=2)


def test_select_per_subject_dict_keys_match_selection() -> None:
    pick = mod.select_representative_subjects(_toy_agreement_df(), n_pick=3)
    for sid in pick.selected:
        assert sid in pick.per_subject_mean_abs_residual


# ---------------------------------------------------------------------------
# _aggregate_subject (DL + GT alignment)
# ---------------------------------------------------------------------------


def _write_synth_hitplot_pair(
    hitplot_root: Path,
    subject_id: str,
    *,
    dl_volumes: dict[str, dict[str, float]],
    gt_volumes: dict[str, dict[str, float]],
) -> None:
    """Write per-subject hitplot_dl.csv and hitplot_gt.csv into the cohort tree.

    ``dl_volumes`` / ``gt_volumes``: {region_name: {WT, TC, ET}} in mm^3.
    """
    sub_dir = hitplot_root / f"sub-{subject_id}" / "hitplot"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for source, data in (("dl", dl_volumes), ("gt", gt_volumes)):
        rows = []
        for label, (name, vols) in enumerate(data.items(), start=1000):
            for comp, vol in vols.items():
                rows.append(
                    {
                        "label": label,
                        "name": name,
                        "compartment": comp,
                        "overlap_volume_mm3": vol,
                    },
                )
        pd.DataFrame(rows).to_csv(sub_dir / f"sub-{subject_id}_hitplot_{source}.csv", index=False)


def test_aggregate_subject_aligns_regions(tmp_path: Path) -> None:
    _write_synth_hitplot_pair(
        tmp_path,
        "0020",
        dl_volumes={
            "ctx-lh-supramarginal": {"WT": 100.0, "TC": 30.0, "ET": 10.0},
            "Brain-Stem": {"WT": 5.0, "TC": 5.0, "ET": 5.0},
        },
        gt_volumes={
            "ctx-lh-supramarginal": {"WT": 90.0, "TC": 25.0, "ET": 8.0},
            "Brain-Stem": {"WT": 0.0, "TC": 0.0, "ET": 0.0},  # GT-absent region
        },
    )
    dl_top, gt_top = mod._aggregate_subject(tmp_path, "0020", k=5)
    # Both top frames must have the SAME index (the alignment contract).
    assert list(dl_top.index) == list(gt_top.index)
    # Brain-Stem appears even though GT is empty (DL non-zero).
    assert "Brain-Stem" in dl_top.index
    # Disjoint NCR/ED/ET schema is enforced.
    assert list(dl_top.columns) == list(mod.CANONICAL_COMPARTMENT_ORDER)


def test_aggregate_subject_raises_when_csvs_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod._aggregate_subject(tmp_path, "9999", k=5)


# ---------------------------------------------------------------------------
# main(): end-to-end small fixture
# ---------------------------------------------------------------------------


def test_main_writes_pdf_png_and_json(tmp_path: Path) -> None:
    hitplot_root = tmp_path / "deriv"
    for sid in ("0001", "0002", "0003", "0004", "0005"):
        _write_synth_hitplot_pair(
            hitplot_root,
            sid,
            dl_volumes={
                "ctx-lh-supramarginal": {"WT": 100.0, "TC": 30.0, "ET": 10.0},
                "Brain-Stem": {"WT": 5.0, "TC": 5.0, "ET": 5.0},
            },
            gt_volumes={
                "ctx-lh-supramarginal": {"WT": 95.0, "TC": 28.0, "ET": 9.0},
                "Brain-Stem": {"WT": 4.0, "TC": 4.0, "ET": 4.0},
            },
        )

    agreement_csv = tmp_path / "agreement.csv"
    pd.DataFrame(
        [
            {"subject_id": "0001", "abs_diff_volume_mm3": 9.0},
            {"subject_id": "0002", "abs_diff_volume_mm3": 10.0},
            {"subject_id": "0003", "abs_diff_volume_mm3": 11.0},
            {"subject_id": "0004", "abs_diff_volume_mm3": 12.0},
            {"subject_id": "0005", "abs_diff_volume_mm3": 13.0},
        ],
    ).to_csv(agreement_csv, index=False)

    out_pdf = tmp_path / "supp.pdf"
    out_png = tmp_path / "supp.png"
    out_json = tmp_path / "supp.json"
    rc = mod.main(
        [
            "--agreement-csv",
            str(agreement_csv),
            "--hitplot-root",
            str(hitplot_root),
            "--top-k",
            "2",
            "--out-pdf",
            str(out_pdf),
            "--out-png",
            str(out_png),
            "--out-json",
            str(out_json),
        ],
    )
    assert rc == 0
    assert out_pdf.is_file()
    assert out_png.is_file()
    assert out_json.is_file()
    payload = json.loads(out_json.read_text())
    assert len(payload["selected_subjects"]) == 5
    assert payload["selection_rule"].startswith("S-D")
    assert "cohort_mean_abs_residual_mm3" in payload
