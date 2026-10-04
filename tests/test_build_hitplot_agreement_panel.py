"""Unit tests for ``scripts/build_hitplot_agreement_panel.py`` (PR-7i).

Loaded via ``importlib`` because ``scripts/`` is intentionally not on
``sys.path``. Coverage rubric mirrors the PR-7g (Table 2) and PR-7h
(Table 3) producer tests:

* per-compartment Bland-Altman + CCC summary on hand-checked fixtures;
* identical / mirror-image / orthogonal pair invariants;
* CSV loader (happy-path / strict-mode / missing-file / malformed /
  schema-mismatch / probabilistic-CSV-rejection);
* PDF / JSON / CSV artefact contract;
* CLI orchestration + exit codes 0/1/2/3.

``\\rev{...}``-block figure in ``paper/main.tex`` that
addresses R2.8 ("sensitivity analysis of Hit-Plot to segmentation
choice").
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hpgs.metrics import concordance_correlation_coefficient


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_hitplot_agreement_panel.py"
    spec = importlib.util.spec_from_file_location(
        "build_hitplot_agreement_panel",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# Hit-Plot CSV path + schema validation
# ---------------------------------------------------------------------------


def test_hitplot_csv_path_layout(tmp_path: Path) -> None:
    p = mod._hitplot_csv_path(tmp_path / "deriv", "0005", "dl")
    assert p == tmp_path / "deriv" / "sub-0005" / "hitplot" / "sub-0005_hitplot_dl.csv"


def test_hitplot_csv_path_normalises_subject_id(tmp_path: Path) -> None:
    a = mod._hitplot_csv_path(tmp_path, "sub-0005", "gt")
    b = mod._hitplot_csv_path(tmp_path, "0005", "gt")
    assert a == b


def test_hitplot_csv_path_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in"):
        mod._hitplot_csv_path(tmp_path, "0005", "bogus")


def test_validate_hitplot_csv_accepts_canonical_schema() -> None:
    df = _make_pr7f_csv(label_to_pct={1: {"WT": 30.0, "TC": 20.0, "ET": 5.0}})
    mod._validate_hitplot_csv(df)


def test_validate_hitplot_csv_rejects_missing_required_column() -> None:
    df = _make_pr7f_csv(label_to_pct={1: {"WT": 10.0, "TC": 10.0, "ET": 5.0}})
    df = df.drop(columns=["pct_of_parcel"])
    with pytest.raises(mod.HitplotCsvError, match="missing required columns"):
        mod._validate_hitplot_csv(df, source="phantom.csv")


def test_validate_hitplot_csv_rejects_probabilistic_csv_schema() -> None:
    """Feeding a hitplot_dl_prob.csv (different schema) must hard-fail."""
    df = pd.DataFrame(
        {
            "label": [1, 2],
            "name": ["A", "B"],
            "compartment": ["WT", "TC"],
            "parcel_voxels": [10, 12],
            # No overlap_voxels / overlap_volume_mm3 / pct_of_parcel.
            "overlap_voxels_mean": [3.5, 4.0],
            "pct_of_parcel_mean": [35.0, 33.3],
        },
    )
    with pytest.raises(mod.HitplotCsvError, match="missing required columns"):
        mod._validate_hitplot_csv(df, source="probs.csv")


def test_validate_hitplot_csv_rejects_unknown_compartment() -> None:
    df = _make_pr7f_csv(label_to_pct={1: {"WT": 10.0, "TC": 10.0, "ET": 5.0}})
    df.loc[0, "compartment"] = "WTF"
    with pytest.raises(mod.HitplotCsvError, match="unexpected compartment"):
        mod._validate_hitplot_csv(df)


# ---------------------------------------------------------------------------
# load_pair_records
# ---------------------------------------------------------------------------


def _make_pr7f_csv(
    label_to_pct: dict[int, dict[str, float]],
    *,
    parcel_voxels: int = 100,
    voxel_volume_mm3: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic PR-7f Hit-Plot CSV in memory.

    Each (label, compartment) pair gets its own ``pct_of_parcel``;
    ``overlap_voxels`` is back-computed from ``parcel_voxels`` and the
    pct, and ``overlap_volume_mm3 = overlap_voxels * voxel_volume_mm3``.
    """
    rows: list[dict] = []
    for lab, comp_to_pct in label_to_pct.items():
        for comp, pct in comp_to_pct.items():
            n_overlap = round(parcel_voxels * pct / 100.0)
            rows.append(
                {
                    "label": lab,
                    "name": f"region-{lab}",
                    "compartment": comp,
                    "parcel_voxels": parcel_voxels,
                    "compartment_voxels": parcel_voxels * 3,
                    "overlap_voxels": n_overlap,
                    "overlap_volume_mm3": float(n_overlap * voxel_volume_mm3),
                    "pct_of_parcel": pct,
                    "pct_of_compartment": pct / 3.0,
                },
            )
    return pd.DataFrame(rows)


def _write_subject_pair(
    hitplot_root: Path,
    subject_id: str,
    *,
    pred_source: str,
    ref_source: str,
    pred_pct: dict[int, dict[str, float]],
    ref_pct: dict[int, dict[str, float]],
) -> None:
    sub = hitplot_root / f"sub-{subject_id}" / "hitplot"
    sub.mkdir(parents=True, exist_ok=True)
    _make_pr7f_csv(pred_pct).to_csv(
        sub / f"sub-{subject_id}_hitplot_{pred_source}.csv",
        index=False,
    )
    _make_pr7f_csv(ref_pct).to_csv(
        sub / f"sub-{subject_id}_hitplot_{ref_source}.csv",
        index=False,
    )


def test_load_pair_records_happy_path(tmp_path: Path) -> None:
    _write_subject_pair(
        tmp_path,
        "0005",
        pred_source="dl",
        ref_source="gt",
        pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        ref_pct={1: {"WT": 48.0, "TC": 28.0, "ET": 9.0}},
    )

    pairs, problems = mod.load_pair_records(
        tmp_path,
        ["0005"],
        prediction="dl",
        reference="gt",
    )
    assert problems == []
    assert len(pairs) == 3
    assert set(pairs["compartment"]) == {"WT", "TC", "ET"}
    assert set(pairs.columns) >= {
        "subject_id",
        "label",
        "compartment",
        "pct_of_parcel_dl",
        "pct_of_parcel_gt",
        "overlap_volume_mm3_dl",
        "overlap_volume_mm3_gt",
        "diff_pct_of_parcel",
        "mean_pct_of_parcel",
        "abs_diff_volume_mm3",
    }


def test_load_pair_records_diff_and_mean_columns_correct(tmp_path: Path) -> None:
    _write_subject_pair(
        tmp_path,
        "0005",
        pred_source="dl",
        ref_source="gt",
        pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        ref_pct={1: {"WT": 48.0, "TC": 32.0, "ET": 12.0}},
    )
    pairs, _ = mod.load_pair_records(tmp_path, ["0005"], prediction="dl", reference="gt")
    wt = pairs[pairs["compartment"] == "WT"].iloc[0]
    assert wt["diff_pct_of_parcel"] == pytest.approx(2.0)
    assert wt["mean_pct_of_parcel"] == pytest.approx(49.0)


def test_load_pair_records_missing_csv_records_problem(tmp_path: Path) -> None:
    pairs, problems = mod.load_pair_records(
        tmp_path,
        ["0005"],
        prediction="dl",
        reference="gt",
    )
    assert pairs.empty
    assert len(problems) == 1
    assert problems[0][0] == "0005"
    assert "not found" in problems[0][1]


def test_load_pair_records_strict_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_pair_records(
            tmp_path,
            ["0005"],
            prediction="dl",
            reference="gt",
            strict=True,
        )


def test_load_pair_records_malformed_csv_records_problem(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0005" / "hitplot"
    sub.mkdir(parents=True, exist_ok=True)
    # Pred CSV missing required columns.
    pd.DataFrame({"foo": [1, 2]}).to_csv(sub / "sub-0005_hitplot_dl.csv", index=False)
    _make_pr7f_csv({1: {"WT": 10.0, "TC": 10.0, "ET": 5.0}}).to_csv(
        sub / "sub-0005_hitplot_gt.csv",
        index=False,
    )

    pairs, problems = mod.load_pair_records(
        tmp_path,
        ["0005"],
        prediction="dl",
        reference="gt",
    )
    assert pairs.empty
    assert len(problems) == 1
    assert "missing required columns" in problems[0][1]


def test_load_pair_records_strict_raises_on_malformed(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0005" / "hitplot"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"foo": [1, 2]}).to_csv(sub / "sub-0005_hitplot_dl.csv", index=False)
    _make_pr7f_csv({1: {"WT": 10.0, "TC": 10.0, "ET": 5.0}}).to_csv(
        sub / "sub-0005_hitplot_gt.csv",
        index=False,
    )
    with pytest.raises(mod.HitplotCsvError):
        mod.load_pair_records(
            tmp_path,
            ["0005"],
            prediction="dl",
            reference="gt",
            strict=True,
        )


def test_load_pair_records_drops_nan_rows(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0005" / "hitplot"
    sub.mkdir(parents=True, exist_ok=True)
    pred_df = _make_pr7f_csv({1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}})
    ref_df = _make_pr7f_csv({1: {"WT": 48.0, "TC": 28.0, "ET": 9.0}})
    # Inject a NaN in the prediction's TC pct.
    pred_df.loc[
        (pred_df["label"] == 1) & (pred_df["compartment"] == "TC"),
        "pct_of_parcel",
    ] = np.nan
    pred_df.to_csv(sub / "sub-0005_hitplot_dl.csv", index=False)
    ref_df.to_csv(sub / "sub-0005_hitplot_gt.csv", index=False)

    pairs, problems = mod.load_pair_records(
        tmp_path,
        ["0005"],
        prediction="dl",
        reference="gt",
    )
    assert problems == []
    assert len(pairs) == 2
    assert "TC" not in set(pairs["compartment"])


def test_load_pair_records_inner_join_on_shared_labels_only(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0005" / "hitplot"
    sub.mkdir(parents=True, exist_ok=True)
    _make_pr7f_csv(
        {
            1: {"WT": 50.0, "TC": 30.0, "ET": 10.0},
            2: {"WT": 40.0, "TC": 25.0, "ET": 8.0},
        }
    ).to_csv(sub / "sub-0005_hitplot_dl.csv", index=False)
    _make_pr7f_csv(
        {
            # No label 2 in the reference -> dropped from the join.
            1: {"WT": 48.0, "TC": 32.0, "ET": 12.0},
            3: {"WT": 60.0, "TC": 50.0, "ET": 30.0},
        }
    ).to_csv(sub / "sub-0005_hitplot_gt.csv", index=False)

    pairs, problems = mod.load_pair_records(
        tmp_path,
        ["0005"],
        prediction="dl",
        reference="gt",
    )
    assert problems == []
    assert set(pairs["label"]) == {1}


def test_load_pair_records_multiple_subjects_concatenated(tmp_path: Path) -> None:
    for sid, off in [("0005", 0.0), ("0026", 5.0), ("0068", -2.0)]:
        _write_subject_pair(
            tmp_path,
            sid,
            pred_source="dl",
            ref_source="gt",
            pred_pct={1: {"WT": 50.0 + off, "TC": 30.0, "ET": 10.0}},
            ref_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        )

    pairs, _ = mod.load_pair_records(
        tmp_path,
        ["0005", "0026", "0068"],
        prediction="dl",
        reference="gt",
    )
    assert set(pairs["subject_id"]) == {"0005", "0026", "0068"}
    assert len(pairs) == 9  # 3 subjects x 3 compartments


# ---------------------------------------------------------------------------
# summarise_per_compartment: invariants
# ---------------------------------------------------------------------------


def _pairs_from_arrays(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    compartment: str = "WT",
    voxel_volume_mm3: float = 1.0,
) -> pd.DataFrame:
    """Build a `pairs` frame of length len(pred) for one compartment."""
    n = len(pred)
    return pd.DataFrame(
        {
            "subject_id": [f"{i:04d}" for i in range(n)],
            "label": np.arange(n),
            "compartment": [compartment] * n,
            "pct_of_parcel_dl": pred.astype(float),
            "pct_of_parcel_gt": ref.astype(float),
            "overlap_volume_mm3_dl": pred.astype(float) * voxel_volume_mm3,
            "overlap_volume_mm3_gt": ref.astype(float) * voxel_volume_mm3,
            "diff_pct_of_parcel": pred.astype(float) - ref.astype(float),
            "mean_pct_of_parcel": 0.5 * (pred + ref).astype(float),
            "abs_diff_volume_mm3": np.abs(pred - ref).astype(float) * voxel_volume_mm3,
        },
    )


def test_summarise_identical_inputs_yields_perfect_agreement() -> None:
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    pairs = _pairs_from_arrays(x, x.copy())
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    cell = summary["WT"]
    assert cell["ccc"] == pytest.approx(1.0)
    assert cell["mean_diff_pct_of_parcel"] == pytest.approx(0.0)
    assert cell["std_diff_pct_of_parcel"] == pytest.approx(0.0)
    assert cell["loa_lower_pct_of_parcel"] == pytest.approx(0.0)
    assert cell["loa_upper_pct_of_parcel"] == pytest.approx(0.0)
    assert cell["mean_abs_diff_volume_mm3"] == pytest.approx(0.0)
    assert cell["n_pairs"] == 5
    assert cell["n_subjects"] == 5
    assert cell["n_regions"] == 5


def test_summarise_mirror_image_yields_negative_one_ccc() -> None:
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = 2 * np.mean(x) - x  # CCC == -1.0 by construction
    pairs = _pairs_from_arrays(x, y)
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    assert summary["WT"]["ccc"] == pytest.approx(-1.0)


def test_summarise_constant_offset_reduces_ccc_below_one() -> None:
    """CCC penalises constant bias (Pearson would still be 1.0)."""
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = x + 10.0
    pairs = _pairs_from_arrays(x, y)
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    assert summary["WT"]["ccc"] is not None
    assert summary["WT"]["ccc"] < 1.0
    # diff = prediction - reference = x - (x + 10) = -10
    assert summary["WT"]["mean_diff_pct_of_parcel"] == pytest.approx(-10.0)


def test_summarise_bland_altman_against_closed_form() -> None:
    rng = np.random.default_rng(20260417)
    x = rng.uniform(0.0, 100.0, size=64)
    y = x + rng.normal(loc=0.5, scale=2.0, size=64)
    pairs = _pairs_from_arrays(x, y)
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    cell = summary["WT"]
    diff = x - y
    expected_mean = float(np.mean(diff))
    expected_std = float(np.std(diff, ddof=1))  # ddof=1 sample std
    assert cell["mean_diff_pct_of_parcel"] == pytest.approx(expected_mean)
    assert cell["std_diff_pct_of_parcel"] == pytest.approx(expected_std)
    assert cell["loa_lower_pct_of_parcel"] == pytest.approx(
        expected_mean - 1.96 * expected_std,
    )
    assert cell["loa_upper_pct_of_parcel"] == pytest.approx(
        expected_mean + 1.96 * expected_std,
    )


def test_summarise_ccc_matches_metrics_module_helper() -> None:
    """Cross-check against ``hpgs.metrics.concordance_correlation_coefficient``
    on the same fixture --- guards against a future refactor that would
    re-implement the CCC inline (we want the canonical helper)."""
    rng = np.random.default_rng(20260417)
    x = rng.uniform(0.0, 100.0, size=32)
    y = x * 0.9 + rng.normal(scale=3.0, size=32) + 1.0
    pairs = _pairs_from_arrays(x, y)
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    expected = concordance_correlation_coefficient(x, y)
    assert summary["WT"]["ccc"] == pytest.approx(expected)


def test_summarise_mean_abs_diff_volume_matches_closed_form() -> None:
    pred = np.array([10.0, 20.0, 30.0])
    ref = np.array([12.0, 18.0, 33.0])
    pairs = _pairs_from_arrays(pred, ref, voxel_volume_mm3=2.5)
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    expected = float(np.mean(np.abs(pred - ref)) * 2.5)
    assert summary["WT"]["mean_abs_diff_volume_mm3"] == pytest.approx(expected)


def test_summarise_empty_compartment_returns_json_safe_nulls() -> None:
    pairs = _pairs_from_arrays(np.array([1.0, 2.0]), np.array([1.0, 2.0]))  # only WT
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    for comp in ("TC", "ET"):
        cell = summary[comp]
        assert cell["n_pairs"] == 0
        assert cell["ccc"] is None
        assert cell["mean_diff_pct_of_parcel"] is None
        assert cell["mean_abs_diff_volume_mm3"] is None


def test_summarise_singleton_pair_keeps_mean_drops_std() -> None:
    pairs = _pairs_from_arrays(np.array([42.0]), np.array([41.0]))
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    cell = summary["WT"]
    assert cell["mean_diff_pct_of_parcel"] == pytest.approx(1.0)
    # ddof=1 std of a single sample is undefined -> JSON-safe None.
    assert cell["std_diff_pct_of_parcel"] is None
    assert cell["loa_lower_pct_of_parcel"] is None
    assert cell["loa_upper_pct_of_parcel"] is None


def test_summary_n_subjects_and_n_regions_distinct_from_n_pairs() -> None:
    pairs = pd.concat(
        [
            _pairs_from_arrays(np.array([10.0, 20.0, 30.0]), np.array([10.0, 20.0, 30.0])).assign(
                subject_id="0005",
            ),
            _pairs_from_arrays(np.array([15.0, 25.0, 35.0]), np.array([15.0, 25.0, 35.0])).assign(
                subject_id="0026",
            ),
        ],
        ignore_index=True,
    )
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    cell = summary["WT"]
    assert cell["n_pairs"] == 6
    assert cell["n_subjects"] == 2
    assert cell["n_regions"] == 3


def test_json_safe_passes_finite() -> None:
    assert mod._json_safe(0.0) == 0.0
    assert mod._json_safe(-1.5) == pytest.approx(-1.5)


def test_json_safe_returns_none_for_nan_and_inf() -> None:
    assert mod._json_safe(float("nan")) is None
    assert mod._json_safe(float("inf")) is None
    assert mod._json_safe(-float("inf")) is None
    assert mod._json_safe(None) is None


# ---------------------------------------------------------------------------
# render_panel_pdf
# ---------------------------------------------------------------------------


def test_render_panel_pdf_writes_non_empty_file(tmp_path: Path) -> None:
    pairs = _pairs_from_arrays(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    out_pdf = tmp_path / "fig.pdf"
    mod.render_panel_pdf(pairs, summary, out_pdf, prediction="dl", reference="gt")
    assert out_pdf.is_file()
    assert out_pdf.stat().st_size > 1000  # not a stub


def test_render_panel_pdf_creates_parent_dir(tmp_path: Path) -> None:
    pairs = _pairs_from_arrays(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    out_pdf = tmp_path / "deep" / "nested" / "fig.pdf"
    mod.render_panel_pdf(pairs, summary, out_pdf, prediction="dl", reference="gt")
    assert out_pdf.is_file()


def test_render_panel_pdf_handles_empty_compartments(tmp_path: Path) -> None:
    """The 'no data' panel layout must not crash matplotlib on 0-row inputs."""
    pairs = mod._empty_pairs_frame("dl", "gt")
    summary = mod.summarise_per_compartment(pairs, prediction="dl", reference="gt")
    out_pdf = tmp_path / "fig.pdf"
    mod.render_panel_pdf(pairs, summary, out_pdf, prediction="dl", reference="gt")
    assert out_pdf.is_file()


# ---------------------------------------------------------------------------
# CLI orchestration: main() exit codes
# ---------------------------------------------------------------------------


def _fake_cohort_metadata(subject_ids: list[str]):
    class _FakeCM:
        def __init__(self, sids: list[str]) -> None:
            self.df = pd.DataFrame(index=sids)
            self.n = len(sids)
            self.seed = 0

    return _FakeCM(subject_ids)


def test_main_returns_0_on_clean_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hitplot_root = tmp_path / "deriv"
    for sid in ("0005", "0026", "0068"):
        _write_subject_pair(
            hitplot_root,
            sid,
            pred_source="dl",
            ref_source="gt",
            pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
            ref_pct={1: {"WT": 48.0, "TC": 32.0, "ET": 9.0}},
        )

    cm = _fake_cohort_metadata(["0005", "0026", "0068"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    out_pdf = tmp_path / "out.pdf"
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(out_pdf),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert out_pdf.is_file()
    assert out_json.is_file()
    assert out_csv.is_file()


def test_main_returns_1_when_a_compartment_has_no_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hitplot_root = tmp_path / "deriv"
    # Only WT data; TC + ET will be empty -> rc 1.
    _write_subject_pair(
        hitplot_root,
        "0005",
        pred_source="dl",
        ref_source="gt",
        pred_pct={1: {"WT": 50.0}},
        ref_pct={1: {"WT": 48.0}},
    )

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    assert mod.main() == 1


def test_main_returns_2_when_cohort_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a, **_kw):
        raise ValueError("boom")

    monkeypatch.setattr(mod, "load_cohort_metadata", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(tmp_path),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    assert mod.main() == 2


def test_main_returns_2_when_hitplot_root_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(tmp_path / "no-such-root"),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    assert mod.main() == 2


def test_main_returns_2_when_prediction_equals_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--prediction",
            "dl",
            "--reference",
            "dl",
            "--hitplot-root",
            str(tmp_path),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    assert mod.main() == 2


def test_main_returns_3_under_strict_when_csv_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hitplot_root = tmp_path / "deriv"
    hitplot_root.mkdir()
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--strict",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    assert mod.main() == 3


def test_main_writes_json_with_expected_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hitplot_root = tmp_path / "deriv"
    _write_subject_pair(
        hitplot_root,
        "0005",
        pred_source="dl",
        ref_source="gt",
        pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        ref_pct={1: {"WT": 48.0, "TC": 32.0, "ET": 9.0}},
    )
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(tmp_path / "o.csv"),
        ],
    )
    mod.main()
    doc = json.loads(out_json.read_text())
    assert doc["schema_version"] == "1.0"
    assert doc["comparison"] == {"prediction": "dl", "reference": "gt"}
    assert set(doc["compartments"]) == {"WT", "TC", "ET"}
    assert "summary" in doc
    for comp in ("WT", "TC", "ET"):
        keys = set(doc["summary"][comp])
        assert {
            "ccc",
            "mean_diff_pct_of_parcel",
            "std_diff_pct_of_parcel",
            "loa_lower_pct_of_parcel",
            "loa_upper_pct_of_parcel",
            "mean_abs_diff_volume_mm3",
            "n_pairs",
            "n_subjects",
            "n_regions",
        }.issubset(keys)


def test_main_audit_csv_is_long_format_and_round_trippable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hitplot_root = tmp_path / "deriv"
    _write_subject_pair(
        hitplot_root,
        "0005",
        pred_source="dl",
        ref_source="gt",
        pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        ref_pct={1: {"WT": 48.0, "TC": 32.0, "ET": 9.0}},
    )
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-csv",
            str(out_csv),
        ],
    )
    mod.main()
    df = pd.read_csv(out_csv)
    assert {
        "subject_id",
        "label",
        "compartment",
        "pct_of_parcel_dl",
        "pct_of_parcel_gt",
        "diff_pct_of_parcel",
        "abs_diff_volume_mm3",
    }.issubset(df.columns)
    assert len(df) == 3


def test_main_alternative_comparison_dl_vs_raidionics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same producer powers any (pred, ref) combination from the
    accepted-source set --- the column suffix machinery must propagate."""
    hitplot_root = tmp_path / "deriv"
    _write_subject_pair(
        hitplot_root,
        "0005",
        pred_source="dl",
        ref_source="raidionics",
        pred_pct={1: {"WT": 50.0, "TC": 30.0, "ET": 10.0}},
        ref_pct={1: {"WT": 47.0, "TC": 28.0, "ET": 8.5}},
    )
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    out_csv = tmp_path / "out.csv"
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hitplot_agreement_panel.py",
            "--prediction",
            "dl",
            "--reference",
            "raidionics",
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(tmp_path / "o.pdf"),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
        ],
    )
    rc = mod.main()
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert "pct_of_parcel_dl" in df.columns
    assert "pct_of_parcel_raidionics" in df.columns
    doc = json.loads(out_json.read_text())
    assert doc["comparison"] == {"prediction": "dl", "reference": "raidionics"}


# ---------------------------------------------------------------------------
# Module-level constants (lock to scope_pr7.md § 3)
# ---------------------------------------------------------------------------


def test_compartments_locked_to_wt_tc_et() -> None:
    assert mod.COMPARTMENTS == ("WT", "TC", "ET")


def test_accepted_sources_match_pr7f_csv_families() -> None:
    assert mod.ACCEPTED_SOURCES == frozenset(
        {"dl", "gt", "raidionics", "segmentglioma", "tumorsynth"},
    )


def test_required_hitplot_columns_match_pr7f_schema() -> None:
    assert "pct_of_parcel" in mod.REQUIRED_HITPLOT_COLUMNS
    assert "overlap_voxels" in mod.REQUIRED_HITPLOT_COLUMNS
    assert "compartment" in mod.REQUIRED_HITPLOT_COLUMNS


def test_loa_k_is_1_96() -> None:
    assert mod.LOA_K == pytest.approx(1.96)
