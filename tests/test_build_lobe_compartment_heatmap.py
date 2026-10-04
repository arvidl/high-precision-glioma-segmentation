"""Unit tests for ``scripts/build_lobe_compartment_heatmap.py`` (PR-7l).

Loaded via ``importlib`` because ``scripts/`` is intentionally not on
``sys.path``. Coverage rubric mirrors the PR-7i (agreement panel) and
PR-7j (legacy-5 panel) producer tests:

* CSV path layout + LUT loader (happy / missing / malformed);
* per-subject aggregator: lobe binning, voxel-volume reconstruction,
  share_of_compartment_pct + share_of_lobe_pct identities;
* cohort summary: empty / single-subject / multi-subject median +
  prevalence semantics;
* PDF / JSON / CSV artefact contract;
* CLI orchestration + exit codes 0 / 1 / 2 / 3.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from hpgs.parcellate.lobes import LOBES


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_lobe_compartment_heatmap.py"
    spec = importlib.util.spec_from_file_location(
        "build_lobe_compartment_heatmap",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


COMPARTMENTS = mod.COMPARTMENTS

# A miniature LUT covering one cortical region per cortical lobe + one
# sub-cortical structure per sub-cortical bucket. Label ids are taken
# from FreeSurferColorLUT verbatim so the test doubles as a regression
# guard for the wmparc id range.
MINI_LUT: dict[int, str] = {
    1028: "ctx-lh-superiorfrontal",  # frontal
    1010: "ctx-lh-isthmuscingulate",  # cingulate
    1035: "ctx-lh-insula",  # insula
    1029: "ctx-lh-superiorparietal",  # parietal
    1030: "ctx-lh-superiortemporal",  # temporal
    1011: "ctx-lh-lateraloccipital",  # occipital
    10: "Left-Thalamus",  # subcortical-deep
    17: "Left-Hippocampus",  # subcortical-limbic
    8: "Left-Cerebellum-Cortex",  # cerebellum
    16: "Brain-Stem",  # brainstem
    4: "Left-Lateral-Ventricle",  # other
}


def _make_hitplot_csv(
    overlap_voxels_per_label_comp: dict[int, dict[str, int]],
    *,
    parcel_voxels: int = 1000,
    voxel_volume_mm3: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic PR-7f Hit-Plot CSV in memory.

    All labels share the same ``parcel_voxels`` so lobe volume = (#labels in
    lobe) * parcel_voxels; this makes hand-checking shares trivial.
    """
    rows: list[dict] = []
    for lab, comp_to_overlap in overlap_voxels_per_label_comp.items():
        for comp in COMPARTMENTS:
            n_overlap = int(comp_to_overlap.get(comp, 0))
            rows.append(
                {
                    "label": lab,
                    "name": MINI_LUT.get(lab, "Unknown"),
                    "compartment": comp,
                    "parcel_voxels": parcel_voxels,
                    "compartment_voxels": parcel_voxels * 5,
                    "overlap_voxels": n_overlap,
                    "overlap_volume_mm3": float(n_overlap * voxel_volume_mm3),
                    "pct_of_parcel": 100.0 * n_overlap / parcel_voxels,
                    "pct_of_compartment": 0.0,  # unused by this producer
                },
            )
    return pd.DataFrame(rows)


def _write_subject(
    hitplot_root: Path,
    subject_id: str,
    *,
    source: str,
    overlap_voxels_per_label_comp: dict[int, dict[str, int]],
    lut: dict[int, str] | None = None,
    voxel_volume_mm3: float = 1.0,
) -> None:
    sub = hitplot_root / f"sub-{subject_id}"
    (sub / "hitplot").mkdir(parents=True, exist_ok=True)
    (sub / "parcellation").mkdir(parents=True, exist_ok=True)
    df = _make_hitplot_csv(
        overlap_voxels_per_label_comp,
        voxel_volume_mm3=voxel_volume_mm3,
    )
    df.to_csv(
        sub / "hitplot" / f"sub-{subject_id}_hitplot_{source}.csv",
        index=False,
    )
    (sub / "parcellation" / f"sub-{subject_id}_wmparc_lut.json").write_text(
        json.dumps({str(k): v for k, v in (lut or MINI_LUT).items()}),
    )


# ---------------------------------------------------------------------------
# Path + LUT loader
# ---------------------------------------------------------------------------


def test_hitplot_csv_path_layout(tmp_path: Path) -> None:
    p = mod._hitplot_csv_path(tmp_path / "deriv", "0005", "gt")
    assert p == tmp_path / "deriv" / "sub-0005" / "hitplot" / "sub-0005_hitplot_gt.csv"


def test_hitplot_csv_path_normalises_subject_id(tmp_path: Path) -> None:
    a = mod._hitplot_csv_path(tmp_path, "sub-0005", "gt")
    b = mod._hitplot_csv_path(tmp_path, "0005", "gt")
    assert a == b


def test_lut_json_path_layout(tmp_path: Path) -> None:
    p = mod._lut_json_path(tmp_path / "deriv", "0005")
    expected = tmp_path / "deriv" / "sub-0005" / "parcellation" / "sub-0005_wmparc_lut.json"
    assert p == expected


def test_load_lut_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "lut.json"
    p.write_text(json.dumps({"1001": "ctx-lh-superiorfrontal", "17": "Left-Hippocampus"}))
    lut = mod._load_lut(p)
    assert lut == {1001: "ctx-lh-superiorfrontal", 17: "Left-Hippocampus"}


def test_load_lut_missing_file(tmp_path: Path) -> None:
    with pytest.raises(mod.LutError, match="not found"):
        mod._load_lut(tmp_path / "missing.json")


def test_load_lut_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "lut.json"
    p.write_text("{not json}")
    with pytest.raises(mod.LutError, match="invalid JSON"):
        mod._load_lut(p)


def test_load_lut_root_is_not_object(tmp_path: Path) -> None:
    p = tmp_path / "lut.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(mod.LutError, match="not an object"):
        mod._load_lut(p)


def test_load_lut_bad_label_key(tmp_path: Path) -> None:
    p = tmp_path / "lut.json"
    p.write_text(json.dumps({"not-a-number": "ctx-lh-x"}))
    with pytest.raises(mod.LutError, match="bad label key"):
        mod._load_lut(p)


# ---------------------------------------------------------------------------
# Hitplot CSV schema validation
# ---------------------------------------------------------------------------


def test_validate_hitplot_csv_accepts_canonical_schema() -> None:
    df = _make_hitplot_csv({1028: {"WT": 100, "TC": 50, "ET": 20}})
    mod._validate_hitplot_csv(df, source="phantom.csv")


def test_validate_hitplot_csv_rejects_missing_required_column() -> None:
    df = _make_hitplot_csv({1028: {"WT": 100, "TC": 50, "ET": 20}})
    df = df.drop(columns=["overlap_voxels"])
    with pytest.raises(mod.HitplotCsvError, match="missing required columns"):
        mod._validate_hitplot_csv(df, source="phantom.csv")


def test_validate_hitplot_csv_rejects_unknown_compartment() -> None:
    df = _make_hitplot_csv({1028: {"WT": 100, "TC": 50, "ET": 20}})
    df.loc[0, "compartment"] = "BOGUS"
    with pytest.raises(mod.HitplotCsvError, match="unexpected compartment"):
        mod._validate_hitplot_csv(df, source="phantom.csv")


# ---------------------------------------------------------------------------
# aggregate_subject
# ---------------------------------------------------------------------------


def test_aggregate_subject_collapses_label_to_lobe() -> None:
    # All overlap goes to the frontal lobe.
    df = _make_hitplot_csv(
        {
            1028: {"WT": 200, "TC": 100, "ET": 50},  # frontal
            1029: {"WT": 0, "TC": 0, "ET": 0},  # parietal
        },
        parcel_voxels=1000,
        voxel_volume_mm3=2.0,
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    # One row per (lobe, compartment); LOBES x COMPARTMENTS = 33.
    assert len(cells) == len(LOBES) * len(COMPARTMENTS)
    # Frontal WT cell carries all the overlap.
    frontal_wt = cells[(cells["lobe"] == "frontal") & (cells["compartment"] == "WT")]
    assert len(frontal_wt) == 1
    assert frontal_wt["overlap_volume_mm3"].iloc[0] == pytest.approx(200 * 2.0)
    # share_of_compartment_pct for frontal/WT must be 100 (no other lobe touched).
    assert frontal_wt["share_of_compartment_pct"].iloc[0] == pytest.approx(100.0)


def test_aggregate_subject_share_sums_to_100_per_compartment() -> None:
    # Spread overlap across two lobes with the same total per compartment.
    df = _make_hitplot_csv(
        {
            1028: {"WT": 300, "TC": 100, "ET": 30},  # frontal
            1029: {"WT": 100, "TC": 100, "ET": 10},  # parietal
        },
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    for comp in COMPARTMENTS:
        sub = cells[cells["compartment"] == comp]
        total_share = sub["share_of_compartment_pct"].dropna().sum()
        assert total_share == pytest.approx(100.0)


def test_aggregate_subject_share_of_lobe_pct_identity() -> None:
    df = _make_hitplot_csv(
        {1028: {"WT": 250, "TC": 0, "ET": 0}},
        parcel_voxels=1000,
        voxel_volume_mm3=1.0,
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    frontal_wt = cells[(cells["lobe"] == "frontal") & (cells["compartment"] == "WT")]
    # parcel_voxels=1000, voxel_volume=1.0, so lobe_volume_mm3 = 1000.
    # overlap_volume_mm3 = 250 -> share_of_lobe_pct = 25.0
    assert frontal_wt["share_of_lobe_pct"].iloc[0] == pytest.approx(25.0)


def test_aggregate_subject_empty_compartment_yields_nan_share() -> None:
    # All overlap voxels are zero -> compartment denominator is zero ->
    # share_of_compartment_pct must be NaN everywhere (not 0/0).
    df = _make_hitplot_csv({1028: {"WT": 0, "TC": 0, "ET": 0}})
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    assert cells["share_of_compartment_pct"].isna().all()


def test_aggregate_subject_handles_subcortical_labels() -> None:
    df = _make_hitplot_csv(
        {
            10: {"WT": 80, "TC": 80, "ET": 40},  # Left-Thalamus -> subcortical-deep
            17: {"WT": 20, "TC": 20, "ET": 10},  # Left-Hippocampus -> subcortical-limbic
        },
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    deep_wt = cells[(cells["lobe"] == "subcortical-deep") & (cells["compartment"] == "WT")]
    limbic_wt = cells[(cells["lobe"] == "subcortical-limbic") & (cells["compartment"] == "WT")]
    assert deep_wt["overlap_volume_mm3"].iloc[0] == pytest.approx(80.0)
    assert limbic_wt["overlap_volume_mm3"].iloc[0] == pytest.approx(20.0)


def test_aggregate_subject_voxel_volume_inferred_from_csv() -> None:
    # voxel_volume_mm3 = 8 (e.g. 2x2x2 mm voxels): the producer should
    # back-compute it from overlap_volume_mm3 / overlap_voxels.
    df = _make_hitplot_csv(
        {1028: {"WT": 100, "TC": 0, "ET": 0}},
        parcel_voxels=1000,
        voxel_volume_mm3=8.0,
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT)
    frontal_wt = cells[(cells["lobe"] == "frontal") & (cells["compartment"] == "WT")]
    # parcel_voxels=1000, voxel_volume=8 -> lobe_volume_mm3 = 8000.
    # share_of_lobe_pct = 100 * (100*8) / 8000 = 10.0.
    assert frontal_wt["share_of_lobe_pct"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# load_cohort_records
# ---------------------------------------------------------------------------


def test_load_cohort_records_happy_path(tmp_path: Path) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 50, "ET": 20}},
    )
    _write_subject(
        tmp_path,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1029: {"WT": 200, "TC": 100, "ET": 40}},
    )
    audit, problems = mod.load_cohort_records(
        tmp_path,
        ["0005", "0006"],
        source="gt",
    )
    assert problems == []
    assert audit["subject_id"].nunique() == 2
    # Per subject, len(LOBES) x len(COMPARTMENTS) cells -> 2 * 33 = 66 rows.
    assert len(audit) == 2 * len(LOBES) * len(COMPARTMENTS)


def test_load_cohort_records_missing_csv_recorded_as_problem(
    tmp_path: Path,
) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 50, "ET": 20}},
    )
    audit, problems = mod.load_cohort_records(
        tmp_path,
        ["0005", "0007"],
        source="gt",
    )
    # 0007 is missing -> reported but does not abort the run.
    assert audit["subject_id"].nunique() == 1
    assert len(problems) == 1
    assert problems[0][0] == "0007"
    assert "not found" in problems[0][1]


def test_load_cohort_records_missing_lut_recorded_as_problem(
    tmp_path: Path,
) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 50, "ET": 20}},
    )
    # Delete the LUT after writing.
    (tmp_path / "sub-0005" / "parcellation" / "sub-0005_wmparc_lut.json").unlink()
    audit, problems = mod.load_cohort_records(
        tmp_path,
        ["0005"],
        source="gt",
    )
    assert audit.empty
    assert len(problems) == 1
    assert "wmparc LUT" in problems[0][1]


def test_load_cohort_records_strict_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_cohort_records(
            tmp_path,
            ["0005"],
            source="gt",
            strict=True,
        )


# ---------------------------------------------------------------------------
# summarise_cells
# ---------------------------------------------------------------------------


def test_summarise_cells_empty_audit_returns_full_grid() -> None:
    summary = mod.summarise_cells(pd.DataFrame(), n_total=50)
    assert len(summary) == len(LOBES) * len(COMPARTMENTS)
    for cell in summary:
        assert cell["n_with_overlap"] == 0
        assert cell["n_total"] == 50
        assert cell["prevalence"] == 0.0
        assert cell["share_of_compartment_pct_median"] is None


def test_summarise_cells_single_subject_median_equals_value(tmp_path: Path) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 300, "TC": 0, "ET": 0}},
    )
    audit, _ = mod.load_cohort_records(tmp_path, ["0005"], source="gt")
    summary = mod.summarise_cells(audit, n_total=1)
    by = {(c["lobe"], c["compartment"]): c for c in summary}
    frontal_wt = by[("frontal", "WT")]
    # Single subject: 100 % of WT volume in frontal lobe.
    assert frontal_wt["share_of_compartment_pct_median"] == pytest.approx(100.0)
    assert frontal_wt["n_with_overlap"] == 1
    assert frontal_wt["prevalence"] == pytest.approx(1.0)
    # All other lobes for WT: 0 % share, no overlap.
    for lobe in LOBES:
        if lobe == "frontal":
            continue
        cell = by[(lobe, "WT")]
        assert cell["n_with_overlap"] == 0
        assert cell["share_of_compartment_pct_median"] == pytest.approx(0.0)


def test_summarise_cells_prevalence_two_subjects(tmp_path: Path) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 0, "ET": 0}},
    )
    _write_subject(
        tmp_path,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1029: {"WT": 100, "TC": 0, "ET": 0}},
    )
    audit, _ = mod.load_cohort_records(tmp_path, ["0005", "0006"], source="gt")
    summary = mod.summarise_cells(audit, n_total=2)
    by = {(c["lobe"], c["compartment"]): c for c in summary}
    # Each lobe has overlap in 1/2 subjects -> prevalence = 0.5.
    assert by[("frontal", "WT")]["prevalence"] == pytest.approx(0.5)
    assert by[("parietal", "WT")]["prevalence"] == pytest.approx(0.5)
    assert by[("temporal", "WT")]["prevalence"] == pytest.approx(0.0)


def test_summarise_cells_share_of_lobe_when_present_skips_zeros(
    tmp_path: Path,
) -> None:
    # Two subjects: only one has frontal overlap. The when-present
    # quantile must be over the one non-zero subject only, not 0.5 * value.
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 0, "ET": 0}},
    )
    _write_subject(
        tmp_path,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1029: {"WT": 100, "TC": 0, "ET": 0}},
    )
    audit, _ = mod.load_cohort_records(tmp_path, ["0005", "0006"], source="gt")
    summary = mod.summarise_cells(audit, n_total=2)
    by = {(c["lobe"], c["compartment"]): c for c in summary}
    frontal_wt = by[("frontal", "WT")]
    # 0005's frontal lobe = 1 label * 1000 vox = 1000 mm^3; overlap = 100 mm^3
    # -> share_of_lobe_pct = 10.0
    assert frontal_wt["share_of_lobe_pct_median_when_present"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# render_heatmap_pdf
# ---------------------------------------------------------------------------


def test_render_heatmap_pdf_writes_file(tmp_path: Path) -> None:
    summary = mod.summarise_cells(pd.DataFrame(), n_total=50)
    out_pdf = tmp_path / "fig.pdf"
    mod.render_heatmap_pdf(summary, out_pdf, source="gt", n_total=50)
    assert out_pdf.is_file()
    # Sanity-check the file actually starts with a PDF magic header.
    assert out_pdf.read_bytes().startswith(b"%PDF")


def test_render_heatmap_pdf_with_real_data(tmp_path: Path) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 50, "ET": 20}},
    )
    audit, _ = mod.load_cohort_records(tmp_path, ["0005"], source="gt")
    summary = mod.summarise_cells(audit, n_total=1)
    out_pdf = tmp_path / "fig.pdf"
    mod.render_heatmap_pdf(summary, out_pdf, source="gt", n_total=1)
    assert out_pdf.is_file() and out_pdf.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _write_minimal_cohort(tmp_path: Path, subject_ids: list[str]) -> tuple[Path, Path]:
    """Write a minimal cohort YAML + UCSF-PDGM-style metadata CSV.

    Schema mirrors ``configs/cohort_ucsfpdgm_n50.yaml`` and
    ``data/UCSF-PDGM-metadata_v5.csv`` minimally enough for
    :func:`hpgs.io.load_cohort_metadata` to accept the pair.
    """
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "metadata.csv"
    cohort_yaml.write_text(
        "seed: 20260415\n"
        f"n: {len(subject_ids)}\n"
        "name: test-cohort\n"
        "stratification:\n"
        "  os_tertile_cuts_days: [180, 540]\n"
        "subjects:\n" + "".join(f"  - '{sid}'\n" for sid in subject_ids),
    )
    pd.DataFrame(
        {
            "ID": [f"UCSF-PDGM-{sid}" for sid in subject_ids],
            "Sex": ["M"] * len(subject_ids),
            "OS": [365.0] * len(subject_ids),
        },
    ).to_csv(metadata_csv, index=False)
    return cohort_yaml, metadata_csv


def test_main_cli_happy_path(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(tmp_path, ["0005", "0006"])
    hitplot_root = tmp_path / "deriv"
    _write_subject(
        hitplot_root,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1028: {"WT": 100, "TC": 50, "ET": 20}},
    )
    _write_subject(
        hitplot_root,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1029: {"WT": 200, "TC": 100, "ET": 40}},
    )
    out_pdf = tmp_path / "out.pdf"
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lobe_compartment_heatmap.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--hitplot-root",
            str(hitplot_root),
            "--source",
            "gt",
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
    assert out_pdf.is_file() and out_pdf.stat().st_size > 0
    assert out_csv.is_file()
    summary_json = json.loads(out_json.read_text())
    assert summary_json["schema_version"] == "1.0"
    assert summary_json["n_total"] == 2
    assert summary_json["n_with_csv"] == 2
    assert summary_json["source"] == "gt"
    assert summary_json["compartments"] == list(COMPARTMENTS)
    assert summary_json["lobes"] == list(LOBES)


def test_main_cli_returns_2_when_cohort_yaml_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lobe_compartment_heatmap.py",
            "--cohort-yaml",
            str(tmp_path / "missing.yaml"),
            "--metadata-csv",
            str(tmp_path / "missing.csv"),
            "--hitplot-root",
            str(tmp_path / "deriv"),
        ],
    )
    rc = mod.main()
    assert rc == 2


def test_main_cli_returns_2_when_hitplot_root_missing(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(tmp_path, ["0005"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lobe_compartment_heatmap.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--hitplot-root",
            str(tmp_path / "missing-deriv"),
        ],
    )
    rc = mod.main()
    assert rc == 2


def test_main_cli_strict_returns_3_on_missing_csv(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(tmp_path, ["0005"])
    hitplot_root = tmp_path / "deriv"
    hitplot_root.mkdir()
    out_pdf = tmp_path / "out.pdf"
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lobe_compartment_heatmap.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--hitplot-root",
            str(hitplot_root),
            "--out-pdf",
            str(out_pdf),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--strict",
        ],
    )
    rc = mod.main()
    assert rc == 3
