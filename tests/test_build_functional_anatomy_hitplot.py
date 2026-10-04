"""Tests for ``scripts/build_functional_anatomy_hitplot.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_functional_anatomy_hitplot.py"
    spec = importlib.util.spec_from_file_location(
        "build_functional_anatomy_hitplot",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()

COMPARTMENTS = mod.COMPARTMENTS

MINI_LUT: dict[int, str] = {
    1024: "ctx-lh-precentral",
    2024: "ctx-rh-precentral",
    1018: "ctx-lh-parsopercularis",
    1020: "ctx-lh-parstriangularis",
    1030: "ctx-lh-superiortemporal",
    1031: "ctx-lh-supramarginal",
    2031: "ctx-rh-supramarginal",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    251: "CC_Anterior",
    1029: "ctx-lh-superiorparietal",
}


def _make_hitplot_csv(
    overlap_voxels_per_label_comp: dict[int, dict[str, int]],
    *,
    parcel_voxels: int = 1000,
    voxel_volume_mm3: float = 1.0,
) -> pd.DataFrame:
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
                    "pct_of_compartment": 0.0,
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
) -> None:
    sub = hitplot_root / f"sub-{subject_id}"
    (sub / "hitplot").mkdir(parents=True, exist_ok=True)
    (sub / "parcellation").mkdir(parents=True, exist_ok=True)
    _make_hitplot_csv(overlap_voxels_per_label_comp).to_csv(
        sub / "hitplot" / f"sub-{subject_id}_hitplot_{source}.csv",
        index=False,
    )
    (sub / "parcellation" / f"sub-{subject_id}_wmparc_lut.json").write_text(
        json.dumps({str(k): v for k, v in (lut or MINI_LUT).items()}),
    )


def _write_minimal_cohort(
    tmp_path: Path,
    subject_ids: list[str],
    os_days: list[float],
) -> tuple[Path, Path]:
    cohort_yaml = tmp_path / "cohort.yaml"
    metadata_csv = tmp_path / "metadata.csv"
    cohort_yaml.write_text(
        "seed: 20260415\n"
        f"n: {len(subject_ids)}\n"
        "stratification:\n"
        "  os_tertile_cuts_days: [180, 540]\n"
        "subjects:\n" + "".join(f"  - '{sid}'\n" for sid in subject_ids),
    )
    pd.DataFrame(
        {
            "ID": [f"UCSF-PDGM-{sid}" for sid in subject_ids],
            "Sex": ["M"] * len(subject_ids),
            "OS": os_days,
        },
    ).to_csv(metadata_csv, index=False)
    return cohort_yaml, metadata_csv


def test_group_ids_for_label_includes_lobe_and_motor() -> None:
    groups = mod.group_ids_for_label(1024, MINI_LUT)
    assert "lobe:frontal" in groups
    assert "functional:motor_adjacent" in groups


def test_group_ids_for_label_left_language_is_left_lateralized() -> None:
    left = mod.group_ids_for_label(1031, MINI_LUT)
    right = mod.group_ids_for_label(2031, MINI_LUT)
    assert "functional:left_language_adjacent" in left
    assert "functional:left_language_adjacent" not in right
    assert "functional:motor_adjacent" in left
    assert "functional:motor_adjacent" in right


def test_group_ids_for_label_deep_midline_includes_corpus_callosum() -> None:
    groups = mod.group_ids_for_label(251, MINI_LUT)
    assert "lobe:other" in groups
    assert "functional:deep_midline_commissural" in groups


def test_aggregate_subject_uses_original_compartment_denominator() -> None:
    df = _make_hitplot_csv(
        {
            1024: {"WT": 100},  # motor + lobe frontal
            1029: {"WT": 100},  # lobe parietal only
        },
    )
    cells = mod.aggregate_subject("0005", df, MINI_LUT, os_tertile="short")
    motor = cells[
        (cells["group_id"] == "functional:motor_adjacent") & (cells["compartment"] == "WT")
    ]
    frontal = cells[(cells["group_id"] == "lobe:frontal") & (cells["compartment"] == "WT")]
    parietal = cells[(cells["group_id"] == "lobe:parietal") & (cells["compartment"] == "WT")]
    assert motor["share_of_compartment_pct"].iloc[0] == pytest.approx(50.0)
    assert frontal["share_of_compartment_pct"].iloc[0] == pytest.approx(50.0)
    assert parietal["share_of_compartment_pct"].iloc[0] == pytest.approx(50.0)
    assert cells["OS_tertile"].unique().tolist() == ["short"]


def test_summarise_groups_includes_os_tertile_contexts(tmp_path: Path) -> None:
    _write_subject(
        tmp_path,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1024: {"WT": 100}},
    )
    _write_subject(
        tmp_path,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1029: {"WT": 100}},
    )
    audit, problems = mod.load_cohort_records(
        tmp_path,
        ["0005", "0006"],
        {"0005": "short", "0006": "long"},
        source="gt",
    )
    assert problems == []
    summary = mod.summarise_groups(
        audit,
        n_total_by_context={"all": 2, "short": 1, "mid": 0, "long": 1},
        contexts=("all", "short", "mid", "long"),
    )
    by = {(c["context"], c["group_id"], c["compartment"]): c for c in summary}
    assert by[("all", "functional:motor_adjacent", "WT")]["prevalence"] == pytest.approx(0.5)
    assert by[("short", "functional:motor_adjacent", "WT")]["prevalence"] == pytest.approx(1.0)
    assert by[("long", "functional:motor_adjacent", "WT")]["prevalence"] == pytest.approx(0.0)


def test_render_functional_heatmap_pdf_writes_file(tmp_path: Path) -> None:
    summary = mod.summarise_groups(
        pd.DataFrame(columns=mod._audit_columns()),
        n_total_by_context={"all": 1},
        contexts=("all",),
    )
    out_pdf = tmp_path / "fig.pdf"
    mod.render_functional_heatmap_pdf(summary, out_pdf, source="gt", contexts=("all",))
    assert out_pdf.is_file()
    assert out_pdf.read_bytes().startswith(b"%PDF")


def test_main_cli_happy_path_with_os_panels(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(
        tmp_path,
        ["0005", "0006", "0007"],
        [100.0, 365.0, 800.0],
    )
    hitplot_root = tmp_path / "deriv"
    _write_subject(
        hitplot_root,
        "0005",
        source="gt",
        overlap_voxels_per_label_comp={1024: {"WT": 100, "TC": 50, "ET": 20}},
    )
    _write_subject(
        hitplot_root,
        "0006",
        source="gt",
        overlap_voxels_per_label_comp={1031: {"WT": 100, "TC": 50, "ET": 20}},
    )
    _write_subject(
        hitplot_root,
        "0007",
        source="gt",
        overlap_voxels_per_label_comp={251: {"WT": 100, "TC": 50, "ET": 20}},
    )
    out_pdf = tmp_path / "out.pdf"
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_functional_anatomy_hitplot.py",
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
    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["contexts"] == ["all", "short", "mid", "long"]
    assert payload["n_total_by_context"] == {"all": 3, "short": 1, "mid": 1, "long": 1}
    assert "not tractography" in payload["safety_statement"]
    assert any(g["group_id"] == "functional:motor_adjacent" for g in payload["groups"])


def test_main_cli_returns_2_when_hitplot_root_missing(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(tmp_path, ["0005"], [100.0])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_functional_anatomy_hitplot.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--hitplot-root",
            str(tmp_path / "missing"),
        ],
    )
    assert mod.main() == 2


def test_main_cli_strict_returns_3_on_missing_csv(monkeypatch, tmp_path: Path) -> None:
    cohort_yaml, metadata_csv = _write_minimal_cohort(tmp_path, ["0005"], [100.0])
    hitplot_root = tmp_path / "deriv"
    hitplot_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_functional_anatomy_hitplot.py",
            "--cohort-yaml",
            str(cohort_yaml),
            "--metadata-csv",
            str(metadata_csv),
            "--hitplot-root",
            str(hitplot_root),
            "--strict",
        ],
    )
    assert mod.main() == 3
