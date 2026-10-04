"""Tests for ``scripts/bench_runtime.py`` (PR-12, R2.12).

Covers the pure helpers (sidecar loading, cold/warm split, stage
aggregation, time formatting, LaTeX rendering) on synthetic
fixtures so the test suite stays fast and host-independent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

# bench_runtime lives under ``scripts/`` (not a Python package), so we
# load it via importlib rather than ``import scripts.bench_runtime``.
# We register the module in ``sys.modules`` BEFORE exec_module so that
# ``@dataclass(frozen=True)``'s KW_ONLY introspection finds the module.
_BENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_runtime.py"
_spec = importlib.util.spec_from_file_location("bench_runtime", _BENCH_PATH)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_runtime"] = bench
_spec.loader.exec_module(bench)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_seg_sidecar(
    deriv_root: Path,
    subject_id: str,
    elapsed_s: float,
    *,
    device: str = "mps",
    bundle_version: str = "0.4.8",
    schema_version: str = "1.0",
    extra: dict | None = None,
) -> Path:
    sid = subject_id
    p = deriv_root / f"sub-{sid}" / "seg_dl" / f"sub-{sid}_seg_brats3_dl.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "schema_version": schema_version,
        "subject_id": sid,
        "backend": "monai_bundle",
        "bundle_version": bundle_version,
        "device": device,
        "elapsed_s": elapsed_s,
        "voxel_volume_mm3": 1.0,
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "channel_order": ["T1c", "T1", "T2", "FLAIR"],
        "probability_channel_order": ["TC", "WT", "ET"],
        "outputs": {},
    }
    if extra:
        doc.update(extra)
    p.write_text(json.dumps(doc))
    return p


def _write_parc_sidecar(
    deriv_root: Path,
    subject_id: str,
    elapsed_s: float,
    *,
    threads: int = 10,
    version: str = "freesurfer-macOS-darwin_arm64-8.2.0-20260314-d932c45",
    schema_version: str = "1.0",
) -> Path:
    sid = subject_id
    p = deriv_root / f"sub-{sid}" / "parcellation" / f"sub-{sid}_wmparc.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema_version,
        "subject_id": sid,
        "backend": "freesurfer",
        "version": version,
        "input_channel": "T1_bias",
        "freesurfer_subjects_dir": "/tmp/fs_scratch",
        "threads": threads,
        "elapsed_s": elapsed_s,
        "n_labels_present": 172,
        "outputs": {},
    }
    p.write_text(json.dumps(doc))
    return p


# ---------------------------------------------------------------------------
# _coerce_elapsed
# ---------------------------------------------------------------------------


def test_coerce_elapsed_accepts_finite_non_negative_numbers():
    assert bench._coerce_elapsed(0.0) == 0.0
    assert bench._coerce_elapsed(1) == 1.0
    assert bench._coerce_elapsed(123.456) == pytest.approx(123.456)


@pytest.mark.parametrize("bad", [None, True, "hello", "inf", float("inf"), float("nan"), -1.0])
def test_coerce_elapsed_rejects_invalid(bad):
    assert bench._coerce_elapsed(bad) is None


# ---------------------------------------------------------------------------
# Sidecar loaders
# ---------------------------------------------------------------------------


def test_load_seg_records_picks_up_present_subjects(tmp_path: Path):
    deriv = tmp_path / "deriv"
    _write_seg_sidecar(deriv, "0001", 22.0)
    _write_seg_sidecar(deriv, "0002", 1.5)
    records, problems = bench.load_seg_records(deriv, ["0001", "0002", "0003"])
    assert len(records) == 2
    assert {r.subject_id for r in records} == {"0001", "0002"}
    assert all(r.stage == "seg" for r in records)
    assert all(r.device == "mps" for r in records)
    assert len(problems) == 1
    assert problems[0][0] == "0003"
    assert "not found" in problems[0][1]


def test_load_seg_records_records_mtime_for_cold_start(tmp_path: Path):
    deriv = tmp_path / "deriv"
    p1 = _write_seg_sidecar(deriv, "0001", 22.0)
    time.sleep(0.01)
    p2 = _write_seg_sidecar(deriv, "0002", 1.3)
    # Force ordered mtimes (some FSes round to whole seconds).
    os.utime(p1, (1_700_000_000.0, 1_700_000_000.0))
    os.utime(p2, (1_700_000_500.0, 1_700_000_500.0))
    records, _ = bench.load_seg_records(deriv, ["0001", "0002"])
    by_sid = {r.subject_id: r for r in records}
    assert by_sid["0001"].mtime == pytest.approx(1_700_000_000.0)
    assert by_sid["0002"].mtime == pytest.approx(1_700_000_500.0)


def test_load_seg_records_skips_bad_schema_version(tmp_path: Path):
    deriv = tmp_path / "deriv"
    _write_seg_sidecar(deriv, "0001", 1.5, schema_version="9.9")
    records, problems = bench.load_seg_records(deriv, ["0001"])
    assert records == []
    assert len(problems) == 1
    assert "schema_version" in problems[0][1]


def test_load_seg_records_skips_missing_elapsed(tmp_path: Path):
    deriv = tmp_path / "deriv"
    p = deriv / "sub-0001" / "seg_dl" / "sub-0001_seg_brats3_dl.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "subject_id": "0001",
                "backend": "monai_bundle",
                "device": "mps",
                # elapsed_s missing on purpose
            }
        )
    )
    records, problems = bench.load_seg_records(deriv, ["0001"])
    assert records == []
    assert "elapsed_s" in problems[0][1]


def test_load_parc_records_extracts_threads_into_device_label(tmp_path: Path):
    deriv = tmp_path / "deriv"
    _write_parc_sidecar(deriv, "0001", 1500.0, threads=10)
    _write_parc_sidecar(deriv, "0002", 2000.0, threads=4)
    records, _ = bench.load_parc_records(deriv, ["0001", "0002"])
    by_sid = {r.subject_id: r for r in records}
    assert by_sid["0001"].device == "CPU/10t"
    assert by_sid["0002"].device == "CPU/4t"
    assert all(r.stage == "parc" for r in records)


# ---------------------------------------------------------------------------
# split_seg_cold_warm
# ---------------------------------------------------------------------------


def test_split_seg_cold_warm_picks_earliest_mtime():
    r0 = bench.TimingRecord("0001", "seg", 22.0, "mb", "mps", mtime=100.0)
    r1 = bench.TimingRecord("0002", "seg", 1.3, "mb", "mps", mtime=200.0)
    r2 = bench.TimingRecord("0003", "seg", 1.4, "mb", "mps", mtime=150.0)
    cold, warm = bench.split_seg_cold_warm([r2, r0, r1])  # shuffled input
    assert cold is not None
    assert cold.subject_id == "0001"
    assert [r.subject_id for r in warm] == ["0003", "0002"]


def test_split_seg_cold_warm_handles_empty_and_single():
    cold, warm = bench.split_seg_cold_warm([])
    assert cold is None and warm == []
    only = bench.TimingRecord("0001", "seg", 22.0, "mb", "mps", mtime=100.0)
    cold, warm = bench.split_seg_cold_warm([only])
    assert cold is only and warm == []


# ---------------------------------------------------------------------------
# summarise_stages
# ---------------------------------------------------------------------------


def test_summarise_stages_emits_three_rows_with_correct_n_used(tmp_path: Path):
    deriv = tmp_path / "deriv"
    # 3 seg sidecars (1 cold, 2 warm), 2 parc sidecars
    p_cold = _write_seg_sidecar(deriv, "0001", 22.0)
    p_warm1 = _write_seg_sidecar(deriv, "0002", 1.3)
    p_warm2 = _write_seg_sidecar(deriv, "0003", 1.4)
    os.utime(p_cold, (1_700_000_000.0, 1_700_000_000.0))
    os.utime(p_warm1, (1_700_000_500.0, 1_700_000_500.0))
    os.utime(p_warm2, (1_700_001_000.0, 1_700_001_000.0))
    _write_parc_sidecar(deriv, "0001", 1500.0)
    _write_parc_sidecar(deriv, "0002", 2500.0)
    seg_records, _ = bench.load_seg_records(deriv, ["0001", "0002", "0003"])
    parc_records, _ = bench.load_parc_records(deriv, ["0001", "0002", "0003"])

    summaries = bench.summarise_stages(seg_records, parc_records, n_total=3)
    assert len(summaries) == 3
    cold, warm, parc = summaries
    assert cold.n_used == 1
    assert cold.median == pytest.approx(22.0)
    assert warm.n_used == 2
    assert warm.median == pytest.approx(1.35)  # median of [1.3, 1.4]
    assert parc.n_used == 2
    assert parc.median == pytest.approx(2000.0)
    assert parc.n_expected == 3  # cohort total


def test_summarise_stages_picks_up_freesurfer_version(tmp_path: Path):
    deriv = tmp_path / "deriv"
    _write_parc_sidecar(deriv, "0001", 1500.0)
    parc_records, _ = bench.load_parc_records(deriv, ["0001"])
    summaries = bench.summarise_stages([], parc_records, n_total=1)
    parc_row = summaries[2]
    assert "FreeSurfer 8.2.0" in parc_row.backend_label


def test_summarise_stages_with_no_data_emits_three_empty_rows():
    summaries = bench.summarise_stages([], [], n_total=50)
    assert len(summaries) == 3
    for s in summaries:
        assert s.n_used == 0
        assert s.median is None


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "--"),
        (0.0, "0.00\\,s"),
        (1.294, "1.29\\,s"),
        (59.5, "59.50\\,s"),
        (60.0, "1:00\\,min"),
        (75.0, "1:15\\,min"),
        (2060.3, "34:20\\,min"),
        (3600.0, "1:00:00\\,hr"),
        (3725.0, "1:02:05\\,hr"),
        (4384.6, "1:13:05\\,hr"),
    ],
)
def test_fmt_duration(seconds, expected):
    assert bench._fmt_duration(seconds) == expected


def test_fmt_cell_handles_n0_n1_and_nm():
    s_empty = bench.StageSummary("X", "B", "D", n_used=0, n_expected=10)
    assert bench._fmt_cell(s_empty) == "--"
    s_one = bench.StageSummary("X", "B", "D", n_used=1, n_expected=1, elapsed_s=[22.0])
    assert bench._fmt_cell(s_one) == "22.00\\,s"
    s_many = bench.StageSummary(
        "X",
        "B",
        "D",
        n_used=3,
        n_expected=3,
        elapsed_s=[1.3, 1.4, 1.5],
    )
    cell = bench._fmt_cell(s_many)
    assert cell == "1.40\\,s [1.35\\,s, 1.45\\,s]"


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def test_format_table_tex_emits_self_contained_tabular(tmp_path: Path):
    s_cold = bench.StageSummary(
        stage_label="DL segmentation (first-call latency)",
        backend_label="MONAI Bundle 0.4.8 (BraTS-3 SegResNet)",
        device_label="MPS",
        n_used=1,
        n_expected=1,
        elapsed_s=[22.0],
    )
    s_warm = bench.StageSummary(
        stage_label="DL segmentation (steady-state)",
        backend_label="MONAI Bundle 0.4.8 (BraTS-3 SegResNet)",
        device_label="MPS",
        n_used=49,
        n_expected=49,
        elapsed_s=[1.3] * 49,
    )
    s_parc = bench.StageSummary(
        stage_label="Anatomical parcellation",
        backend_label="FreeSurfer 8.2.0 recon-all-clinical (wmparc)",
        device_label="CPU/10t",
        n_used=17,
        n_expected=50,
        elapsed_s=[2000.0] * 17,
    )
    fingerprint = {
        "platform": "macOS-26.4.1-arm64-arm-64bit",
        "machine": "arm64",
        "python": "3.11.15",
    }
    tex = bench.format_table_tex([s_cold, s_warm, s_parc], n_total=50, fingerprint=fingerprint)
    assert "AUTO-GENERATED by scripts/bench_runtime.py" in tex
    assert r"\begin{tabularx}" in tex and r"\end{tabularx}" in tex
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    # The partial parcellation row reports both n_used and n_expected.
    assert "17/50" in tex
    # Steady-state and cold rows show n only (n_used == n_expected for cold;
    # for warm we wrote 49/49 which collapses to "49").
    assert " 49 " in tex
    # Host fingerprint reproduced in comment header.
    assert "macOS-26.4.1-arm64-arm-64bit" in tex
    # FreeSurfer 8.2.0 label landed.
    assert "FreeSurfer 8.2.0" in tex


def test_format_table_tex_handles_all_empty_rows():
    summaries = bench.summarise_stages([], [], n_total=50)
    tex = bench.format_table_tex(summaries, n_total=50, fingerprint={})
    # All three cells should render as -- when no data is available.
    assert tex.count("-- \\\\") + tex.count("--\\\\") >= 3


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def test_write_csv_round_trips_records(tmp_path: Path):
    r1 = bench.TimingRecord("0001", "seg", 22.0, "monai_bundle", "mps", mtime=100.0)
    r2 = bench.TimingRecord("0002", "parc", 1500.0, "freesurfer", "CPU/10t", mtime=200.0)
    out = tmp_path / "table4.csv"
    bench.write_csv([r1, r2], out)
    text = out.read_text().splitlines()
    assert text[0] == "subject_id,stage,elapsed_s,backend,device,sidecar,mtime"
    # Sorted by (stage, subject_id) -> parc before seg.
    assert text[1].startswith("0002,parc,1500.000000")
    assert text[2].startswith("0001,seg,22.000000")
