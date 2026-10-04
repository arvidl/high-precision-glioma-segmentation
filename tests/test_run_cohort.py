"""hpgs-run-cohort drives the same n=50 scripts as the Make targets."""

from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from hpgs.cli import app, cohort_stage_commands


def _record(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, check=False):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("hpgs.cli.subprocess.run", fake_run)
    return calls


def test_cohort_stage_commands_real_run_matches_make_flags(tmp_path: Path) -> None:
    work = tmp_path / "fs"
    commands = cohort_stage_commands(
        cohort_yaml=Path("configs/cohort_ucsfpdgm_n50.yaml"),
        metadata_csv=Path("data/UCSF-PDGM-metadata_v5.csv"),
        cohort_root=Path("data/ucsf_pdgm_cohort50"),
        deriv_root=Path("data/derivatives_cohort50"),
        stages=["segment", "parcellate", "hitplot"],
        subjects=["0005"],
        dry_run=False,
        device="cpu",
        segment_backend="monai_bundle",
        parcellate_backend="freesurfer",
        fs_work_dir=work,
        threads=8,
        skip_existing=True,
    )

    assert [Path(cmd[1]).name for cmd in commands] == [
        "segment_all_cohort50.py",
        "parcellate_all_cohort50.py",
        "compute_hitplot_cohort50.py",
    ]
    segment, parcellate, hitplot = commands
    assert "--no-dry-run" in segment
    assert "--device" in segment and segment[segment.index("--device") + 1] == "cpu"
    assert "--subject" in segment and "0005" in segment
    assert "--skip-existing" in segment
    assert "--fs-work-dir" in parcellate
    assert parcellate[parcellate.index("--fs-work-dir") + 1] == str(work)
    assert parcellate[parcellate.index("--threads") + 1] == "8"
    assert "--require-parcellation" in hitplot


def test_run_cohort_dry_run_calls_three_scripts(monkeypatch) -> None:
    calls = _record(monkeypatch)
    result = CliRunner().invoke(app, ["run-cohort", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert all("--dry-run" in cmd for cmd in calls)
    assert all("--no-dry-run" not in cmd for cmd in calls)
    assert "cohort stages finished: segment, parcellate, hitplot" in result.stdout


def test_run_cohort_real_parcellation_requires_fs_work_dir(monkeypatch) -> None:
    calls = _record(monkeypatch)
    result = CliRunner().invoke(app, ["run-cohort", "--stage", "parcellate"])

    assert result.exit_code != 0
    assert "fs-work-dir" in result.output
    assert calls == []


def test_run_cohort_dummy_parcellation_does_not_need_fs_work_dir(monkeypatch) -> None:
    calls = _record(monkeypatch)
    result = CliRunner().invoke(
        app,
        ["run-cohort", "--stage", "parcellate", "--parcellate-backend", "dummy"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][calls[0].index("--backend") + 1] == "dummy"
    assert "--fs-work-dir" not in calls[0]


def test_run_cohort_stops_after_a_failed_stage(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=False):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("hpgs.cli.subprocess.run", fake_run)
    result = CliRunner().invoke(app, ["run-cohort", "--stage", "segment", "--stage", "hitplot"])

    assert result.exit_code == 1
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "segment_all_cohort50.py"
