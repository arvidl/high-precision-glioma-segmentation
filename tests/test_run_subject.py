"""Focused tests for the Phase 1 single-subject pipeline slice."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hpgs.cli import app
from hpgs.io import resolve_subject_inputs
from hpgs.pipeline import run_subject_pipeline


def _write_subject_files(root: Path, subject_id: str) -> None:
    subject_dir = root / f"sub-{subject_id}"
    subject_dir.mkdir(parents=True)
    suffixes = [
        "T1_bias",
        "T1c_bias",
        "T2_bias",
        "FLAIR_bias",
        "tumor_segmentation",
        "brain_parenchyma_segmentation",
    ]
    for suffix in suffixes:
        (subject_dir / f"sub-{subject_id}_{suffix}.nii.gz").write_bytes(b"nifti")


def _write_config(path: Path, *, data_root: Path, derivatives: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_root: {data_root}",
                f"  derivatives: {derivatives}",
                "preprocess:",
                "  do_synth_sr: false",
                "parcellate:",
                "  engine: synthseg",
                "  robust: true",
                "  parc: true",
                "  fast: false",
                "segment:",
                "  channels: [T1_bias, T1c_bias, T2_bias, FLAIR_bias]",
                "hitplot:",
                "  reference_mask: brain_parenchyma_segmentation",
            ]
        )
        + "\n"
    )


def test_resolve_subject_inputs_requires_expected_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    subject_dir = data_root / "sub-0005"
    subject_dir.mkdir(parents=True)
    (subject_dir / "sub-0005_T1_bias.nii.gz").write_bytes(b"nifti")

    try:
        resolve_subject_inputs(
            data_root,
            "0005",
            channel_names=["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"],
            reference_mask_name="brain_parenchyma_segmentation",
        )
    except FileNotFoundError as exc:
        assert "sub-0005_T1c_bias.nii.gz" in str(exc)
    else:
        raise AssertionError("Expected resolve_subject_inputs to fail for missing files")


def test_run_subject_pipeline_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    derivatives = tmp_path / "derivatives"
    config_path = tmp_path / "config.yaml"
    _write_subject_files(data_root, "0005")
    _write_config(config_path, data_root=data_root, derivatives=derivatives)

    def fake_synthseg(input_path: Path, output_path: Path, **_: object) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"synthseg")
        return output_path

    monkeypatch.setattr("hpgs.pipeline.synthseg", fake_synthseg)
    monkeypatch.setattr(
        "hpgs.pipeline.synthseg_label_lut",
        lambda: {0: "Unknown", 2: "Left-Cerebral-White-Matter"},
    )

    outputs = run_subject_pipeline("0005", config_path)

    manifest = json.loads(outputs["manifest"].read_text())
    assert outputs["synthseg"].is_file()
    assert manifest["subject_id"] == "0005"
    assert manifest["outputs"]["synthseg"] == str(outputs["synthseg"])
    assert manifest["metadata"]["synthseg_lut_size"] == 2


def test_run_subject_cli_invokes_pipeline(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n")

    monkeypatch.setattr(
        "hpgs.cli.run_subject_pipeline",
        lambda subject, config: {
            "manifest": Path("/tmp/fake-manifest.json"),
            "synthseg": Path("/tmp/fake-synthseg.nii.gz"),
        },
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run-subject", "--subject", "0005", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "[hpgs] finished sub-0005" in result.stdout
    assert "fake-manifest.json" in result.stdout
