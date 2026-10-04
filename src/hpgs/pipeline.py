"""Pipeline orchestration for single-subject runs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hpgs.io import SubjectInputs, derivatives_dir, resolve_subject_inputs
from hpgs.parcellate import synthseg, synthseg_label_lut
from hpgs.preprocess import synth_sr


def _load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config structure in {path}")
    return config


def _manifest_payload(
    *,
    subject_inputs: SubjectInputs,
    config_path: Path,
    preprocess_output: Path | None,
    synthseg_output: Path,
    lut_size: int,
) -> dict:
    return {
        "subject_id": subject_inputs.subject_id,
        "config_path": str(config_path),
        "inputs": {
            "subject_dir": str(subject_inputs.subject_dir),
            "channels": {name: str(path) for name, path in subject_inputs.channels.items()},
            "tumor_segmentation": str(subject_inputs.tumor_segmentation),
            "reference_mask": (
                str(subject_inputs.reference_mask) if subject_inputs.reference_mask else None
            ),
        },
        "outputs": {
            "synthsr": str(preprocess_output) if preprocess_output else None,
            "synthseg": str(synthseg_output),
        },
        "metadata": {
            "synthseg_lut_size": lut_size,
        },
    }


def run_subject_pipeline(subject_id: str, config_path: str | Path) -> dict[str, Path]:
    """Run the minimal Phase 1 single-subject slice.

    The current slice resolves one extracted UCSF-PDGM subject, optionally runs
    ``SynthSR`` on the T1-weighted image, and then runs ``SynthSeg`` to produce a
    native-space parcellation.
    """
    config_path = Path(config_path)
    config = _load_config(config_path)

    inputs = resolve_subject_inputs(
        config["paths"]["data_root"],
        subject_id,
        channel_names=config["segment"]["channels"],
        reference_mask_name=config["hitplot"].get("reference_mask"),
    )

    subject_derivatives = derivatives_dir(config["paths"]["derivatives"], inputs.subject_id)
    preprocess_dir = subject_derivatives / "preprocess"
    parcellation_dir = subject_derivatives / "parcellation"

    parcellation_input = inputs.channels["T1_bias"]
    synthsr_output: Path | None = None
    if config["preprocess"].get("do_synth_sr", False):
        synthsr_output = preprocess_dir / "synthsr.nii.gz"
        parcellation_input = synth_sr(parcellation_input, synthsr_output)

    if config["parcellate"]["engine"] != "synthseg":
        raise NotImplementedError("Phase 1 only supports parcellate.engine = synthseg")

    synthseg_output = parcellation_dir / "synthseg.nii.gz"
    synthseg(
        parcellation_input,
        synthseg_output,
        robust=bool(config["parcellate"].get("robust", True)),
        parc=bool(config["parcellate"].get("parc", True)),
        fast=bool(config["parcellate"].get("fast", False)),
    )
    lut = synthseg_label_lut()

    manifest_path = subject_derivatives / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_payload(
                subject_inputs=inputs,
                config_path=config_path,
                preprocess_output=synthsr_output,
                synthseg_output=synthseg_output,
                lut_size=len(lut),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        "subject_dir": inputs.subject_dir,
        "derivatives_dir": subject_derivatives,
        "tumor_segmentation": inputs.tumor_segmentation,
        "reference_mask": inputs.reference_mask if inputs.reference_mask else Path(),
        "synthseg": synthseg_output,
        "manifest": manifest_path,
        **({"synthsr": synthsr_output} if synthsr_output else {}),
    }
