"""Pre-processing: intensity normalisation, optional N4 bias correction, SynthSR."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def zscore_within_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalise ``volume`` using statistics computed inside ``mask``."""
    sel = volume[mask > 0]
    mu, sd = float(sel.mean()), float(sel.std() + 1e-8)
    out = (volume - mu) / sd
    return out.astype(np.float32)


def _freesurfer_path(*parts: str) -> Path:
    fs_home = os.environ.get("FREESURFER_HOME")
    if not fs_home:
        raise OSError("FREESURFER_HOME is not set.")
    return Path(fs_home, *parts)


def _resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidate = _freesurfer_path("bin", name)
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"Could not find FreeSurfer executable: {name}")


def synth_sr(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threads: int = 4,
    cpu: bool = True,
    disable_flipping: bool = False,
    disable_sharpening: bool = False,
) -> Path:
    """Run FreeSurfer 8.2.0 ``mri_synthsr`` and return the output path.

    Wraps ``mri_synthsr --i INPUT --o OUTPUT --threads N [--cpu]``. The output is
    a synthetic 1 mm isotropic MP-RAGE volume.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _resolve_executable("mri_synthsr"),
        "--i",
        str(input_path),
        "--o",
        str(output_path),
        "--threads",
        str(threads),
    ]
    if cpu:
        command.append("--cpu")
    if disable_flipping:
        command.append("--disable_flipping")
    if disable_sharpening:
        command.append("--disable_sharpening")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(detail) from exc

    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    return output_path
