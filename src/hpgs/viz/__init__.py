"""Visualisation helpers for figures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def montage_axial(volume: np.ndarray, n_slices: int = 9, *, ax=None):
    """Render ``n_slices`` evenly-spaced axial slices as a montage."""
    import matplotlib.pyplot as plt  # noqa: PLC0415  — lazy: avoid mpl at import time

    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")
    z = np.linspace(volume.shape[2] * 0.15, volume.shape[2] * 0.85, n_slices, dtype=int)
    cols = int(np.ceil(np.sqrt(n_slices)))
    rows = int(np.ceil(n_slices / cols))
    if ax is None:
        _, ax = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
    flat = np.atleast_1d(ax).ravel()
    for a in flat:
        a.axis("off")
    for i, k in enumerate(z):
        flat[i].imshow(np.rot90(volume[:, :, k]), cmap="gray")
    return ax


def parse_freesurfer_color_lut(
    lut_path: str | Path | None = None,
) -> dict[int, tuple[int, int, int]]:
    """Parse FreeSurferColorLUT.txt into a ``{label: (R, G, B)}`` mapping.

    If ``lut_path`` is None, use ``$FREESURFER_HOME/FreeSurferColorLUT.txt``.
    """
    if lut_path is None:
        fs_home = os.environ.get("FREESURFER_HOME")
        if not fs_home:
            raise OSError("FREESURFER_HOME not set and no lut_path given.")
        lut_path = Path(fs_home) / "FreeSurferColorLUT.txt"
    lut_path = Path(lut_path)
    if not lut_path.is_file():
        raise FileNotFoundError(lut_path)

    lut: dict[int, tuple[int, int, int]] = {}
    for raw in lut_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            label = int(parts[0])
            r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        lut[label] = (r, g, b)
    return lut


def colorize_label_volume(
    labels: np.ndarray,
    lut: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """Return an ``(..., 3)`` uint8 RGB volume from an integer label volume."""
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique = np.unique(labels)
    for value in unique:
        if int(value) == 0:
            continue
        color = lut.get(int(value))
        if color is None:
            continue
        mask = labels == value
        rgb[mask] = color
    return rgb
