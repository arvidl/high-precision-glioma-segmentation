"""Render the manuscript's e/g subject panel from HPGS derivatives.

This helper regenerates the "e/g" subfigure pair for a subject:

- e) tumor segmentation overlay on the HPGS T1-weighted input
- g) FreeSurfer 8.2.0 SynthSeg parcellation overlay on the same input

The output is a 2x3 grid of orthogonal slices (sagittal/coronal/axial)
at a user-specified voxel coordinate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _normalize(volume: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(volume, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _views(volume: np.ndarray, coord: tuple[int, int, int]) -> list[np.ndarray]:
    i, j, k = coord
    return [
        np.rot90(volume[i, :, :]),
        np.rot90(volume[:, j, :]),
        np.rot90(volume[:, :, k]),
    ]


def _plot_tumor_overlay(ax, base: np.ndarray, tumor: np.ndarray, title: str) -> None:
    ax.imshow(base, cmap="gray", interpolation="nearest")
    for label, color, alpha in [(1, "red", 0.55), (2, "lime", 0.45), (4, "yellow", 0.45)]:
        mask = np.ma.masked_where(tumor != label, tumor)
        ax.imshow(mask, cmap=matplotlib.colors.ListedColormap([color]), alpha=alpha)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _plot_synthseg_overlay(ax, base: np.ndarray, synthseg: np.ndarray, title: str) -> None:
    ax.imshow(base, cmap="gray", interpolation="nearest")
    mask = np.ma.masked_where(synthseg <= 0, synthseg)
    ax.imshow(mask, cmap="nipy_spectral", alpha=0.38, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_panel(
    *,
    subject_id: str,
    data_root: Path,
    derivatives_root: Path,
    coord: tuple[int, int, int],
    output: Path,
) -> None:
    subject_dir = data_root / f"sub-{subject_id}"
    deriv_dir = derivatives_root / f"sub-{subject_id}" / "parcellation"

    t1_bias = _load(subject_dir / f"sub-{subject_id}_T1_bias.nii.gz")
    tumor = _load(subject_dir / f"sub-{subject_id}_tumor_segmentation.nii.gz")
    synthseg = _load(deriv_dir / "synthseg.nii.gz")

    base_views = _views(_normalize(t1_bias), coord)
    tumor_views = _views(tumor, coord)
    synthseg_views = _views(synthseg, coord)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plane_titles = ["Sagittal", "Coronal", "Axial"]

    for idx, plane in enumerate(plane_titles):
        title = f"e) Tumor overlay ({plane})" if idx == 0 else plane
        _plot_tumor_overlay(axes[0, idx], base_views[idx], tumor_views[idx], title)

    for idx, plane in enumerate(plane_titles):
        title = f"g) FS 8.2.0 SynthSeg ({plane})" if idx == 0 else plane
        _plot_synthseg_overlay(axes[1, idx], base_views[idx], synthseg_views[idx], title)

    fig.suptitle(f"UCSF-PDGM-{subject_id} HPGS regeneration", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="4-digit UCSF-PDGM subject id.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--derivatives-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coord",
        nargs=3,
        type=int,
        metavar=("I", "J", "K"),
        required=True,
        help="Voxel coordinate used for sagittal/coronal/axial slices.",
    )
    args = parser.parse_args()
    render_panel(
        subject_id=args.subject,
        data_root=args.data_root,
        derivatives_root=args.derivatives_root,
        coord=tuple(args.coord),
        output=args.output,
    )


if __name__ == "__main__":
    _cli()
