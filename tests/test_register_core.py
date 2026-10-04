"""Smoke tests for ``hpgs.register`` primitives.

These tests use small synthetic NIfTI volumes (16^3) so the rigid
registration call returns in <2 s on commodity hardware. The goal is
correctness of plumbing (transforms saved, files written, label
preservation under nearest-neighbour interpolation), not registration
quality on real brain volumes.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hpgs.register import (
    DEFAULT_SEED,
    apply_transform,
    register_rigid,
    to_canonical_ras_file,
)


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_blob_volume(shape: tuple[int, int, int], centre: tuple[int, int, int]) -> np.ndarray:
    """Return a smooth Gaussian-blob volume centred at ``centre``."""
    grid = np.indices(shape, dtype=np.float32)
    cz, cy, cx = centre
    r2 = (grid[0] - cz) ** 2 + (grid[1] - cy) ** 2 + (grid[2] - cx) ** 2
    return np.exp(-r2 / (2 * 3.0**2)).astype(np.float32)


def _lps_affine(shape: tuple[int, int, int]) -> np.ndarray:
    """Return an LPS affine (UCSF-PDGM and one of the LUMIERE conventions)."""
    aff = np.eye(4)
    aff[0, 0] = -1.0
    aff[1, 1] = -1.0
    aff[1, 3] = float(shape[1] - 1)
    return aff


def test_to_canonical_ras_file_reorients_lps_to_ras(tmp_path: Path) -> None:
    shape = (10, 12, 8)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    src = tmp_path / "lps.nii.gz"
    dst = tmp_path / "ras.nii.gz"
    _save_nifti(data, _lps_affine(shape), src)

    out = to_canonical_ras_file(src, dst)

    assert out == dst
    img = nib.load(str(out))
    assert "".join(nib.aff2axcodes(img.affine)) == "RAS"


def test_to_canonical_ras_file_is_idempotent_for_canonical_input(tmp_path: Path) -> None:
    shape = (6, 7, 5)
    data = np.zeros(shape, dtype=np.float32)
    src = tmp_path / "ras.nii.gz"
    dst = tmp_path / "ras_again.nii.gz"
    _save_nifti(data, np.eye(4), src)

    to_canonical_ras_file(src, dst)
    out_img = nib.load(str(dst))

    assert "".join(nib.aff2axcodes(out_img.affine)) == "RAS"
    np.testing.assert_array_equal(np.asarray(out_img.dataobj), data)


def test_register_rigid_recovers_known_translation(tmp_path: Path) -> None:
    pytest.importorskip("ants")
    shape = (16, 16, 16)
    fixed_data = _make_blob_volume(shape, centre=(8, 8, 8))
    moving_data = _make_blob_volume(shape, centre=(8 + 2, 8 + 1, 8 - 1))

    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    _save_nifti(fixed_data, np.eye(4), fixed_path)
    _save_nifti(moving_data, np.eye(4), moving_path)

    out = register_rigid(
        moving=moving_path,
        fixed=fixed_path,
        out_dir=tmp_path / "reg",
        seed=DEFAULT_SEED,
    )

    assert out["warped"].is_file()
    assert all(p.is_file() for p in out["fwdtransforms"])
    assert all(p.is_file() for p in out["invtransforms"])

    warped = np.asarray(nib.load(str(out["warped"])).dataobj)
    sse_before = float(np.mean((moving_data - fixed_data) ** 2))
    sse_after = float(np.mean((warped - fixed_data) ** 2))
    assert sse_after < sse_before * 0.5, (
        f"rigid registration did not improve agreement: "
        f"sse_before={sse_before:.4g}, sse_after={sse_after:.4g}"
    )


def test_apply_transform_preserves_labels_with_nearest_neighbour(tmp_path: Path) -> None:
    pytest.importorskip("ants")
    shape = (16, 16, 16)
    fixed_data = _make_blob_volume(shape, centre=(8, 8, 8))
    moving_data = _make_blob_volume(shape, centre=(8 + 2, 8 + 1, 8 - 1))

    label_data = np.zeros(shape, dtype=np.int16)
    label_data[6:11, 6:11, 6:11] = 1
    # ET as a small blob (not a single voxel): rigid + nearest-neighbour can drop
    # one-voxel labels when ANTs converges to slightly different transforms (~1%).
    label_data[7:10, 7:10, 7:10] = 2

    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    label_path = tmp_path / "moving_labels.nii.gz"
    _save_nifti(fixed_data, np.eye(4), fixed_path)
    _save_nifti(moving_data, np.eye(4), moving_path)
    _save_nifti(label_data, np.eye(4), label_path)

    out = register_rigid(
        moving=moving_path,
        fixed=fixed_path,
        out_dir=tmp_path / "reg",
        seed=DEFAULT_SEED,
    )

    warped_label_path = tmp_path / "warped_labels.nii.gz"
    apply_transform(
        moving=label_path,
        reference=fixed_path,
        transform_paths=out["fwdtransforms"],
        out_path=warped_label_path,
        interpolator="nearestNeighbor",
    )

    warped_labels = np.asarray(nib.load(str(warped_label_path)).dataobj)
    unique_warped = set(int(v) for v in np.unique(warped_labels))
    unique_source = set(int(v) for v in np.unique(label_data))
    assert unique_warped.issubset(unique_source), (
        f"nearestNeighbor introduced new labels: warped={sorted(unique_warped)}, "
        f"source={sorted(unique_source)}"
    )
    assert unique_warped == unique_source, (
        f"nearestNeighbor lost labels: warped={sorted(unique_warped)}, "
        f"source={sorted(unique_source)}"
    )


def test_register_rigid_raises_on_missing_input(tmp_path: Path) -> None:
    fixed_path = tmp_path / "fixed.nii.gz"
    _save_nifti(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4), fixed_path)
    with pytest.raises(FileNotFoundError):
        register_rigid(
            moving=tmp_path / "missing.nii.gz",
            fixed=fixed_path,
            out_dir=tmp_path / "reg",
        )
