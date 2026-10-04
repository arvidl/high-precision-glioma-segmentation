"""Orientation tests — canonical-RAS loading, voxel-coord translation, display.

These tests nail down the radiological-orientation semantics used by Figure 2:

* ``hpgs.io.load_nifti_canonical`` reorients an LPS-oriented NIfTI (the
  UCSF-PDGM convention) to canonical RAS on load.
* ``hpgs.io.orig_voxel_to_ras`` maps a voxel coordinate expressed in the
  original (LPS) array frame into the reoriented RAS frame consistently.
* The renderer's ``_radiological`` display transform produces a
  radiological-convention picture: for an axial RAS slice (R, A), the patient's
  right side ends up on the image's LEFT (image-left = patient-right) and the
  anterior direction points UP.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from hpgs.io import load_nifti_canonical, orig_voxel_to_ras

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_render_module():
    spec = importlib.util.spec_from_file_location(
        "render_figure2_panels", REPO_ROOT / "scripts" / "render_figure2_panels.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lps_affine(shape: tuple[int, int, int]) -> np.ndarray:
    """Return an LPS affine (matching UCSF-PDGM) for a volume of ``shape``."""
    aff = np.eye(4)
    aff[0, 0] = -1.0
    aff[1, 1] = -1.0
    aff[0, 3] = 0.0
    aff[1, 3] = float(shape[1] - 1)
    return aff


def test_load_canonical_reorients_lps_to_ras(tmp_path):
    shape = (6, 7, 5)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / "lps.nii.gz"
    nib.save(nib.Nifti1Image(data, _lps_affine(shape)), str(path))

    data_ras, orig_shape, transform = load_nifti_canonical(path)

    assert orig_shape == shape
    assert data_ras.shape == shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                new_i, new_j, new_k = orig_voxel_to_ras((i, j, k), orig_shape, transform)
                assert data_ras[new_i, new_j, new_k] == data[i, j, k]


def test_orig_voxel_to_ras_flips_L_and_P_axes(tmp_path):
    shape = (10, 12, 8)
    data = np.zeros(shape, dtype=np.float32)
    data[1, 2, 3] = 7.0
    path = tmp_path / "marker.nii.gz"
    nib.save(nib.Nifti1Image(data, _lps_affine(shape)), str(path))

    data_ras, orig_shape, transform = load_nifti_canonical(path)
    new_coord = orig_voxel_to_ras((1, 2, 3), orig_shape, transform)

    expected = (shape[0] - 1 - 1, shape[1] - 1 - 2, 3)
    assert new_coord == expected
    assert data_ras[new_coord] == 7.0


def test_radiological_display_places_patient_right_on_image_left():
    render = _load_render_module()

    size_r, size_a, size_s = 8, 9, 7

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[size_r - 1, :, :] = 1.0
    axial = ras[:, :, 3]
    displayed = render._radiological(axial)
    assert displayed.shape == (size_a, size_r)
    assert displayed[:, 0].mean() == 1.0
    assert displayed[:, -1].mean() == 0.0

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[:, size_a - 1, :] = 1.0
    axial = ras[:, :, 3]
    displayed = render._radiological(axial)
    assert displayed[0, :].mean() == 1.0
    assert displayed[-1, :].mean() == 0.0

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[size_r - 1, :, :] = 1.0
    coronal = ras[:, 4, :]
    displayed = render._radiological(coronal)
    assert displayed.shape == (size_s, size_r)
    assert displayed[:, 0].mean() == 1.0
    assert displayed[:, -1].mean() == 0.0

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[:, :, size_s - 1] = 1.0
    coronal = ras[:, 4, :]
    displayed = render._radiological(coronal)
    assert displayed[0, :].mean() == 1.0
    assert displayed[-1, :].mean() == 0.0

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[:, size_a - 1, :] = 1.0
    sagittal = ras[4, :, :]
    displayed = render._radiological(sagittal)
    assert displayed.shape == (size_s, size_a)
    assert displayed[:, 0].mean() == 1.0
    assert displayed[:, -1].mean() == 0.0

    ras = np.zeros((size_r, size_a, size_s), dtype=np.float32)
    ras[:, :, size_s - 1] = 1.0
    sagittal = ras[4, :, :]
    displayed = render._radiological(sagittal)
    assert displayed[0, :].mean() == 1.0
    assert displayed[-1, :].mean() == 0.0


def test_crosshair_coordinates_for_radiological_axial():
    render = _load_render_module()

    ras_shape = (20, 24, 16)
    coord_ras = (15, 18, 10)

    class _FakeAx:
        def __init__(self):
            self.hy = None
            self.vx = None

        def axhline(self, y, **kwargs):
            self.hy = y

        def axvline(self, x, **kwargs):
            self.vx = x

    axial_ax = _FakeAx()
    render._draw_crosshair(axial_ax, 2, coord_ras, ras_shape)
    assert axial_ax.vx == ras_shape[0] - 1 - coord_ras[0]
    assert axial_ax.hy == ras_shape[1] - 1 - coord_ras[1]

    coronal_ax = _FakeAx()
    render._draw_crosshair(coronal_ax, 1, coord_ras, ras_shape)
    assert coronal_ax.vx == ras_shape[0] - 1 - coord_ras[0]
    assert coronal_ax.hy == ras_shape[2] - 1 - coord_ras[2]

    sagittal_ax = _FakeAx()
    render._draw_crosshair(sagittal_ax, 0, coord_ras, ras_shape)
    assert sagittal_ax.vx == ras_shape[1] - 1 - coord_ras[1]
    assert sagittal_ax.hy == ras_shape[2] - 1 - coord_ras[2]
