"""Smoke tests for the Figure 2 renderer and its FreeSurfer color-LUT helper.

These tests construct tiny synthetic volumes and exercise:

- ``hpgs.viz.parse_freesurfer_color_lut`` and ``colorize_label_volume``.
- ``scripts.render_figure2_panels.render_subject_panels`` on a 4x4x4 synthetic
  subject directory, producing all eight panel PNGs without requiring any
  FreeSurfer binaries.

The synthetic subject does not need real SynthSR or recon-all-clinical
outputs; the renderer falls back gracefully when label overlays are missing
and substitutes T1 for SynthSR.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hpgs.viz import colorize_label_volume, parse_freesurfer_color_lut

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_figure2_panels.py"


def _load_render_module():
    spec = importlib.util.spec_from_file_location("render_figure2_panels", RENDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def _make_subject(root: Path, sid: str) -> None:
    subj = root / f"sub-{sid}"
    rng = np.random.default_rng(0)
    base = rng.random((16, 16, 16)).astype(np.float32)
    for name in ("T1", "T1_bias", "T1c_bias", "FLAIR_bias"):
        _write_nifti(subj / f"sub-{sid}_{name}.nii.gz", base + 0.1 * rng.random(base.shape))
    tumor = np.zeros_like(base, dtype=np.int16)
    tumor[6:10, 6:10, 6:10] = 1
    tumor[7:9, 7:9, 7:9] = 2
    tumor[8, 8, 8] = 4
    _write_nifti(subj / f"sub-{sid}_tumor_segmentation.nii.gz", tumor)


def _make_derivatives(root: Path, sid: str) -> None:
    deriv = root / f"sub-{sid}"
    par = deriv / "parcellation"
    pre = deriv / "preprocess"
    labels = np.zeros((16, 16, 16), dtype=np.int32)
    labels[2:8, :, :] = 2
    labels[8:14, :, :] = 41
    labels[:, :, 0:8] = np.where(labels[:, :, 0:8] == 0, 3, labels[:, :, 0:8])
    _write_nifti(par / "synthseg.nii.gz", labels)
    _write_nifti(par / "aparc+aseg.nii.gz", labels)
    _write_nifti(par / "wmparc.nii.gz", labels)
    _write_nifti(pre / "synthSR_standalone.nii.gz", labels.astype(np.float32))


def test_parse_freesurfer_color_lut_handles_missing_env(monkeypatch, tmp_path):
    fake = tmp_path / "FreeSurferColorLUT.txt"
    fake.write_text(
        "# header\n0   Unknown 0 0 0 0\n2   Left-WM 245 245 245 0\n41  Right-WM 0 225 0 0\n"
    )
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    lut = parse_freesurfer_color_lut(fake)
    assert lut[2] == (245, 245, 245)
    assert lut[41] == (0, 225, 0)


def test_colorize_label_volume():
    labels = np.array([[[0, 2], [41, 2]]], dtype=np.int32)
    lut = {2: (10, 20, 30), 41: (200, 50, 25)}
    rgb = colorize_label_volume(labels, lut)
    assert rgb.shape == (*labels.shape, 3)
    assert tuple(rgb[0, 0, 1]) == (10, 20, 30)
    assert tuple(rgb[0, 1, 0]) == (200, 50, 25)
    assert tuple(rgb[0, 0, 0]) == (0, 0, 0)


def test_render_subject_panels_produces_eight_pngs(tmp_path, monkeypatch):
    render = _load_render_module()

    data_root = tmp_path / "data"
    deriv_root = tmp_path / "deriv"
    out_dir = tmp_path / "out"
    sid = "0099"
    _make_subject(data_root, sid)
    _make_derivatives(deriv_root, sid)

    lut_file = tmp_path / "FreeSurferColorLUT.txt"
    lut_file.write_text(
        "0 Unknown 0 0 0 0\n"
        "2 Left-WM 245 245 245 0\n"
        "3 Left-Ctx 205 62 78 0\n"
        "41 Right-WM 0 225 0 0\n"
    )
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))

    outputs = render.render_subject_panels(
        subject_id=sid,
        data_root=data_root,
        derivatives_root=deriv_root,
        coord=(8, 8, 8),
        output_dir=out_dir,
        name_prefix="test-",
    )
    assert len(outputs) == 8
    letters = [p.name.split("_panel_")[-1].removesuffix(".png") for p in outputs]
    assert letters == ["a", "b", "c", "d", "e", "f", "g", "h"]
    for p in outputs:
        assert p.is_file()
        assert p.stat().st_size > 0


def test_render_subject_panels_errors_when_inputs_missing(tmp_path, monkeypatch):
    render = _load_render_module()
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))
    (tmp_path / "FreeSurferColorLUT.txt").write_text("0 Unknown 0 0 0 0\n")
    with pytest.raises(FileNotFoundError):
        render.render_subject_panels(
            subject_id="9999",
            data_root=tmp_path / "nope",
            derivatives_root=tmp_path / "nope-deriv",
            coord=(1, 1, 1),
            output_dir=tmp_path / "out",
            name_prefix="x-",
        )
