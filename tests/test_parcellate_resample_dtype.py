"""Regression test: ``resample_like`` must force ``-odt int`` for label maps.

Without ``-odt int``, ``mri_convert`` silently inherits the dtype of the
``-rl`` reference image. When that reference is uint16 (encountered on
LUMIERE Patient-048 week-000-1 whose registered T1 is uint16), the int32
wmparc gets re-stored via the MGH header's slope/intercept rescaling and
every label code is multiplied by ~13.10 (mapping the [0, 5002] FS code
range onto the full [0, 65535] uint16 range), corrupting downstream
Hit-Plot computation. The fix is in ``hpgs.parcellate.resample_like``:
when ``interpolation == "nearest"`` we always append ``-odt int`` so the
label codes survive the round-trip intact regardless of reference dtype.

The test mocks the ``mri_convert`` subprocess so it runs in CI without
FreeSurfer on the host, and asserts the exact command string contains
``-odt int`` for nearest-neighbour and does NOT contain ``-odt int`` for
trilinear (continuous-valued resampling like SynthSR which legitimately
inherits the reference dtype).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from hpgs.parcellate import resample_like


def _save_dummy_nifti(path: Path, dtype: np.dtype = np.int32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 4), dtype=dtype)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


def test_resample_like_forces_odt_int_for_nearest(tmp_path: Path) -> None:
    src = tmp_path / "wmparc.nii.gz"
    ref = tmp_path / "T1.nii.gz"
    out = tmp_path / "wmparc_native.nii.gz"
    _save_dummy_nifti(src)
    _save_dummy_nifti(ref, dtype=np.uint16)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        captured.append(list(cmd))
        # Must produce the output file or resample_like raises.
        _save_dummy_nifti(out)

    with (
        patch("hpgs.parcellate._run_checked", side_effect=fake_run),
        patch("hpgs.parcellate._resolve_executable", return_value="/fake/mri_convert"),
    ):
        resample_like(src, ref, out, interpolation="nearest")

    assert len(captured) == 1
    cmd = captured[0]
    assert "-rt" in cmd
    assert cmd[cmd.index("-rt") + 1] == "nearest"
    # The fix: nearest-neighbour label resampling MUST force int dtype.
    assert "-odt" in cmd, f"missing -odt flag: {cmd}"
    assert cmd[cmd.index("-odt") + 1] == "int"


@pytest.mark.parametrize("interpolation", ["trilinear", "cubic"])
def test_resample_like_does_not_force_int_for_continuous(
    tmp_path: Path, interpolation: str
) -> None:
    """SynthSR-style continuous resampling must keep the reference dtype."""
    src = tmp_path / "synthSR.nii.gz"
    ref = tmp_path / "T1.nii.gz"
    out = tmp_path / "synthSR_native.nii.gz"
    _save_dummy_nifti(src, dtype=np.float32)
    _save_dummy_nifti(ref, dtype=np.float32)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        captured.append(list(cmd))
        _save_dummy_nifti(out, dtype=np.float32)

    with (
        patch("hpgs.parcellate._run_checked", side_effect=fake_run),
        patch("hpgs.parcellate._resolve_executable", return_value="/fake/mri_convert"),
    ):
        resample_like(src, ref, out, interpolation=interpolation)

    cmd = captured[0]
    assert cmd[cmd.index("-rt") + 1] == interpolation
    # Continuous resampling must NOT force int output (would lose precision).
    assert "-odt" not in cmd, f"unexpected -odt flag for {interpolation}: {cmd}"
