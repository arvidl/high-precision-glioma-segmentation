"""Brain parcellation wrappers around FreeSurfer 8.2.0 SynthSeg / recon-all-clinical.

Public high-level API (PR-7c onwards)::

    from hpgs.parcellate import predict_wmparc, ParcellationResult

    result = predict_wmparc(
        input_nifti=".../sub-0005_T1_bias.nii.gz",
        subject_id="0005",
        backend="freesurfer",    # real; needs FREESURFER_HOME
        fs_work_dir=".../fs_subjects",
    )
    result.label_map      # (D, H, W) int16, wmparc resampled onto native mpMRI grid
    result.affine         # (4, 4) float64, copied from the input T1 header
    result.lut            # {label_id (int): anatomical name (str)}
    result.backend        # "freesurfer" | "dummy"
    result.version        # FREESURFER_HOME-resolved version string, or "dummy"

The ``"dummy"`` backend is the CI-runnable companion used by the PR-7c
cohort runner test suite and by the PR-7 smoke pipeline: it emits a
deterministic synthetic wmparc with a small but realistic LUT subset of
``FreeSurferColorLUT`` so the downstream PR-7f Hit-Plot cohort runner
(which consumes ``parcellation/sub-XXXX_wmparc_native.nii.gz`` +
``..._wmparc_lut.json``) can be exercised end-to-end without
FreeSurfer on the host.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from hpgs.parcellate.lobes import (
    LOBE_OF_CORTICAL_STEM,
    LOBE_OF_SUBCORTICAL_NAME,
    LOBES,
    assign_lobe,
    assign_lobe_from_label,
    strip_hemisphere_prefix,
)

__all__ = [
    "FS_DUMMY_LUT",
    "LOBE_OF_CORTICAL_STEM",
    "LOBE_OF_SUBCORTICAL_NAME",
    "LOBES",
    "ParcellationBackend",
    "ParcellationResult",
    "assign_lobe",
    "assign_lobe_from_label",
    "predict_wmparc",
    "recon_all_clinical",
    "resample_like",
    "strip_hemisphere_prefix",
    "synthseg",
    "synthseg_label_lut",
]

ParcellationBackend = Literal["freesurfer", "dummy"]


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


def _run_checked(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(detail) from exc


def synthseg(
    input_path: str | Path,
    output_path: str | Path,
    *,
    robust: bool = True,
    parc: bool = True,
    fast: bool = False,
) -> Path:
    """Run FreeSurfer 8.2.0 ``mri_synthseg`` and return the output path.

    The command requires a working FreeSurfer 8.2.0 installation.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [_resolve_executable("mri_synthseg"), "--i", str(input_path), "--o", str(output_path)]
    if robust:
        command.append("--robust")
    if parc:
        command.append("--parc")
    if fast:
        command.append("--fast")

    _run_checked(command)
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    return output_path


def recon_all_clinical(
    input_path: str | Path,
    subject_id: str,
    *,
    subjects_dir: str | Path,
    threads: int = 4,
    overwrite: bool = False,
) -> Path:
    """Run FreeSurfer 8.2.0 ``recon-all-clinical.sh`` for one subject.

    Produces ``synthSR.mgz``, ``synthseg.mgz``, ``aparc+aseg.mgz`` and ``wmparc.mgz``
    in ``<subjects_dir>/<subject_id>/mri/``. Returns the subject MRI directory.

    Idempotent: if the expected outputs already exist and ``overwrite`` is False,
    the subprocess is skipped.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    subject_mri = subjects_dir / subject_id / "mri"
    expected = [
        subject_mri / "synthSR.mgz",
        subject_mri / "aparc+aseg.mgz",
        subject_mri / "wmparc.mgz",
    ]
    if all(p.is_file() for p in expected) and not overwrite:
        return subject_mri

    command = [
        _resolve_executable("recon-all-clinical.sh"),
        "-i",
        str(input_path),
        "-subjid",
        subject_id,
        "-threads",
        str(threads),
        "-sdir",
        str(subjects_dir),
    ]
    _run_checked(command)

    missing = [p for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"recon-all-clinical did not produce: {missing}")
    return subject_mri


def resample_like(
    source: str | Path,
    reference: str | Path,
    output: str | Path,
    *,
    interpolation: str = "nearest",
) -> Path:
    """Resample ``source`` onto the voxel grid of ``reference`` via ``mri_convert``.

    ``interpolation`` is one of ``nearest``, ``trilinear`` or ``cubic`` (per
    ``mri_convert -rt``). Used to bring FS conformed-space outputs
    (``aparc+aseg.mgz``, ``wmparc.mgz``, ``synthSR.mgz``) back onto the native
    mpMRI grid so they align with the bias-corrected channels and the tumor mask.

    For nearest-neighbour resampling we explicitly force ``-odt int`` so the
    output dtype is independent of the reference image's dtype. Without it,
    ``mri_convert`` silently inherits the reference dtype: a uint16 reference
    triggers an int32-label -> uint16 rescaling that maps wmparc codes
    (max ~5002) onto the full [0, 65535] range, corrupting every label code
    via the MGH header's stored ``slope`` field. This was observed on
    LUMIERE Patient-048 week-000-1, whose registered T1 happens to be
    uint16; all other tp had float32 T1 channels and were unaffected. With
    ``-odt int`` the wmparc lands as int32 regardless of reference dtype
    and the FS label scheme survives the round-trip intact.
    """
    source = Path(source)
    reference = Path(reference)
    output = Path(output)
    for p in (source, reference):
        if not p.is_file():
            raise FileNotFoundError(p)

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _resolve_executable("mri_convert"),
        str(source),
        str(output),
        "-rl",
        str(reference),
        "-rt",
        interpolation,
    ]
    if interpolation == "nearest":
        # Label maps must keep their integer codes intact; -odt int forces
        # int32 output regardless of reference dtype (see docstring).
        command += ["-odt", "int"]
    _run_checked(command)
    if not output.is_file():
        raise FileNotFoundError(output)
    return output


def synthseg_label_lut() -> dict[int, str]:
    """Return the SynthSeg label → name LUT used downstream by Hit-Plot.

    We parse the FreeSurfer LUT that ships with the local installation.
    """
    lut_path = _freesurfer_path("FreeSurferColorLUT.txt")
    if not lut_path.is_file():
        raise FileNotFoundError(lut_path)

    lut: dict[int, str] = {}
    for raw_line in lut_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            label = int(parts[0])
        except ValueError:
            continue
        lut[label] = parts[1]

    if not lut:
        raise ValueError(f"No labels parsed from {lut_path}")
    return lut


# ---------------------------------------------------------------------------
# PR-7c: unified parcellation adapter (freesurfer + dummy backends)
# ---------------------------------------------------------------------------


# A deterministic 9-region subset of FreeSurferColorLUT.txt the dummy
# backend uses. Real FreeSurfer runs emit ~100 regions; this subset is
# enough to give the PR-7f Hit-Plot cohort runner *something* to join
# against in CI without requiring FreeSurfer on the host. Label ids are
# the canonical FreeSurfer codes so a test fixture produced by the
# dummy backend is still schema-identical to a real wmparc.
FS_DUMMY_LUT: dict[int, str] = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    17: "Left-Hippocampus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    53: "Right-Hippocampus",
    16: "Brain-Stem",
    8: "Left-Cerebellum-Cortex",
}


@dataclass(frozen=True)
class ParcellationResult:
    """One cohort subject's resampled ``wmparc`` + its LUT.

    Attributes
    ----------
    label_map
        ``(D, H, W)`` ``int16`` array of FreeSurfer label codes on the
        *native* mpMRI grid (i.e. already resampled out of FS
        conformed space via :func:`resample_like`). ``int16`` matches
        the FreeSurfer wmparc dtype and comfortably fits the ``~3000``
        region codes the full FreeSurferColorLUT reserves.
    affine
        ``(4, 4)`` voxel-to-world affine copied from the input NIfTI
        header so downstream consumers can save the array without
        re-loading the input.
    backend
        ``"freesurfer"`` (real) or ``"dummy"`` (CI / smoke).
    version
        Provenance string. For the real backend this is the
        ``FREESURFER_HOME``-resolved version tag (or ``"unknown"`` if
        we cannot read it); for the dummy backend this is ``"dummy"``.
    lut
        ``{label_id: anatomical_name}`` restricted to labels actually
        present in :attr:`label_map`. This is the same dict shape the
        on-disk ``sub-XXXX_wmparc_lut.json`` exposes after string-ifying
        the integer keys (see
        :func:`scripts.parcellate_all_cohort50._write_lut_json`).
    input_path
        Optional record of the input NIfTI path (handy for sidecar
        provenance).
    """

    label_map: np.ndarray
    affine: np.ndarray
    backend: str
    version: str
    lut: dict[int, str]
    input_path: Path | None = field(default=None)


def _dummy_wmparc(
    shape: tuple[int, int, int],
    *,
    subject_id: str,
) -> np.ndarray:
    """Deterministic synthetic wmparc on a 3-D grid.

    The layout is a stacked pair of nested ellipsoids (left + right
    hemisphere) surrounded by BG, with a small "hippocampus" and
    "brain-stem" blob so every label in :data:`FS_DUMMY_LUT` appears
    in at least a few voxels. The subject-id is mixed into the
    centroid offsets (modulo 8) so two different cohort subjects
    don't produce the exact same synthetic volume and the cohort
    runner's per-subject distinctness is exercised.
    """
    depth, height, width = shape
    cz, cy, cx = depth // 2, height // 2, width // 2
    # Per-subject jitter; deterministic, small.
    try:
        seed = int(subject_id) % 8
    except ValueError:
        seed = 0
    cx = max(0, cx - seed)

    z = np.arange(depth)[:, None, None]
    y = np.arange(height)[None, :, None]
    x = np.arange(width)[None, None, :]

    rz = max(2, depth // 3)
    ry = max(2, height // 3)
    rx = max(2, width // 4)
    half_w = max(4, width // 8)

    # Left hemisphere: WM shell + cortex outer band + hippocampus blob.
    left_cx = max(1, cx - half_w)
    left_r = np.sqrt(
        ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - left_cx) / rx) ** 2,
    )
    # Right hemisphere (mirrored).
    right_cx = min(width - 2, cx + half_w)
    right_r = np.sqrt(
        ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - right_cx) / rx) ** 2,
    )
    # Brain-stem: a thin column along the inferior midline.
    stem_mask = (
        (z < max(2, depth // 3))
        & (np.abs(x - cx) <= max(1, width // 16))
        & (np.abs(y - cy) <= max(1, height // 10))
    )
    # Cerebellum (left): a small ellipsoid below the stem.
    cereb_cz = max(1, depth // 6)
    cereb_cy = max(1, height // 4)
    cereb_cx = max(1, cx - max(1, width // 6))
    cereb_r = np.sqrt(
        ((z - cereb_cz) / max(2, depth // 6)) ** 2
        + ((y - cereb_cy) / max(2, height // 8)) ** 2
        + ((x - cereb_cx) / max(2, width // 10)) ** 2,
    )

    label = np.zeros(shape, dtype=np.int16)

    # Write order matters --- later writes win. Put WM first (shells),
    # then cortex (outer band overwrites WM boundary), then small
    # structures on top so they remain visible.
    label[left_r < 1.0] = 2  # Left-Cerebral-White-Matter
    label[right_r < 1.0] = 41  # Right-Cerebral-White-Matter
    label[(left_r >= 0.85) & (left_r < 1.0)] = 3  # Left-Cerebral-Cortex
    label[(right_r >= 0.85) & (right_r < 1.0)] = 42  # Right-Cerebral-Cortex

    # Hippocampus: small blobs nested inside each hemisphere's WM.
    hippo_r_left = np.sqrt(
        ((z - cz) / max(2, depth // 8)) ** 2
        + ((y - cy) / max(2, height // 8)) ** 2
        + ((x - left_cx) / max(2, width // 16)) ** 2,
    )
    hippo_r_right = np.sqrt(
        ((z - cz) / max(2, depth // 8)) ** 2
        + ((y - cy) / max(2, height // 8)) ** 2
        + ((x - right_cx) / max(2, width // 16)) ** 2,
    )
    label[hippo_r_left < 0.5] = 17  # Left-Hippocampus
    label[hippo_r_right < 0.5] = 53  # Right-Hippocampus

    label[stem_mask] = 16  # Brain-Stem
    label[cereb_r < 1.0] = 8  # Left-Cerebellum-Cortex
    return label


def _restrict_lut_to_present_labels(
    lut: dict[int, str],
    label_map: np.ndarray,
) -> dict[int, str]:
    """Keep only LUT entries whose label appears in ``label_map``.

    Drops the (typically huge) tail of FreeSurferColorLUT that a real
    subject's wmparc never instantiates, so the on-disk JSON stays
    small and the downstream Hit-Plot runner never allocates rows
    for labels that cannot contribute.
    """
    present = np.unique(label_map).tolist()
    return {int(lab): lut[int(lab)] for lab in present if int(lab) in lut}


def _freesurfer_version() -> str:
    """Return the FreeSurfer version string, or ``"unknown"``.

    Reads ``$FREESURFER_HOME/VERSION`` / ``$FREESURFER_HOME/build-stamp.txt``
    best-effort. Never raises --- the cohort sidecar can tolerate
    ``"unknown"`` for provenance.
    """
    fs_home = os.environ.get("FREESURFER_HOME")
    if not fs_home:
        return "unknown"
    for name in ("VERSION", "build-stamp.txt"):
        candidate = Path(fs_home) / name
        if candidate.is_file():
            try:
                return candidate.read_text().strip().splitlines()[0][:200]
            except OSError:
                continue
    return "unknown"


def predict_wmparc(
    *,
    input_nifti: str | Path,
    subject_id: str,
    backend: ParcellationBackend = "freesurfer",
    fs_work_dir: str | Path | None = None,
    threads: int = 4,
    overwrite: bool = False,
) -> ParcellationResult:
    """Produce the native-space wmparc + LUT for one subject.

    Parameters
    ----------
    input_nifti
        Path to the T1-weighted (or any single-channel) NIfTI used to
        seed FreeSurfer. For the UCSF-PDGM cohort this is the
        bias-corrected T1 (``sub-XXXX_T1_bias.nii.gz``); the dummy
        backend only uses it to propagate the affine + grid shape.
    subject_id
        4-digit UCSF-PDGM id (``"0005"``). Used as the FreeSurfer
        ``-subjid``.
    backend
        ``"freesurfer"`` (real; requires ``FREESURFER_HOME`` and
        ``recon-all-clinical.sh`` on PATH) or ``"dummy"`` (CI).
    fs_work_dir
        Scratch ``SUBJECTS_DIR`` for FreeSurfer. Required when
        ``backend == "freesurfer"``; ignored by the dummy backend.
    threads
        ``-threads`` forwarded to ``recon-all-clinical.sh``.
    overwrite
        If True, re-run ``recon-all-clinical`` even if outputs exist
        on disk. The cohort runner exposes this as ``--no-skip-existing``.

    Returns
    -------
    ParcellationResult
        The int16 ``wmparc_native`` array, its affine, the LUT
        restricted to labels present in the array, and backend /
        version provenance.
    """
    input_nifti = Path(input_nifti)
    if not input_nifti.is_file():
        raise FileNotFoundError(input_nifti)

    img = nib.load(str(input_nifti))
    native_shape = tuple(int(s) for s in np.asarray(img.dataobj).shape[:3])
    native_affine = np.asarray(img.affine, dtype=np.float64)

    if backend == "dummy":
        label_map = _dummy_wmparc(native_shape, subject_id=subject_id)
        lut = _restrict_lut_to_present_labels(FS_DUMMY_LUT, label_map)
        return ParcellationResult(
            label_map=label_map,
            affine=native_affine,
            backend="dummy",
            version="dummy",
            lut=lut,
            input_path=input_nifti,
        )

    if backend != "freesurfer":
        raise ValueError(
            f"Unknown parcellation backend: {backend!r} (expected 'freesurfer' or 'dummy')",
        )

    if fs_work_dir is None:
        raise ValueError(
            "backend='freesurfer' requires fs_work_dir (scratch SUBJECTS_DIR).",
        )
    fs_work_dir = Path(fs_work_dir)
    subject_mri = recon_all_clinical(
        input_nifti,
        subject_id,
        subjects_dir=fs_work_dir,
        threads=threads,
        overwrite=overwrite,
    )
    wmparc_mgz = subject_mri / "wmparc.mgz"
    if not wmparc_mgz.is_file():
        raise FileNotFoundError(wmparc_mgz)

    # Resample FS conformed-space wmparc back onto the native mpMRI
    # grid via nearest-neighbour interpolation (label maps must never
    # be linearly interpolated); the output sits next to the FS outputs.
    wmparc_native_mgz = subject_mri / "wmparc_native.mgz"
    resample_like(
        wmparc_mgz,
        input_nifti,
        wmparc_native_mgz,
        interpolation="nearest",
    )

    resampled = nib.load(str(wmparc_native_mgz))
    label_map = np.asarray(resampled.dataobj).astype(np.int16)
    if label_map.shape != native_shape:
        raise RuntimeError(
            f"resampled wmparc shape {label_map.shape} != native grid {native_shape}",
        )

    full_lut = synthseg_label_lut()
    lut = _restrict_lut_to_present_labels(full_lut, label_map)
    return ParcellationResult(
        label_map=label_map,
        affine=native_affine,
        backend="freesurfer",
        version=_freesurfer_version(),
        lut=lut,
        input_path=input_nifti,
    )
