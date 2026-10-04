"""I/O helpers — NIfTI loading, BIDS-ish path resolution, dataset descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from hpgs.io.cohort import CohortMetadata, cohort_summary, load_cohort_metadata

__all__ = [
    "CohortMetadata",
    "SubjectInputs",
    "cohort_summary",
    "derivatives_dir",
    "load_cohort_metadata",
    "load_nifti",
    "load_nifti_canonical",
    "normalize_subject_id",
    "orig_voxel_to_ras",
    "resolve_subject_inputs",
    "save_nifti",
    "subject_dir",
]


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return ``(data, affine)``."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """Save ``data`` with ``affine`` as a compressed NIfTI."""
    nib.save(nib.Nifti1Image(data, affine), str(path))


def load_nifti_canonical(
    path: str | Path,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    """Load a NIfTI and reorient it to canonical RAS voxel ordering.

    Returns ``(data_ras, original_shape, transform)`` where ``transform`` is the
    nibabel orientation transform that maps the original array to RAS. Callers
    that have voxel coordinates expressed in the *original* array can translate
    them with :func:`orig_voxel_to_ras`.
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    orig_ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    data_ras = nib.orientations.apply_orientation(data, transform)
    return data_ras, tuple(int(s) for s in data.shape[:3]), transform


def orig_voxel_to_ras(
    coord: tuple[int, int, int],
    original_shape: tuple[int, ...],
    transform: np.ndarray,
) -> tuple[int, int, int]:
    """Translate a voxel coord from the original NIfTI frame into canonical RAS.

    ``transform`` is the 3x2 orientation transform returned by
    :func:`load_nifti_canonical` (or by ``nib.orientations.ornt_transform``).
    """
    out = [0, 0, 0]
    for new_axis in range(3):
        old_axis = int(transform[new_axis, 0])
        flip = int(transform[new_axis, 1])
        c = int(coord[old_axis])
        if flip == -1:
            c = int(original_shape[old_axis]) - 1 - c
        out[new_axis] = c
    return out[0], out[1], out[2]


def subject_dir(root: str | Path, subject_id: str) -> Path:
    """Return the per-subject directory under ``root``.

    ``subject_id`` is the 4-digit UCSF-PDGM identifier (e.g. ``"0005"``).
    """
    p = Path(root) / f"sub-{subject_id}"
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


@dataclass(frozen=True)
class SubjectInputs:
    """Resolved UCSF-PDGM subject inputs in the extracted flat layout."""

    subject_id: str
    subject_dir: Path
    channels: dict[str, Path]
    tumor_segmentation: Path
    reference_mask: Path | None = None


def normalize_subject_id(subject_id: str) -> str:
    """Normalise ``subject_id`` to a 4-digit UCSF-PDGM identifier."""
    normalized = subject_id.removeprefix("sub-").strip()
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError(f"Expected a 4-digit subject id, got: {subject_id!r}")
    return normalized


def derivatives_dir(root: str | Path, subject_id: str) -> Path:
    """Return and create the derivatives directory for a subject."""
    normalized = normalize_subject_id(subject_id)
    path = Path(root) / f"sub-{normalized}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_subject_inputs(
    root: str | Path,
    subject_id: str,
    *,
    channel_names: list[str],
    tumor_segmentation_name: str = "tumor_segmentation",
    reference_mask_name: str | None = None,
) -> SubjectInputs:
    """Resolve expected UCSF-PDGM files for one subject.

    Files are expected in the extracted flat layout:
    ``sub-0005/sub-0005_T1_bias.nii.gz`` etc.
    """
    normalized = normalize_subject_id(subject_id)
    subdir = subject_dir(root, normalized)
    prefix = f"sub-{normalized}_"

    def expected_path(name: str) -> Path:
        path = subdir / f"{prefix}{name}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    channels = {name: expected_path(name) for name in channel_names}
    tumor_segmentation = expected_path(tumor_segmentation_name)
    reference_mask = expected_path(reference_mask_name) if reference_mask_name else None

    return SubjectInputs(
        subject_id=normalized,
        subject_dir=subdir,
        channels=channels,
        tumor_segmentation=tumor_segmentation,
        reference_mask=reference_mask,
    )
