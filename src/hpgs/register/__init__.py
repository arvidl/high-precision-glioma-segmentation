"""Registration helpers — thin ANTsPyX wrappers used to align channels and atlases.

Functions
---------
- :func:`to_canonical_ras_file`  -- reorient a NIfTI on disk to canonical RAS.
- :func:`register_rigid`         -- ANTsPyX Rigid registration of moving -> fixed.
- :func:`apply_transform`        -- apply a saved ANTs transform to another volume.

Determinism
-----------
``register_rigid`` forwards ``random_seed`` to ANTs for reproducibility, but rigid
registration is not bit-identical across repeated calls on the same inputs (transform
files can differ slightly between runs). Treat ``seed`` as the project default
(:data:`DEFAULT_SEED`, from ``AGENTS.md``) for orchestration consistency, not as a
guarantee of identical transforms. :func:`to_canonical_ras_file` and
:func:`apply_transform` given the same inputs and transform files are deterministic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import ants
import nibabel as nib

DEFAULT_SEED: int = 20260415

__all__ = [
    "DEFAULT_SEED",
    "apply_transform",
    "register_rigid",
    "to_canonical_ras_file",
]


def to_canonical_ras_file(src: str | Path, dst: str | Path) -> Path:
    """Reorient a NIfTI on disk to canonical RAS and write to ``dst``.

    Wraps :func:`nibabel.as_closest_canonical`. Idempotent: if the input is
    already in canonical RAS, the output is byte-identical to the input but
    written to ``dst``.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    img = nib.load(str(src_path))
    canonical = nib.as_closest_canonical(img)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(canonical, str(dst_path))
    return dst_path


def register_rigid(
    moving: str | Path,
    fixed: str | Path,
    out_dir: str | Path,
    *,
    out_prefix: str = "moving_to_fixed_",
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> dict[str, Path]:
    """Rigid registration of ``moving`` onto ``fixed`` using ANTsPyX.

    Both inputs are read with ``ants.image_read`` (ANTs handles its own
    orientation internally; pass canonical-RAS files via :func:`to_canonical_ras_file`
    if you want to enforce a uniform header orientation across runs).

    The ``seed`` is passed to ANTs as ``random_seed``. It stabilises optimisation but
    does not make successive registrations byte-identical on the same pair of volumes;
    saved forward transforms may differ slightly between runs. For label volumes, use
    multi-voxel regions for rare labels—single-voxel labels can disappear under
    ``nearestNeighbor`` when the estimated transform shifts.

    Parameters
    ----------
    seed
        Pseudo-random seed forwarded to ``ants.registration`` (default
        :data:`DEFAULT_SEED`).

    Returns
    -------
    dict
        Mapping with keys ``"warped"``, ``"fwdtransforms"``, ``"invtransforms"``;
        each value is a :class:`pathlib.Path` written under ``out_dir`` (the
        ANTs registration outputs are copied there with the ``out_prefix``).

    Raises
    ------
    ImportError
        If ``antspyx`` is not installed.
    FileNotFoundError
        If ``moving`` or ``fixed`` does not exist.
    """
    moving_path = Path(moving)
    fixed_path = Path(fixed)
    if not moving_path.is_file():
        raise FileNotFoundError(moving_path)
    if not fixed_path.is_file():
        raise FileNotFoundError(fixed_path)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fixed_img = ants.image_read(str(fixed_path))
    moving_img = ants.image_read(str(moving_path))

    reg = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform="Rigid",
        random_seed=seed,
        verbose=verbose,
    )

    warped_path = out_path / f"{out_prefix}warped.nii.gz"
    ants.image_write(reg["warpedmovout"], str(warped_path))

    fwd_paths = [_persist_transform(p, out_path, f"{out_prefix}fwd_") for p in reg["fwdtransforms"]]
    inv_paths = [_persist_transform(p, out_path, f"{out_prefix}inv_") for p in reg["invtransforms"]]

    return {
        "warped": warped_path,
        "fwdtransforms": fwd_paths,
        "invtransforms": inv_paths,
    }


def apply_transform(
    moving: str | Path,
    reference: str | Path,
    transform_paths: list[str | Path],
    out_path: str | Path,
    *,
    interpolator: str = "linear",
) -> Path:
    """Apply a stack of saved transforms via ``ants.apply_transforms``.

    ``transform_paths`` is the list returned in
    ``register_rigid(...)["fwdtransforms"]``. Use ``interpolator="nearestNeighbor"``
    when warping label volumes (segmentations, parcellations) to preserve labels.
    """
    moving_path = Path(moving)
    reference_path = Path(reference)
    out_file = Path(out_path)
    if not moving_path.is_file():
        raise FileNotFoundError(moving_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    reference_img = ants.image_read(str(reference_path))
    moving_img = ants.image_read(str(moving_path))
    warped = ants.apply_transforms(
        fixed=reference_img,
        moving=moving_img,
        transformlist=[str(p) for p in transform_paths],
        interpolator=interpolator,
    )
    ants.image_write(warped, str(out_file))
    return out_file


def _persist_transform(src: str, out_dir: Path, prefix: str) -> Path:
    """Copy an ANTs-emitted transform file (in /tmp by default) into ``out_dir``."""
    src_path = Path(src)
    dst_path = out_dir / f"{prefix}{src_path.name}"
    shutil.copy2(src_path, dst_path)
    return dst_path
