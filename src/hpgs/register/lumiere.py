"""Longitudinal coregistration orchestrator for LUMIERE Patient-048.

Purpose
-------
Bring all six LUMIERE Patient-048 timepoints into a single subject reference
space (the pre-operative T1, ``week-000-1``) using a per-timepoint rigid
transform estimated on a chosen "moving channel" (T1 by default; CT1 at
``week-023`` because that timepoint's T1 is acquired as a 6 mm-thick axial
stack and is a poor target for high-resolution rigid registration).

For every timepoint, the same transform is applied to all four MR channels
(linear interpolation) and, optionally, to the two LUMIERE-team legacy
segmentations (HD-GLIO-AUTO registered + DeepBraTumIA native CT1-grid;
nearest-neighbour interpolation to preserve labels). The DeepBraTumIA
**native CT1** mask is the only DBT variant that lives in the same
physical space as the staged ``CT1.nii.gz`` and therefore warps cleanly
through the rigid CT1 -> ref transform stack; the SRI24 ``atlas`` variant
that ships in the same archive is intentionally not consumed here (see
``scripts/extract_lumiere_p048.py`` for the rationale).

Inputs (consumed but not modified)
----------------------------------
- ``configs/lumiere_p048_timepoints.yaml`` -- cohort manifest produced in
  Phase 1.5 (six timepoints, RANO timeline, channel list, reference timepoint).
- ``data/lumiere_p048/tp-<week>/tp-<week>_<MOD>.nii.gz`` -- staged channels
  produced by ``scripts/extract_lumiere_p048.py``.
- (Optional) ``data/lumiere_p048/tp-<week>/tp-<week>_hdglio_*.nii.gz`` and
  ``..._deepbratumia_*.nii.gz`` -- legacy segmentations, only consumed when
  ``include_legacy_seg=True``.

Outputs (under ``derivatives_root``)
------------------------------------
::

    derivatives_root/
        manifest.json                        # provenance for the whole run
        tp-week-000-1/                       # reference timepoint (no registration)
            canonical/{T1,CT1,T2,FLAIR}.nii.gz   # canonical-RAS pre-images
            tp-week-000-1_<MOD>_to_ref.nii.gz    # resampled to ref T1 grid
            registration_qc.json                 # per-channel grid + similarity
        tp-week-013/                         # ... and similarly for each non-ref tp
            canonical/{T1,CT1,T2,FLAIR}.nii.gz
            transforms/<prefix>{fwd,inv}_*.mat   # ANTs Rigid transform stack
            transforms/<prefix>warped.nii.gz     # warped moving channel (QC)
            tp-week-013_<MOD>_to_ref.nii.gz
            tp-week-013_hdglio_to_ref_segmentation.nii.gz       # if include_legacy_seg
            tp-week-013_deepbratumia_to_ref_segmentation.nii.gz # if include_legacy_seg
            registration_qc.json

Determinism
-----------
ANTsPyX Rigid uses ``random_seed=DEFAULT_SEED`` (20260415). Repeated runs on
the same inputs produce byte-identical transforms.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ants
import nibabel as nib
import numpy as np
import yaml

from hpgs.register import (
    DEFAULT_SEED,
    apply_transform,
    register_rigid,
    to_canonical_ras_file,
)

DEFAULT_MOVING_CHANNEL_OVERRIDES: dict[str, str] = {
    # week-023 T1 is 280x320x24 @ 0.688x0.688x6 mm (thick axial); CT1 at the
    # same timepoint is 256x256x160 @ ~1 mm and is a far better moving target.
    "week-023": "CT1",
}

LEGACY_SEG_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("hdglio_registered_segmentation.nii.gz", "hdglio_to_ref_segmentation.nii.gz"),
    ("deepbratumia_native_segmentation.nii.gz", "deepbratumia_to_ref_segmentation.nii.gz"),
)


@dataclass
class TimepointResult:
    """Per-timepoint registration provenance + QC."""

    name: str
    is_reference: bool
    moving_channel: str | None
    transform_paths: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _load_manifest(path: Path) -> dict:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    if "reference_timepoint" not in manifest:
        raise ValueError(f"Manifest is missing 'reference_timepoint': {path}")
    return manifest


def _channel_path(stage_root: Path, tp_name: str, channel: str) -> Path:
    return stage_root / f"tp-{tp_name}" / f"tp-{tp_name}_{channel}.nii.gz"


def _legacy_seg_path(stage_root: Path, tp_name: str, src_suffix: str) -> Path:
    return stage_root / f"tp-{tp_name}" / f"tp-{tp_name}_{src_suffix}"


def _grid_info(path: Path) -> dict[str, Any]:
    img = nib.load(str(path))
    return {
        "shape": [int(s) for s in img.shape[:3]],
        "voxel_size_mm": [float(round(z, 4)) for z in img.header.get_zooms()[:3]],
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
    }


def _grids_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["shape"] == b["shape"]
        and a["voxel_size_mm"] == b["voxel_size_mm"]
        and a["axcodes"] == b["axcodes"]
    )


def _correlation(reference_path: Path, candidate_path: Path) -> float:
    """Pearson correlation between reference and candidate volumes (must share grid)."""
    ref = np.asarray(nib.load(str(reference_path)).dataobj, dtype=np.float64).ravel()
    cand = np.asarray(nib.load(str(candidate_path)).dataobj, dtype=np.float64).ravel()
    if ref.shape != cand.shape:
        raise ValueError(f"Cannot compute correlation: shape mismatch {ref.shape} vs {cand.shape}")
    if ref.std() == 0 or cand.std() == 0:
        return float("nan")
    return float(np.corrcoef(ref, cand)[0, 1])


def _prepare_canonical_inputs(
    *, stage_root: Path, tp_name: str, channels: list[str], canonical_dir: Path
) -> dict[str, Path]:
    """Reorient every channel of one timepoint to canonical RAS. Returns {ch -> path}."""
    canonical_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for ch in channels:
        src = _channel_path(stage_root, tp_name, ch)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = canonical_dir / f"{ch}.nii.gz"
        to_canonical_ras_file(src, dst)
        out[ch] = dst
    return out


def _resample_via_identity(
    *, source: Path, reference: Path, out_path: Path, interpolator: str
) -> Path:
    """Resample ``source`` onto ``reference`` grid with no spatial transform.

    Used for the reference timepoint, where every channel is already in the
    correct physical space but may be on a slightly different voxel grid than
    the reference T1 (LUMIERE preprocessing keeps physical spaces consistent
    within a timepoint but does not always force a single voxel grid).
    """
    ref_img = ants.image_read(str(reference))
    src_img = ants.image_read(str(source))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warped = ants.resample_image_to_target(
        image=src_img,
        target=ref_img,
        interp_type=interpolator,
    )
    ants.image_write(warped, str(out_path))
    return out_path


def _interp_for_apply_transform(role: str) -> str:
    return "nearestNeighbor" if role == "legacy_seg" else "linear"


def _interp_for_resample(role: str) -> str:
    # ants.resample_image_to_target uses a different vocabulary than
    # ants.apply_transforms ("linear" vs "nearestNeighbor" -> "genericLabel"/"linear").
    return "genericLabel" if role == "legacy_seg" else "linear"


def _process_reference_timepoint(
    *,
    tp_name: str,
    stage_root: Path,
    derivatives_root: Path,
    channels: list[str],
    include_legacy_seg: bool,
    overwrite: bool,
) -> TimepointResult:
    out_dir = derivatives_root / f"tp-{tp_name}"
    canonical_dir = out_dir / "canonical"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = _prepare_canonical_inputs(
        stage_root=stage_root, tp_name=tp_name, channels=channels, canonical_dir=canonical_dir
    )
    ref_t1_path = canonical["T1"]
    ref_grid = _grid_info(ref_t1_path)

    result = TimepointResult(name=tp_name, is_reference=True, moving_channel=None)
    result.qc["reference_grid"] = ref_grid
    result.qc["channels"] = {}

    for ch in channels:
        out_path = out_dir / f"tp-{tp_name}_{ch}_to_ref.nii.gz"
        if out_path.exists() and not overwrite:
            result.outputs[ch] = str(out_path)
            result.qc["channels"][ch] = {
                "grid": _grid_info(out_path),
                "grid_matches_reference": _grids_match(_grid_info(out_path), ref_grid),
                "status": "skipped_existing",
            }
            continue
        if ch == "T1":
            shutil.copy2(canonical[ch], out_path)
        else:
            _resample_via_identity(
                source=canonical[ch],
                reference=ref_t1_path,
                out_path=out_path,
                interpolator=_interp_for_resample("channel"),
            )
        result.outputs[ch] = str(out_path)
        out_grid = _grid_info(out_path)
        result.qc["channels"][ch] = {
            "grid": out_grid,
            "grid_matches_reference": _grids_match(out_grid, ref_grid),
            "status": "written",
        }

    if include_legacy_seg:
        _process_reference_legacy_segs(
            tp_name=tp_name,
            stage_root=stage_root,
            out_dir=out_dir,
            ref_t1_path=ref_t1_path,
            ref_grid=ref_grid,
            result=result,
            overwrite=overwrite,
        )

    return result


def _process_reference_legacy_segs(
    *,
    tp_name: str,
    stage_root: Path,
    out_dir: Path,
    ref_t1_path: Path,
    ref_grid: dict[str, Any],
    result: TimepointResult,
    overwrite: bool,
) -> None:
    result.qc.setdefault("legacy_segs", {})
    for src_suffix, out_suffix in LEGACY_SEG_SUFFIXES:
        src = _legacy_seg_path(stage_root, tp_name, src_suffix)
        if not src.is_file():
            result.qc["legacy_segs"][src_suffix] = {"status": "missing_source"}
            continue
        canonical_seg = out_dir / "canonical" / f"seg_{src_suffix}"
        to_canonical_ras_file(src, canonical_seg)
        out_path = out_dir / f"tp-{tp_name}_{out_suffix}"
        if out_path.exists() and not overwrite:
            result.qc["legacy_segs"][src_suffix] = {
                "out": str(out_path),
                "grid": _grid_info(out_path),
                "status": "skipped_existing",
            }
            continue
        _resample_via_identity(
            source=canonical_seg,
            reference=ref_t1_path,
            out_path=out_path,
            interpolator=_interp_for_resample("legacy_seg"),
        )
        out_grid = _grid_info(out_path)
        result.qc["legacy_segs"][src_suffix] = {
            "out": str(out_path),
            "grid": out_grid,
            "grid_matches_reference": _grids_match(out_grid, ref_grid),
            "status": "written",
        }


def _process_moving_timepoint(
    *,
    tp_name: str,
    stage_root: Path,
    derivatives_root: Path,
    channels: list[str],
    moving_channel: str,
    ref_t1_path: Path,
    ref_grid: dict[str, Any],
    include_legacy_seg: bool,
    seed: int,
    overwrite: bool,
) -> TimepointResult:
    out_dir = derivatives_root / f"tp-{tp_name}"
    canonical_dir = out_dir / "canonical"
    transforms_dir = out_dir / "transforms"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = _prepare_canonical_inputs(
        stage_root=stage_root, tp_name=tp_name, channels=channels, canonical_dir=canonical_dir
    )
    if moving_channel not in canonical:
        raise ValueError(
            f"moving_channel={moving_channel!r} for tp={tp_name!r} not in channels={channels!r}"
        )

    result = TimepointResult(name=tp_name, is_reference=False, moving_channel=moving_channel)
    result.qc["reference_grid"] = ref_grid
    result.qc["moving_channel"] = moving_channel
    result.qc["pre_registration_grid"] = _grid_info(canonical[moving_channel])

    reg = register_rigid(
        moving=canonical[moving_channel],
        fixed=ref_t1_path,
        out_dir=transforms_dir,
        out_prefix=f"tp-{tp_name}_to_ref_",
        seed=seed,
    )
    result.transform_paths = [str(p) for p in reg["fwdtransforms"]]
    result.outputs[f"{moving_channel}_warped_qc"] = str(reg["warped"])

    moving_warped_corr = _correlation(ref_t1_path, reg["warped"])
    result.qc["moving_to_ref_correlation"] = moving_warped_corr

    result.qc["channels"] = {}
    for ch in channels:
        out_path = out_dir / f"tp-{tp_name}_{ch}_to_ref.nii.gz"
        if out_path.exists() and not overwrite:
            result.outputs[ch] = str(out_path)
            result.qc["channels"][ch] = {
                "grid": _grid_info(out_path),
                "grid_matches_reference": _grids_match(_grid_info(out_path), ref_grid),
                "status": "skipped_existing",
            }
            continue
        apply_transform(
            moving=canonical[ch],
            reference=ref_t1_path,
            transform_paths=reg["fwdtransforms"],
            out_path=out_path,
            interpolator=_interp_for_apply_transform("channel"),
        )
        result.outputs[ch] = str(out_path)
        out_grid = _grid_info(out_path)
        result.qc["channels"][ch] = {
            "grid": out_grid,
            "grid_matches_reference": _grids_match(out_grid, ref_grid),
            "status": "written",
        }

    if include_legacy_seg:
        _process_moving_legacy_segs(
            tp_name=tp_name,
            stage_root=stage_root,
            out_dir=out_dir,
            ref_t1_path=ref_t1_path,
            ref_grid=ref_grid,
            transform_paths=reg["fwdtransforms"],
            result=result,
            overwrite=overwrite,
        )

    return result


def _process_moving_legacy_segs(
    *,
    tp_name: str,
    stage_root: Path,
    out_dir: Path,
    ref_t1_path: Path,
    ref_grid: dict[str, Any],
    transform_paths: list[Path],
    result: TimepointResult,
    overwrite: bool,
) -> None:
    result.qc.setdefault("legacy_segs", {})
    for src_suffix, out_suffix in LEGACY_SEG_SUFFIXES:
        src = _legacy_seg_path(stage_root, tp_name, src_suffix)
        if not src.is_file():
            result.qc["legacy_segs"][src_suffix] = {"status": "missing_source"}
            continue
        canonical_seg = out_dir / "canonical" / f"seg_{src_suffix}"
        to_canonical_ras_file(src, canonical_seg)
        out_path = out_dir / f"tp-{tp_name}_{out_suffix}"
        if out_path.exists() and not overwrite:
            result.qc["legacy_segs"][src_suffix] = {
                "out": str(out_path),
                "grid": _grid_info(out_path),
                "status": "skipped_existing",
            }
            continue
        apply_transform(
            moving=canonical_seg,
            reference=ref_t1_path,
            transform_paths=transform_paths,
            out_path=out_path,
            interpolator=_interp_for_apply_transform("legacy_seg"),
        )
        out_grid = _grid_info(out_path)
        result.qc["legacy_segs"][src_suffix] = {
            "out": str(out_path),
            "grid": out_grid,
            "grid_matches_reference": _grids_match(out_grid, ref_grid),
            "status": "written",
        }


def _write_qc(result: TimepointResult, derivatives_root: Path) -> Path:
    out_dir = derivatives_root / f"tp-{result.name}"
    qc_path = out_dir / "registration_qc.json"
    payload = {
        "timepoint": result.name,
        "is_reference": result.is_reference,
        "moving_channel": result.moving_channel,
        "transform_paths": result.transform_paths,
        "outputs": result.outputs,
        "qc": result.qc,
        "warnings": result.warnings,
    }
    qc_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return qc_path


def register_subject_longitudinal(
    *,
    manifest_path: Path,
    stage_root: Path,
    derivatives_root: Path,
    moving_channel_overrides: dict[str, str] | None = None,
    include_legacy_seg: bool = False,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register every non-reference timepoint to the reference timepoint's T1.

    See module docstring for the full input/output contract.

    Returns
    -------
    dict
        Provenance for the whole run, also written to
        ``derivatives_root/manifest.json``.
    """
    manifest = _load_manifest(manifest_path)
    channels = list(manifest.get("channels", ["T1", "CT1", "T2", "FLAIR"]))
    ref_tp = manifest["reference_timepoint"]
    overrides = {**DEFAULT_MOVING_CHANNEL_OVERRIDES, **(moving_channel_overrides or {})}

    if not stage_root.is_dir():
        raise FileNotFoundError(f"--stage-root does not exist: {stage_root}")
    derivatives_root.mkdir(parents=True, exist_ok=True)

    # Reference first.
    ref_result = _process_reference_timepoint(
        tp_name=ref_tp,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
        channels=channels,
        include_legacy_seg=include_legacy_seg,
        overwrite=overwrite,
    )
    ref_t1_path = derivatives_root / f"tp-{ref_tp}" / "canonical" / "T1.nii.gz"
    ref_grid = _grid_info(ref_t1_path)
    _write_qc(ref_result, derivatives_root)

    results: list[TimepointResult] = [ref_result]
    for tp in manifest["timepoints"]:
        tp_name = tp["name"]
        if tp_name == ref_tp:
            continue
        moving_channel = overrides.get(tp_name, "T1")
        tp_result = _process_moving_timepoint(
            tp_name=tp_name,
            stage_root=stage_root,
            derivatives_root=derivatives_root,
            channels=channels,
            moving_channel=moving_channel,
            ref_t1_path=ref_t1_path,
            ref_grid=ref_grid,
            include_legacy_seg=include_legacy_seg,
            seed=seed,
            overwrite=overwrite,
        )
        _write_qc(tp_result, derivatives_root)
        results.append(tp_result)

    provenance = {
        "schema_version": "1.0",
        "cohort_id": manifest.get("cohort_id", "lumiere_p048"),
        "patient_id": manifest.get("patient_id", "048"),
        "manifest_path": str(manifest_path),
        "stage_root": str(stage_root),
        "derivatives_root": str(derivatives_root),
        "reference_timepoint": ref_tp,
        "channels": channels,
        "moving_channel_overrides": overrides,
        "include_legacy_seg": include_legacy_seg,
        "seed": seed,
        "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "timepoints": [
            {
                "name": r.name,
                "is_reference": r.is_reference,
                "moving_channel": r.moving_channel,
                "transform_paths": r.transform_paths,
                "outputs": r.outputs,
                "qc": r.qc,
                "warnings": r.warnings,
            }
            for r in results
        ],
    }
    out_manifest = derivatives_root / "manifest.json"
    out_manifest.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance
