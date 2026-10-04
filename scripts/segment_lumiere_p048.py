"""Run the unified MONAI Bundle BraTS-3 SegResNet on every LUMIERE Patient-048 timepoint.

This is the LUMIERE analogue of ``scripts/segment_all_cohort50.py``: it
fans the round-2 unified DL segmenter (``hpgs.segment.unified.predict_brats3``)
out over the six longitudinal timepoints of LUMIERE Patient-048 and writes
a per-timepoint artefact contract that mirrors the UCSF-PDGM contract
(label map + sigmoid probabilities + sidecar JSON), plus a single
cohort-level ``volumes.csv`` that the round-3 longitudinal panels
(Figures 6 + 11) consume directly.

Inputs (consumed but not modified)
----------------------------------
- ``configs/lumiere_p048_timepoints.yaml`` -- cohort manifest (single
  source of truth for tp ordering, days, RANO, channel list).
- ``data/lumiere_p048/derivatives/tp-<week>/tp-<week>_{T1,CT1,T2,FLAIR}_to_ref.nii.gz``
  -- registered channels produced by ``scripts/register_lumiere_p048.py``.
  ``CT1`` (LUMIERE name) is mapped to ``T1c`` (BraTS bundle name) at the
  ``predict_brats3`` boundary; the bundle's required channel order is
  ``(T1c, T1, T2, FLAIR)``.

Outputs (under the existing ``derivatives_root``)
-------------------------------------------------
::

    derivatives_root/
        segmentation_summary.csv             # one row per tp (Figures 6 + 11)
        segmentation_summary.json            # full provenance for the run
        tp-<week>/
            seg_dl/
                tp-<week>_seg_brats3_dl.nii.gz       # uint8 BraTS multi-class
                tp-<week>_seg_brats3_dl_probs.nii.gz # 4-D float32 (TC, WT, ET)
                tp-<week>_seg_brats3_dl.json         # provenance + volumes

The ``segmentation_summary.csv`` columns are:

    timepoint, days_since_baseline, rano, voxel_volume_mm3,
    vol_WT_ml, vol_TC_ml, vol_ET_ml, vol_NCR_ml, vol_ED_ml

Backends
--------
``--backend monai_bundle`` (default)
    Real DL inference. First call will download the
    ``brats_mri_segmentation`` bundle (~140 MB) into ``--bundle-dir``.
``--backend dummy``
    Deterministic synthetic field. Exercises the channel-ordering and
    file-IO contract end-to-end without weights or network. Used by the
    integration smoke test.

Device caveat (Apple silicon / MPS)
-----------------------------------
The MONAI ``brats_mri_segmentation`` v0.4.8 SegResNet collapses to
all-negative logits on the Apple Metal (MPS) backend with PyTorch >= 2.x:
on the same Patient-048 week-000-1 input, MPS returns
``logits in [-11.5, -4.2]`` and zero tumour everywhere, while CPU returns
``logits in [-12.8, +14.9]`` and the expected ~140 mL whole-tumour. The
default device is therefore ``cpu``. Pass ``--device cuda`` on a GPU box
or ``--device mps`` if you want to re-test the MPS path on a future
PyTorch version. ``--device auto`` will still pick CUDA where available.

Usage
-----

    # Sanity run on one timepoint (good for first-time validation):
    uv run python scripts/segment_lumiere_p048.py --timepoint week-000-1

    # Full longitudinal segmentation (5 - 30 minutes on MPS / CPU):
    uv run python scripts/segment_lumiere_p048.py

    # Re-aggregate volumes.csv from existing per-tp segmentations
    # without re-running inference:
    uv run python scripts/segment_lumiere_p048.py --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml
from nibabel.orientations import (
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)

from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR, derive_subregions
from hpgs.segment.unified import (
    CHANNEL_ORDER,
    PROB_CHANNEL_ORDER,
    SegmentationResult,
    predict_brats3,
)

# The MONAI ``brats_mri_segmentation`` bundle was trained on BraTS-2018 data,
# which lives in SRI24 atlas space: shape ``(240, 240, 155)`` and orientation
# ``(L, P, S)``. Our LUMIERE registration pipeline produces canonical RAS
# (``hpgs.register.to_canonical_ras_file``); feeding RAS straight into the
# bundle inverts the L--R and A--P axes and the model sees a mirrored brain on
# a permuted slicing plane, which empirically flattens every sigmoid output
# below threshold. We therefore reorient each channel to LPS just-in-time
# before inference and reorient the predicted label map / probabilities back
# to the input's RAS frame so the on-disk derivatives stay consistent with
# the registered channels next to them.
BUNDLE_ORIENTATION: tuple[str, str, str] = ("L", "P", "S")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_DERIVATIVES = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"

SEG_SIDECAR_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _channel_path(derivatives_root: Path, tp_name: str, channel: str) -> Path:
    return derivatives_root / f"tp-{tp_name}" / f"tp-{tp_name}_{channel}_to_ref.nii.gz"


def _seg_dir(derivatives_root: Path, tp_name: str) -> Path:
    return derivatives_root / f"tp-{tp_name}" / "seg_dl"


def _seg_paths(derivatives_root: Path, tp_name: str) -> dict[str, Path]:
    seg_dir = _seg_dir(derivatives_root, tp_name)
    base = f"tp-{tp_name}_seg_brats3_dl"
    return {
        "label_map": seg_dir / f"{base}.nii.gz",
        "probabilities": seg_dir / f"{base}_probs.nii.gz",
        "sidecar": seg_dir / f"{base}.json",
    }


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def _voxel_spacing_mm(affine: np.ndarray) -> tuple[float, float, float]:
    diag = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return float(diag[0]), float(diag[1]), float(diag[2])


def _volumes_ml_from_label_map(label_map: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    """WT/TC/ET/NCR/ED volumes in mL from a BraTS multi-class label map."""
    masks = derive_subregions(label_map)
    ncr = (label_map == LABEL_NCR).sum()
    ed = (label_map == LABEL_ED).sum()
    factor = voxel_volume_mm3 / 1000.0
    return {
        "vol_WT_ml": round(float(masks.wt.sum()) * factor, 4),
        "vol_TC_ml": round(float(masks.tc.sum()) * factor, 4),
        "vol_ET_ml": round(float(masks.et.sum()) * factor, 4),
        "vol_NCR_ml": round(float(ncr) * factor, 4),
        "vol_ED_ml": round(float(ed) * factor, 4),
    }


# ---------------------------------------------------------------------------
# Per-timepoint segmentation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Orientation helpers (RAS <-> LPS reorientation around the bundle call).
# ---------------------------------------------------------------------------


def _reorient_image_to(
    img: nib.Nifti1Image, target_axcodes: tuple[str, str, str]
) -> nib.Nifti1Image:
    """Reorient a Nifti1Image to ``target_axcodes`` (e.g. ``('L','P','S')``)."""
    target = axcodes2ornt(target_axcodes)
    current = io_orientation(img.affine)
    transform = ornt_transform(current, target)
    return img.as_reoriented(transform)


def _reorient_file(src: Path, dst: Path, target_axcodes: tuple[str, str, str]) -> nib.Nifti1Image:
    """Read ``src``, reorient to ``target_axcodes``, save to ``dst``, return image."""
    img = nib.load(str(src))
    out = _reorient_image_to(img, target_axcodes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return out


def _reorient_array_to_reference(
    data: np.ndarray,
    affine_in: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient ``data`` (with affine ``affine_in``) into ``reference_img``'s frame.

    Used to flip the bundle's LPS-space prediction back into the RAS-space frame
    of the registered channels on disk. Returns ``(data_out, affine_out)``.
    """
    spatial_shape = data.shape[:3]
    in_img = nib.Nifti1Image(
        np.zeros(spatial_shape, dtype=np.uint8) if data.ndim > 3 else data,
        affine_in,
    )
    target_axcodes = nib.aff2axcodes(reference_img.affine)
    target = axcodes2ornt(target_axcodes)
    current = io_orientation(in_img.affine)
    transform = ornt_transform(current, target)

    if data.ndim == 3:
        out_img = in_img.as_reoriented(transform)
        return np.asarray(out_img.dataobj), out_img.affine

    if data.ndim != 4:
        raise ValueError(f"_reorient_array_to_reference: unsupported ndim {data.ndim}")
    # Reorient each 3-D channel separately and re-stack.
    channels = []
    out_affine: np.ndarray | None = None
    for c in range(data.shape[3]):
        ch_img = nib.Nifti1Image(data[..., c], affine_in)
        ch_out = ch_img.as_reoriented(transform)
        channels.append(np.asarray(ch_out.dataobj))
        out_affine = ch_out.affine
    assert out_affine is not None
    return np.stack(channels, axis=-1), out_affine


def _check_inputs(derivatives_root: Path, tp_name: str) -> tuple[bool, list[str]]:
    """Confirm all four registered channels exist for one timepoint."""
    missing: list[str] = []
    for ch in ("T1", "CT1", "T2", "FLAIR"):
        path = _channel_path(derivatives_root, tp_name, ch)
        if not path.is_file():
            missing.append(str(path))
    return (not missing, missing)


def _write_outputs(
    *,
    tp_name: str,
    result: SegmentationResult,
    derivatives_root: Path,
    voxel_volume_mm3: float,
    voxel_spacing_mm: tuple[float, float, float],
    elapsed_s: float,
    device: str,
) -> dict[str, Any]:
    """Write label map, probabilities, and sidecar JSON for one timepoint."""
    paths = _seg_paths(derivatives_root, tp_name)
    paths["label_map"].parent.mkdir(parents=True, exist_ok=True)

    nib.save(
        nib.Nifti1Image(result.label_map.astype(np.uint8), result.affine),
        str(paths["label_map"]),
    )
    probs_xyzt = np.moveaxis(result.probabilities, 0, -1).astype(np.float32)
    nib.save(nib.Nifti1Image(probs_xyzt, result.affine), str(paths["probabilities"]))

    volumes = _volumes_ml_from_label_map(result.label_map, voxel_volume_mm3)
    sidecar = {
        "schema_version": SEG_SIDECAR_SCHEMA_VERSION,
        "timepoint": tp_name,
        "backend": result.backend,
        "bundle_version": result.bundle_version,
        "device": device,
        "elapsed_s": round(float(elapsed_s), 3),
        "voxel_volume_mm3": round(voxel_volume_mm3, 6),
        "voxel_spacing_mm": [round(s, 6) for s in voxel_spacing_mm],
        "channel_order": list(CHANNEL_ORDER),
        "probability_channel_order": list(PROB_CHANNEL_ORDER),
        "label_scheme": {
            "background": 0,
            "necrosis_or_non_enhancing": LABEL_NCR,
            "edema": LABEL_ED,
            "enhancing_tumor": LABEL_ET,
        },
        "volumes_ml": volumes,
        "outputs": {
            "label_map": str(paths["label_map"]),
            "probabilities": str(paths["probabilities"]),
        },
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2))
    return sidecar


def _segment_one_timepoint(
    *,
    tp_name: str,
    derivatives_root: Path,
    backend: str,
    device: str,
    bundle_dir: Path | None,
) -> dict[str, Any]:
    """Reorient registered channels to LPS, run the bundle, reorient back to RAS."""
    src_paths = {
        "T1": _channel_path(derivatives_root, tp_name, "T1"),
        "CT1": _channel_path(derivatives_root, tp_name, "CT1"),
        "T2": _channel_path(derivatives_root, tp_name, "T2"),
        "FLAIR": _channel_path(derivatives_root, tp_name, "FLAIR"),
    }
    # Reference image: the registered T1 in its on-disk (RAS) frame; we
    # reorient outputs back into this image's frame for downstream consistency.
    reference_ras = nib.load(str(src_paths["T1"]))

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"hpgs_seg_{tp_name}_") as tmp:
        tmp_dir = Path(tmp)
        bundle_paths: dict[str, Path] = {}
        for name, src in src_paths.items():
            dst = tmp_dir / f"{name}_lps.nii.gz"
            _reorient_file(src, dst, BUNDLE_ORIENTATION)
            bundle_paths[name] = dst

        result = predict_brats3(
            t1=bundle_paths["T1"],
            t1c=bundle_paths["CT1"],  # LUMIERE 'CT1' == BraTS-bundle 'T1c'
            t2=bundle_paths["T2"],
            flair=bundle_paths["FLAIR"],
            backend=backend,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
            bundle_dir=bundle_dir,
        )
    elapsed_s = time.time() - t0

    # Reorient label map (3-D) and probabilities (4-D, channel-first) back into
    # the registered channels' RAS frame so the seg_dl/ outputs sit on the same
    # voxel grid as the *_to_ref.nii.gz files next to them.
    label_ras, label_affine = _reorient_array_to_reference(
        result.label_map.astype(np.uint8), result.affine, reference_ras
    )
    probs_xyzc = np.moveaxis(result.probabilities, 0, -1).astype(np.float32)
    probs_ras, probs_affine = _reorient_array_to_reference(probs_xyzc, result.affine, reference_ras)
    np.testing.assert_allclose(label_affine, probs_affine, atol=1e-6)

    result_ras = SegmentationResult(
        label_map=label_ras.astype(np.uint8),
        probabilities=np.moveaxis(probs_ras, -1, 0).astype(np.float32),
        affine=label_affine,
        backend=result.backend,
        channel_paths={k: Path(v) for k, v in src_paths.items()},
        bundle_version=result.bundle_version,
    )

    voxel_vol = _voxel_volume_mm3(result_ras.affine)
    voxel_spacing = _voxel_spacing_mm(result_ras.affine)

    sidecar = _write_outputs(
        tp_name=tp_name,
        result=result_ras,
        derivatives_root=derivatives_root,
        voxel_volume_mm3=voxel_vol,
        voxel_spacing_mm=voxel_spacing,
        elapsed_s=elapsed_s,
        device=device,
    )
    sidecar["bundle_orientation"] = list(BUNDLE_ORIENTATION)
    sidecar["output_orientation"] = list(nib.aff2axcodes(reference_ras.affine))
    # Refresh the on-disk JSON now that we've added the orientation fields.
    paths = _seg_paths(derivatives_root, tp_name)
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2))
    return sidecar


def _read_existing_sidecar(derivatives_root: Path, tp_name: str) -> dict[str, Any] | None:
    paths = _seg_paths(derivatives_root, tp_name)
    if not (paths["sidecar"].is_file() and paths["label_map"].is_file()):
        return None
    return json.loads(paths["sidecar"].read_text())


# ---------------------------------------------------------------------------
# Cohort-level aggregation (volumes.csv consumed by the figure builders).
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "timepoint",
    "days_since_baseline",
    "rano",
    "voxel_volume_mm3",
    "vol_WT_ml",
    "vol_TC_ml",
    "vol_ET_ml",
    "vol_NCR_ml",
    "vol_ED_ml",
)


def _write_summary(
    *,
    manifest: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    derivatives_root: Path,
    backend: str,
    device: str,
) -> tuple[Path, Path]:
    """Write segmentation_summary.{csv,json} at the derivatives root."""
    csv_path = derivatives_root / "segmentation_summary.csv"
    json_path = derivatives_root / "segmentation_summary.json"

    rows: list[dict[str, Any]] = []
    for tp_meta in manifest["timepoints"]:
        tp = tp_meta["name"]
        sc = sidecars.get(tp)
        if sc is None:
            continue
        vols = sc["volumes_ml"]
        rows.append(
            {
                "timepoint": tp,
                "days_since_baseline": tp_meta.get("days_since_baseline"),
                "rano": tp_meta.get("rano"),
                "voxel_volume_mm3": sc["voxel_volume_mm3"],
                **{k: vols[k] for k in CSV_COLUMNS if k in vols},
            }
        )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "cohort_id": manifest.get("cohort_id"),
        "patient_id": manifest.get("patient_id"),
        "manifest_path": str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        "derivatives_root": str(derivatives_root),
        "backend": backend,
        "device": device,
        "channel_order": list(CHANNEL_ORDER),
        "probability_channel_order": list(PROB_CHANNEL_ORDER),
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Cohort manifest YAML. Default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Root containing the registered channels and where seg_dl/ subdirs "
            f"will be written. Default: {DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--timepoint",
        action="append",
        dest="timepoints",
        default=None,
        metavar="NAME",
        help="Only process this timepoint name (e.g. 'week-013'); repeat for several.",
    )
    parser.add_argument(
        "--backend",
        choices=("monai_bundle", "dummy"),
        default="monai_bundle",
        help="Segmentation backend (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help=(
            "Inference device (default: %(default)s). The bundle SegResNet "
            "v0.4.8 is broken on Apple MPS -- see the module docstring -- so "
            "we default to cpu; pass --device cuda on a GPU box."
        ),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional MONAI bundle cache (default: ~/.cache/hpgs/monai_bundles).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip timepoints whose seg_dl outputs already exist (still re-aggregates the CSV).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run inference and overwrite existing outputs.",
    )
    return parser.parse_args(argv)


def _select_timepoints(
    manifest: dict[str, Any], requested: list[str] | None
) -> list[dict[str, Any]]:
    all_tps = manifest["timepoints"]
    if not requested:
        return all_tps
    by_name = {tp["name"]: tp for tp in all_tps}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(
            f"error: --timepoint values not in manifest: {missing}; available={list(by_name)}"
        )
    return [by_name[name] for name in requested]


def _print_table(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 88)
    print("LUMIERE Patient-048 unified DL segmentation report")
    print("=" * 88)
    print(
        f"{'timepoint':<14}{'status':<10}{'time(s)':>8}"
        f"{'WT(mL)':>10}{'TC(mL)':>10}{'ET(mL)':>10}{'NCR(mL)':>10}{'ED(mL)':>10}"
    )
    print("-" * 88)
    for r in records:
        if r.get("error"):
            print(
                f"{r['timepoint']:<14}{'FAILED':<10}{'-':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>10}"
            )
            print(f"        {r['error']}")
            continue
        status = r.get("status", "ok")
        sc = r["sidecar"]
        v = sc["volumes_ml"]
        elapsed = sc.get("elapsed_s", 0.0)
        print(
            f"{r['timepoint']:<14}{status:<10}{elapsed:>8.1f}"
            f"{v['vol_WT_ml']:>10.2f}{v['vol_TC_ml']:>10.2f}{v['vol_ET_ml']:>10.2f}"
            f"{v['vol_NCR_ml']:>10.2f}{v['vol_ED_ml']:>10.2f}"
        )
    print("=" * 88)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    derivatives_root = args.derivatives_root.expanduser().resolve()

    if not manifest_path.is_file():
        print(f"[err] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not derivatives_root.is_dir():
        print(
            f"[err] derivatives-root not found: {derivatives_root}\n"
            "  Run scripts/register_lumiere_p048.py first.",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(manifest_path)
    timepoints = _select_timepoints(manifest, args.timepoints)

    print(
        f"LUMIERE Patient-048 segmentation  manifest={manifest_path}\n"
        f"                                  derivatives_root={derivatives_root}\n"
        f"                                  backend={args.backend}  device={args.device}  "
        f"n_timepoints={len(timepoints)}  skip_existing={args.skip_existing}  "
        f"overwrite={args.overwrite}"
    )
    print("-" * 88)

    sidecars: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    n_failed = 0

    for tp_meta in timepoints:
        tp = tp_meta["name"]
        ok, missing = _check_inputs(derivatives_root, tp)
        if not ok:
            err = f"missing registered channel(s): {missing}"
            print(f"[run] {tp}: FAILED ({err})", file=sys.stderr)
            records.append({"timepoint": tp, "error": err})
            n_failed += 1
            continue

        if args.skip_existing and not args.overwrite:
            existing = _read_existing_sidecar(derivatives_root, tp)
            if existing is not None:
                print(f"[run] {tp}: SKIPPED (existing seg_dl outputs)")
                sidecars[tp] = existing
                records.append({"timepoint": tp, "sidecar": existing, "status": "skipped"})
                continue

        print(f"[run] {tp}: segmenting ...", flush=True)
        try:
            sidecar = _segment_one_timepoint(
                tp_name=tp,
                derivatives_root=derivatives_root,
                backend=args.backend,
                device=args.device,
                bundle_dir=args.bundle_dir,
            )
        except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[run] {tp}: FAILED: {exc}", file=sys.stderr)
            records.append({"timepoint": tp, "error": str(exc)})
            n_failed += 1
            continue
        sidecars[tp] = sidecar
        records.append({"timepoint": tp, "sidecar": sidecar, "status": "ok"})

    csv_path, _json_path = _write_summary(
        manifest=manifest,
        sidecars=sidecars,
        derivatives_root=derivatives_root,
        backend=args.backend,
        device=args.device,
    )
    _print_table(records)
    csv_label = csv_path.relative_to(REPO_ROOT) if csv_path.is_relative_to(REPO_ROOT) else csv_path
    print(f"[ok] cohort summary: {csv_label}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
