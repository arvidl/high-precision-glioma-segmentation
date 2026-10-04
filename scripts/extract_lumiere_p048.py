"""Stage the LUMIERE Patient-048 longitudinal mpMRI tree into ``data/lumiere_p048/``.

Mirrors the staging convention used for ``data/ucsf_pdgm_cohort50/``: the
LUMIERE archive's binaries are never committed; this script copies them
from a local LUMIERE clone into a clean BIDS-ish layout under ``data/``
(which is git-ignored), so every reproducer can stage the same Patient-048
cohort with one command.

Source layout (LUMIERE archive, Patient-048 sub-tree)::

    Patient-048/<week-NNN[-K]>/{T1,CT1,T2,FLAIR}.nii.gz
    Patient-048/<week-NNN[-K]>/HD-GLIO-AUTO-segmentation/...
    Patient-048/<week-NNN[-K]>/DeepBraTumIA-segmentation/
        atlas/segmentation/seg_mask.nii.gz       # SRI24 atlas-space (NOT used)
        native/segmentation/ct1_seg_mask.nii.gz  # CT1-grid native space (used)
        native/segmentation/{t1,t2,flair}_seg_mask.nii.gz   # other native grids
        native/transformation/*.tfm

Staged layout::

    data/lumiere_p048/
        manifest.json
        tp-week-000-1/
            tp-week-000-1_T1.nii.gz
            tp-week-000-1_CT1.nii.gz
            tp-week-000-1_T2.nii.gz
            tp-week-000-1_FLAIR.nii.gz
            # (only with --include-legacy-seg)
            tp-week-000-1_hdglio_registered_segmentation.nii.gz
            tp-week-000-1_deepbratumia_native_segmentation.nii.gz
        tp-week-000-2/...
        ...
        tp-week-049/...

DeepBraTumIA variant choice
---------------------------
The LUMIERE archive ships *two* DeepBraTumIA outputs per timepoint: an
``atlas/segmentation/seg_mask.nii.gz`` in SRI24 atlas space (LAS,
182x218x182 @ 1 mm) and four ``native/segmentation/<channel>_seg_mask.nii.gz``
files (one per channel, each on the corresponding raw channel's grid).
The round-2 staging used the **atlas** variant, but that mask is not in
the patient's CT1 space, so applying our rigid CT1->ref transform
warps it to the wrong anatomy and produces near-zero per-compartment
overlap with the unified-segmenter prediction (validated empirically;
see ``outputs/inventory/lumiere_p048_segmentation_summary.json`` from
the earlier round-2 run). The round-3 staging therefore uses the
**native CT1-grid** variant (``ct1_seg_mask.nii.gz``), which lives in
the same physical space as the staged ``CT1.nii.gz`` and warps cleanly
through the existing rigid transform stack onto the reference grid.

The set of timepoints to stage is read from
``configs/lumiere_p048_timepoints.yaml`` (no hard-coded list; mirrors how
``extract_ucsfpdgm.py`` keeps its cohort definition in one place).

Usage
-----

    uv run python scripts/extract_lumiere_p048.py \
        --src ~/Dropbox/Arvid/China_2025/LUMIERE/Patient-048

    uv run python scripts/extract_lumiere_p048.py \
        --src /path/to/lumiere/Patient-048 \
        --dst ./data/lumiere_p048 \
        --include-legacy-seg

The four MR channels (T1, CT1, T2, FLAIR) are always staged. The legacy
LUMIERE-team tumour segmentations (HD-GLIO-AUTO, DeepBraTumIA) are staged
only with ``--include-legacy-seg``; in the round-3 revision they are used
as a comparator overlay, while the primary tumour segmentation comes from
the unified MONAI Bundle BraTS SegResNet run on the staged channel stacks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_DST = REPO_ROOT / "data" / "lumiere_p048"

# (suffix in source, suffix in stage)
CHANNEL_FILES: tuple[tuple[str, str], ...] = (
    ("T1.nii.gz", "T1.nii.gz"),
    ("CT1.nii.gz", "CT1.nii.gz"),
    ("T2.nii.gz", "T2.nii.gz"),
    ("FLAIR.nii.gz", "FLAIR.nii.gz"),
)

# (relative path in source, suffix in stage)
LEGACY_SEG_FILES: tuple[tuple[str, str], ...] = (
    (
        "HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz",
        "hdglio_registered_segmentation.nii.gz",
    ),
    (
        # Native CT1-grid mask: same physical space as the staged ``CT1.nii.gz``
        # so the rigid CT1 -> ref transform warps it cleanly to the reference
        # grid. The ``atlas`` variant in SRI24 space is intentionally NOT
        # staged (see "DeepBraTumIA variant choice" in the module docstring).
        "DeepBraTumIA-segmentation/native/segmentation/ct1_seg_mask.nii.gz",
        "deepbratumia_native_segmentation.nii.gz",
    ),
)


def _load_manifest(path: Path) -> dict:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _resolve_first_match(parent: Path, candidates: list[str]) -> Path | None:
    """Return the first candidate path that exists, or None."""
    for relative in candidates:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    return None


def _copy_one(src_file: Path, dst_file: Path, *, overwrite: bool) -> str:
    """Copy ``src_file`` -> ``dst_file``. Returns one of {copied, skipped}."""
    if dst_file.exists() and not overwrite:
        return "skipped"
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    return "copied"


def stage_timepoint(
    *,
    tp_name: str,
    src_root: Path,
    dst_root: Path,
    include_legacy_seg: bool,
    overwrite: bool,
) -> dict:
    """Stage one timepoint. Returns a per-file provenance record."""
    src_dir = src_root / tp_name
    dst_dir = dst_root / f"tp-{tp_name}"

    record: dict = {"timepoint": tp_name, "src_dir": str(src_dir), "files": []}
    if not src_dir.is_dir():
        record["status"] = "missing_source_dir"
        return record

    for src_suffix, stage_suffix in CHANNEL_FILES:
        src_file = src_dir / src_suffix
        dst_file = dst_dir / f"tp-{tp_name}_{stage_suffix}"
        if not src_file.is_file():
            record["files"].append(
                {"role": "channel", "suffix": stage_suffix, "status": "missing_source"}
            )
            continue
        status = _copy_one(src_file, dst_file, overwrite=overwrite)
        record["files"].append(
            {
                "role": "channel",
                "suffix": stage_suffix,
                "src": str(src_file.relative_to(src_root)),
                "dst": str(dst_file.relative_to(dst_root)),
                "status": status,
            }
        )

    if include_legacy_seg:
        for src_rel, stage_suffix in LEGACY_SEG_FILES:
            src_file = src_dir / src_rel
            dst_file = dst_dir / f"tp-{tp_name}_{stage_suffix}"
            if not src_file.is_file():
                record["files"].append(
                    {"role": "legacy_seg", "suffix": stage_suffix, "status": "missing_source"}
                )
                continue
            status = _copy_one(src_file, dst_file, overwrite=overwrite)
            record["files"].append(
                {
                    "role": "legacy_seg",
                    "suffix": stage_suffix,
                    "src": str(src_file.relative_to(src_root)),
                    "dst": str(dst_file.relative_to(dst_root)),
                    "status": status,
                }
            )

    return record


def stage(
    *,
    manifest_path: Path,
    src_root: Path,
    dst_root: Path,
    include_legacy_seg: bool = False,
    overwrite: bool = False,
) -> dict:
    """Stage all timepoints listed in the manifest. Returns a provenance dict."""
    if not src_root.is_dir():
        raise FileNotFoundError(f"--src does not exist: {src_root}")

    manifest = _load_manifest(manifest_path)
    dst_root.mkdir(parents=True, exist_ok=True)

    records = [
        stage_timepoint(
            tp_name=tp["name"],
            src_root=src_root,
            dst_root=dst_root,
            include_legacy_seg=include_legacy_seg,
            overwrite=overwrite,
        )
        for tp in manifest["timepoints"]
    ]

    n_copied = sum(1 for rec in records for f in rec["files"] if f.get("status") == "copied")
    n_skipped = sum(1 for rec in records for f in rec["files"] if f.get("status") == "skipped")
    n_missing = sum(
        1 for rec in records for f in rec["files"] if f.get("status") == "missing_source"
    )

    provenance = {
        "schema_version": "1.0",
        "cohort_id": manifest.get("cohort_id", "lumiere_p048"),
        "patient_id": manifest.get("patient_id", "048"),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT))
        if manifest_path.is_relative_to(REPO_ROOT)
        else str(manifest_path),
        "source_root": str(src_root),
        "stage_root": str(dst_root),
        "include_legacy_seg": include_legacy_seg,
        "overwrite": overwrite,
        "staged_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {"copied": n_copied, "skipped": n_skipped, "missing_source": n_missing},
        "timepoints": records,
    }

    out_manifest = dst_root / "manifest.json"
    out_manifest.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    manifest_label = (
        out_manifest.relative_to(REPO_ROOT)
        if out_manifest.is_relative_to(REPO_ROOT)
        else out_manifest
    )
    print(
        f"[ok] {n_copied} file(s) copied, {n_skipped} skipped (already present), "
        f"{n_missing} missing in source. Manifest: {manifest_label}"
    )
    if n_missing:
        print(
            "[warn] some expected source files were not found; "
            "see manifest.json for the per-file breakdown.",
            file=sys.stderr,
        )
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the LUMIERE Patient-048 directory in your local LUMIERE clone.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination root for staged data. Default: {DEFAULT_DST.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Cohort manifest YAML listing the timepoints to stage. Default: "
            f"{DEFAULT_MANIFEST.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--include-legacy-seg",
        action="store_true",
        help=(
            "Also stage the LUMIERE-team segmentations (HD-GLIO-AUTO registered, "
            "DeepBraTumIA native CT1-grid) as comparator overlays."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-staged files. Default: skip files that exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage(
        manifest_path=args.manifest.expanduser().resolve(),
        src_root=args.src.expanduser().resolve(),
        dst_root=args.dst.expanduser().resolve(),
        include_legacy_seg=args.include_legacy_seg,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
