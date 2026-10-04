"""Inventory the LUMIERE Patient-048 longitudinal mpMRI tree (Phase 1, read-only).

This script walks a local LUMIERE Patient-048 folder and reports what is
actually on disk, so the Phase 2 producer code (longitudinal volume
trajectories + Hit-Rate radial plots with the unified MONAI Bundle BraTS
SegResNet) can be planned against verified ground truth instead of
guesses.

The script is **read-only**: it never writes into the LUMIERE tree. It
prints a human-readable summary to stdout and, with ``--out-json``,
emits a machine-readable inventory JSON that Phase 2 can consume.

Expected directory layout (per the public LUMIERE archive
``ysuter/gbm-data-longitudinal``):

    Patient-048/
        week-000-1/                     # one timepoint
            T1.nii.gz
            CT1.nii.gz
            T2.nii.gz
            FLAIR.nii.gz
            HD-GLIO-AUTO-segmentation/
                native/                 # per-channel-space segmentations
                registered/             # registered + brain-extracted +
                                        # one common-grid segmentation
            DeepBraTumIA-segmentation/
                native/
                atlas/
        week-000-2/  ...
        week-013/    ...
        ...

What it reports
---------------

For every timepoint:

* presence of the four channel NIfTIs (T1, CT1, T2, FLAIR);
* presence and contents of HD-GLIO-AUTO-segmentation and
  DeepBraTumIA-segmentation subtrees;
* per-NIfTI shape / voxel size / orientation / dtype / size on disk
  (read with nibabel, header only -- no voxel data is loaded).

Across timepoints:

* whether the HD-GLIO-AUTO ``registered/segmentation.nii.gz`` grid is
  consistent across all timepoints (cheap proxy for "is the dataset
  pre-registered into a common subject space?");
* whether the per-timepoint registered segmentation grid matches the
  per-timepoint channel grids (cheap proxy for "do we need to resample
  before running the unified engine?").

Usage
-----

    uv run python scripts/inspect_lumiere_p048.py
    uv run python scripts/inspect_lumiere_p048.py \
        --root ~/Dropbox/Arvid/China_2025/LUMIERE/Patient-048
    uv run python scripts/inspect_lumiere_p048.py \
        --root /path/to/lumiere/Patient-048 \
        --out-json outputs/inventory/lumiere_p048_inventory.json

The default ``--root`` is the author's local mirror under
``~/Dropbox/Arvid/China_2025/LUMIERE/Patient-048``. On any other
machine, pass an explicit ``--root`` pointing at the public LUMIERE
``Patient-048`` folder (downloaded from
``https://github.com/ysuter/gbm-data-longitudinal``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib

DEFAULT_ROOT = Path("~/Dropbox/Arvid/China_2025/LUMIERE/Patient-048").expanduser()

CHANNEL_NAMES: tuple[str, ...] = ("T1", "CT1", "T2", "FLAIR")
SEG_SUBDIRS: tuple[str, ...] = ("HD-GLIO-AUTO-segmentation", "DeepBraTumIA-segmentation")


@dataclass(frozen=True)
class NiftiHeader:
    """Header-only summary of a NIfTI file. Voxel data is never loaded."""

    path: str
    size_bytes: int
    shape: tuple[int, ...]
    voxel_size_mm: tuple[float, ...]
    orientation: str
    dtype: str

    @classmethod
    def from_path(cls, root: Path, path: Path) -> NiftiHeader:
        img = nib.load(str(path))
        zooms = tuple(float(z) for z in img.header.get_zooms())
        try:
            ornt = nib.aff2axcodes(img.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "?"
        return cls(
            path=str(path.relative_to(root)),
            size_bytes=path.stat().st_size,
            shape=tuple(int(s) for s in img.shape),
            voxel_size_mm=zooms,
            orientation=orientation,
            dtype=str(img.get_data_dtype()),
        )


@dataclass
class TimepointInventory:
    """Inventory of a single timepoint folder."""

    name: str
    channels: dict[str, NiftiHeader | None] = field(default_factory=dict)
    hdglioauto_native: list[NiftiHeader] = field(default_factory=list)
    hdglioauto_registered: list[NiftiHeader] = field(default_factory=list)
    hdglioauto_registered_segmentation: NiftiHeader | None = None
    deepbratumia_native: list[NiftiHeader] = field(default_factory=list)
    deepbratumia_atlas: list[NiftiHeader] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)


def _probe_dir(root: Path, dir_path: Path) -> list[NiftiHeader]:
    """Return NIfTI headers for every .nii / .nii.gz under ``dir_path`` (recursive)."""
    if not dir_path.is_dir():
        return []
    out: list[NiftiHeader] = []
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            try:
                out.append(NiftiHeader.from_path(root, path))
            except Exception as exc:  # pragma: no cover - read errors are diagnostics
                print(
                    f"[warn] could not read {path.relative_to(root)}: {exc}",
                    file=sys.stderr,
                )
    return out


def inventory_timepoint(root: Path, tp_dir: Path) -> TimepointInventory:
    inv = TimepointInventory(name=tp_dir.name)

    for channel in CHANNEL_NAMES:
        candidate = tp_dir / f"{channel}.nii.gz"
        inv.channels[channel] = (
            NiftiHeader.from_path(root, candidate) if candidate.is_file() else None
        )

    hdg = tp_dir / "HD-GLIO-AUTO-segmentation"
    inv.hdglioauto_native = _probe_dir(root, hdg / "native")
    inv.hdglioauto_registered = _probe_dir(root, hdg / "registered")
    reg_seg = hdg / "registered" / "segmentation.nii.gz"
    if reg_seg.is_file():
        inv.hdglioauto_registered_segmentation = NiftiHeader.from_path(root, reg_seg)

    dbt = tp_dir / "DeepBraTumIA-segmentation"
    inv.deepbratumia_native = _probe_dir(root, dbt / "native")
    inv.deepbratumia_atlas = _probe_dir(root, dbt / "atlas")

    expected = set(CHANNEL_NAMES) | set(SEG_SUBDIRS) | {f"{c}.nii.gz" for c in CHANNEL_NAMES}
    for path in sorted(tp_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.name in expected or path.name in SEG_SUBDIRS:
            continue
        inv.extra_files.append(path.name)

    return inv


def discover_timepoints(root: Path) -> list[Path]:
    """Return sorted list of timepoint subfolders under ``root``."""
    if not root.is_dir():
        raise FileNotFoundError(f"LUMIERE Patient-048 root not found: {root}")
    timepoints = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not timepoints:
        raise FileNotFoundError(f"No timepoint subfolders under {root}")
    return timepoints


def _grid_key(hdr: NiftiHeader | None) -> tuple | None:
    """Comparable grid signature: shape + voxel size rounded to 1e-3 mm."""
    if hdr is None:
        return None
    return hdr.shape + tuple(round(z, 3) for z in hdr.voxel_size_mm)


def _format_voxel(zooms: Iterable[float]) -> str:
    return " x ".join(f"{z:.3f}" for z in zooms)


def _format_size(n_bytes: int) -> str:
    units = (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10))
    for label, scale in units:
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {label}"
    return f"{n_bytes} B"


def _print_channels(inv: TimepointInventory) -> None:
    print("  channels:")
    for channel in CHANNEL_NAMES:
        hdr = inv.channels.get(channel)
        if hdr is None:
            print(f"    - {channel:<6}  MISSING")
            continue
        print(
            f"    - {channel:<6}  shape={list(hdr.shape)}  "
            f"voxel=[{_format_voxel(hdr.voxel_size_mm)}] mm  "
            f"orient={hdr.orientation}  dtype={hdr.dtype}  "
            f"({_format_size(hdr.size_bytes)})"
        )


def _print_segmentation_subtrees(inv: TimepointInventory) -> None:
    print("  HD-GLIO-AUTO-segmentation/:")
    print(f"    native:     {len(inv.hdglioauto_native)} file(s)")
    print(f"    registered: {len(inv.hdglioauto_registered)} file(s)")
    if inv.hdglioauto_registered_segmentation is not None:
        hdr = inv.hdglioauto_registered_segmentation
        print(
            f"      -> registered/segmentation.nii.gz  "
            f"shape={list(hdr.shape)}  voxel=[{_format_voxel(hdr.voxel_size_mm)}] mm  "
            f"orient={hdr.orientation}  dtype={hdr.dtype}"
        )
    else:
        print("      -> registered/segmentation.nii.gz  MISSING")

    print("  DeepBraTumIA-segmentation/:")
    print(f"    native: {len(inv.deepbratumia_native)} file(s)")
    print(f"    atlas:  {len(inv.deepbratumia_atlas)} file(s)")

    if inv.extra_files:
        print(f"  extra (unexpected) entries: {inv.extra_files}")


def _print_registered_grid_consistency(inventories: list[TimepointInventory]) -> None:
    reg_grids = {inv.name: _grid_key(inv.hdglioauto_registered_segmentation) for inv in inventories}
    unique = {g for g in reg_grids.values() if g is not None}
    if not unique:
        print("  HD-GLIO-AUTO registered segmentation: NONE PRESENT")
        return
    if len(unique) == 1:
        print(
            "  HD-GLIO-AUTO registered/segmentation.nii.gz grid: "
            "CONSISTENT across all timepoints (single common grid)"
        )
        return
    print(
        "  HD-GLIO-AUTO registered/segmentation.nii.gz grid: "
        f"VARIES across timepoints ({len(unique)} distinct grids)"
    )
    for tp_name, grid in reg_grids.items():
        print(f"    - {tp_name}: {grid}")


def _print_channel_grid_consistency(inventories: list[TimepointInventory]) -> None:
    print("\nPer-channel grid consistency across timepoints:")
    for channel in CHANNEL_NAMES:
        present = [
            inv.channels.get(channel)
            for inv in inventories
            if inv.channels.get(channel) is not None
        ]
        if not present:
            print(f"  {channel:<6}: NONE PRESENT")
            continue
        unique = {_grid_key(h) for h in present}
        if len(unique) == 1:
            ref = present[0]
            print(
                f"  {channel:<6}: CONSISTENT  shape={list(ref.shape)}  "
                f"voxel=[{_format_voxel(ref.voxel_size_mm)}] mm"
            )
        else:
            print(
                f"  {channel:<6}: VARIES across timepoints "
                f"({len(unique)} distinct grids; each timepoint has its own native grid)"
            )


def _print_readiness_summary(inventories: list[TimepointInventory]) -> None:
    n_complete = sum(
        all(inv.channels.get(c) is not None for c in CHANNEL_NAMES)
        and inv.hdglioauto_registered_segmentation is not None
        for inv in inventories
    )
    print()
    print(
        f"Reproducibility-readiness summary: "
        f"{n_complete} / {len(inventories)} timepoints have all 4 raw channels + "
        f"the common-grid HD-GLIO-AUTO segmentation."
    )


def print_summary(inventories: list[TimepointInventory]) -> None:
    print("=" * 78)
    print(f"LUMIERE Patient-048 inventory  ({len(inventories)} timepoints)")
    print("=" * 78)

    for inv in inventories:
        print(f"\n[{inv.name}]")
        _print_channels(inv)
        _print_segmentation_subtrees(inv)

    print("\n" + "-" * 78)
    print("Cross-timepoint consistency:")
    print("-" * 78)
    _print_registered_grid_consistency(inventories)
    _print_channel_grid_consistency(inventories)
    _print_readiness_summary(inventories)


def _to_json(inventories: list[TimepointInventory], root: Path) -> dict:
    return {
        "schema_version": "1.0",
        "root": str(root),
        "n_timepoints": len(inventories),
        "channels_expected": list(CHANNEL_NAMES),
        "timepoints": [
            {
                "name": inv.name,
                "channels": {
                    name: (asdict(hdr) if hdr is not None else None)
                    for name, hdr in inv.channels.items()
                },
                "hdglioauto": {
                    "native": [asdict(h) for h in inv.hdglioauto_native],
                    "registered": [asdict(h) for h in inv.hdglioauto_registered],
                    "registered_segmentation": (
                        asdict(inv.hdglioauto_registered_segmentation)
                        if inv.hdglioauto_registered_segmentation is not None
                        else None
                    ),
                },
                "deepbratumia": {
                    "native": [asdict(h) for h in inv.deepbratumia_native],
                    "atlas": [asdict(h) for h in inv.deepbratumia_atlas],
                },
                "extra_files": inv.extra_files,
            }
            for inv in inventories
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Path to the LUMIERE Patient-048 folder. Default: "
            f"{DEFAULT_ROOT} (author's local mirror)."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write a structured inventory JSON sidecar.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    timepoints = discover_timepoints(root)
    inventories = [inventory_timepoint(root, tp) for tp in timepoints]

    print_summary(inventories)

    if args.out_json is not None:
        out = args.out_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_json(inventories, root)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nInventory JSON written to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
