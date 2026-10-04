"""Aggregate segmentation QC for the LUMIERE Patient-048 derivatives tree.

The analogue of ``scripts/summarize_lumiere_p048_registration.py`` for the
unified MONAI Bundle BraTS three-class SegResNet outputs written by
``scripts/segment_lumiere_p048.py``. Reads every per-timepoint
``seg_dl/tp-<name>_seg_brats3_dl.{json,nii.gz}`` plus the registration-
warped legacy comparator masks (``tp-<name>_hdglio_to_ref_segmentation.nii.gz``
and ``tp-<name>_deepbratumia_to_ref_segmentation.nii.gz`` from the
registration step) and computes three views that together answer the
question "is the unified-segmenter run plausible enough to be the headline
engine in the round-3 longitudinal panel?":

1. **Per-tp segmentation QC table** -- timepoint, RANO label,
   voxel volume (mm^3), output orientation, runtime, and the five
   compartment volumes in mL straight from each per-tp sidecar.
2. **Volume-over-time table** -- the same compartment volumes, laid out
   so reviewers can eyeball the resection drop and the recurrence surge.
3. **Comparator agreement** -- per-(timepoint, compartment) Dice + Jaccard
   for DL vs HD-GLIO (LUMIERE-provided 2-class: WT and ET only) and
   DL vs DeepBraTumIA (3-class: WT, TC, ET). Skipped silently when the
   comparator NIfTI is not on disk; comparator label-scheme mapping is
   documented inline.

Output contract
---------------

Stdout: human-readable per-tp + agreement tables.

JSON sidecar (when ``--json-out`` is given): full structured summary,
shaped::

    {
      "schema_version": "1.0",
      "cohort_id": "lumiere_p048",
      "derivatives_root": "...",
      "reference_timepoint": "week-000-1",
      "per_timepoint": [
        {"timepoint": "week-000-1", "rano": "Pre-Op",
         "days_since_baseline": 0, "voxel_volume_mm3": 1.0,
         "elapsed_s": 30.788, "output_orientation": ["R", "A", "S"],
         "vol_WT_ml": 140.081, "vol_TC_ml": 34.12, "vol_ET_ml": 23.953,
         "vol_NCR_ml": 10.167, "vol_ED_ml": 105.961},
        ...
      ],
      "comparator_agreement": {
        "hdglio": {"week-000-1": {"WT": {"dice": ..., "jaccard": ...,
                                         "vol_dl_ml": ..., "vol_comp_ml": ...},
                                  "ET": {...}}},
        "deepbratumia": {"week-000-1": {"WT": {...}, "TC": {...}, "ET": {...}}}
      },
      "comparator_label_schemes": {
        "hdglio": {"WT": [1, 2], "ET": [2]},
        "deepbratumia": {"WT": [1, 2, 3], "TC": [1, 2], "ET": [1]}
      }
    }

Markdown sidecar (when ``--md-out`` is given): a thin human-readable
mirror of the JSON, suitable to paste into the revision tracker.

Usage
-----

    uv run python scripts/summarize_lumiere_p048_segmentation.py

    uv run python scripts/summarize_lumiere_p048_segmentation.py \\
        --derivatives-root ./data/lumiere_p048/derivatives \\
        --json-out ./outputs/inventory/lumiere_p048_segmentation_summary.json \\
        --md-out   ./outputs/inventory/lumiere_p048_segmentation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DERIVATIVES = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"
DEFAULT_SUMMARY_CSV = DEFAULT_DERIVATIVES / "segmentation_summary.csv"
SCHEMA_VERSION = "1.0"

# DL (BraTS 3-class) label scheme used by ``hpgs.segment.unified``.
DL_LABEL_NCR = 1
DL_LABEL_ED = 2
DL_LABEL_ET = 4

# Comparator label schemes (validated empirically against the staged
# Patient-048 derivatives, see commit message for the inspection log).
COMPARATOR_LABELS: dict[str, dict[str, tuple[int, ...]]] = {
    # HD-GLIO: 2-class (1 = T2-hyperintense / non-enhancing core+edema,
    # 2 = contrast-enhancing tumour). WT = both; ET = label 2 only.
    # HD-GLIO does not separately resolve TC vs ET, so we only score WT
    # and ET against HD-GLIO.
    "hdglio": {"WT": (1, 2), "ET": (2,)},
    # DeepBraTumIA (native CT1-grid mask, ``ct1_seg_mask.nii.gz``): 3-class
    # (1 = enhancing tumour, 2 = necrosis, 3 = edema). The label semantics
    # were validated empirically against the round-3 staged Patient-048
    # tree: at the pre-op timepoint the per-label volumes are
    # 1: 26.86 mL (DL ET 23.95 mL, HD-GLIO ET 25.90 mL), 2: 6.97 mL
    # (DL NCR 10.17 mL), 3: 97.21 mL (DL ED 105.96 mL); see commit message.
    # Maps to the BraTS three-compartment scheme:
    #     WT = all tumour labels  (1 + 2 + 3)
    #     TC = enhancing + necrosis  (1 + 2)
    #     ET = enhancing  (1)
    "deepbratumia": {"WT": (1, 2, 3), "TC": (1, 2), "ET": (1,)},
}
COMPARATOR_DISPLAY_ORDER: tuple[str, ...] = ("hdglio", "deepbratumia")
COMPARTMENTS_ORDER: tuple[str, ...] = ("WT", "TC", "ET")


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def _load_run_manifest(derivatives_root: Path) -> dict[str, Any]:
    path = derivatives_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No manifest.json under {derivatives_root}; "
            "did you run scripts/register_lumiere_p048.py?"
        )
    return json.loads(path.read_text())


def _load_summary_csv(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Map timepoint -> CSV row (string fields preserved). Tolerates absence."""
    if not csv_path.is_file():
        return {}
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {r["timepoint"]: r for r in rows}


def _load_seg_sidecar(derivatives_root: Path, tp: str) -> dict[str, Any] | None:
    p = derivatives_root / f"tp-{tp}" / "seg_dl" / f"tp-{tp}_seg_brats3_dl.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _load_label_map(path: Path) -> tuple[np.ndarray, float] | None:
    if not path.is_file():
        return None
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    voxel_volume_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    return data, voxel_volume_mm3


# ---------------------------------------------------------------------------
# Per-tp QC table.
# ---------------------------------------------------------------------------


def per_tp_segmentation_table(
    *,
    run_manifest: dict[str, Any],
    derivatives_root: Path,
    csv_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tp_meta in run_manifest["timepoints"]:
        tp = tp_meta["name"]
        sidecar = _load_seg_sidecar(derivatives_root, tp)
        csv_row = csv_rows.get(tp, {})
        row: dict[str, Any] = {
            "timepoint": tp,
            "rano": csv_row.get("rano") or "-",
            "days_since_baseline": (
                float(csv_row["days_since_baseline"])
                if csv_row.get("days_since_baseline")
                else None
            ),
        }
        if sidecar is None:
            row.update(
                {
                    "status": "missing",
                    "voxel_volume_mm3": None,
                    "elapsed_s": None,
                    "output_orientation": None,
                    "vol_WT_ml": None,
                    "vol_TC_ml": None,
                    "vol_ET_ml": None,
                    "vol_NCR_ml": None,
                    "vol_ED_ml": None,
                }
            )
        else:
            vols = sidecar.get("volumes_ml", {})
            row.update(
                {
                    "status": "ok",
                    "voxel_volume_mm3": sidecar.get("voxel_volume_mm3"),
                    "elapsed_s": sidecar.get("elapsed_s"),
                    "output_orientation": sidecar.get("output_orientation"),
                    "device": sidecar.get("device"),
                    "backend": sidecar.get("backend"),
                    "bundle_version": sidecar.get("bundle_version"),
                    "vol_WT_ml": vols.get("vol_WT_ml"),
                    "vol_TC_ml": vols.get("vol_TC_ml"),
                    "vol_ET_ml": vols.get("vol_ET_ml"),
                    "vol_NCR_ml": vols.get("vol_NCR_ml"),
                    "vol_ED_ml": vols.get("vol_ED_ml"),
                }
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Comparator agreement (Dice + Jaccard per compartment).
# ---------------------------------------------------------------------------


def _dice(a: np.ndarray, b: np.ndarray) -> float | None:
    """Dice coefficient on boolean masks; ``None`` when both are empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    n_a = int(a.sum())
    n_b = int(b.sum())
    if n_a == 0 and n_b == 0:
        return None
    inter = int(np.logical_and(a, b).sum())
    return float(2.0 * inter) / float(n_a + n_b)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(bool)
    b = b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    inter = int(np.logical_and(a, b).sum())
    return float(inter) / float(union)


def _dl_compartment_mask(label_map: np.ndarray, comp: str) -> np.ndarray:
    if comp == "WT":
        return label_map > 0
    if comp == "TC":
        return np.isin(label_map, (DL_LABEL_NCR, DL_LABEL_ET))
    if comp == "ET":
        return label_map == DL_LABEL_ET
    raise ValueError(f"Unknown compartment {comp!r}")


def _comparator_compartment_mask(label_map: np.ndarray, kind: str, comp: str) -> np.ndarray | None:
    """Return the comparator's mask for the BraTS-3 compartment, or ``None``."""
    scheme = COMPARATOR_LABELS.get(kind, {})
    label_codes = scheme.get(comp)
    if label_codes is None:
        return None
    return np.isin(label_map, label_codes)


def comparator_agreement(
    *,
    run_manifest: dict[str, Any],
    derivatives_root: Path,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """For each comparator x timepoint x compartment: Dice/JSC + volumes."""
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for tp_meta in run_manifest["timepoints"]:
        tp = tp_meta["name"]
        dl_path = derivatives_root / f"tp-{tp}" / "seg_dl" / f"tp-{tp}_seg_brats3_dl.nii.gz"
        dl_loaded = _load_label_map(dl_path)
        if dl_loaded is None:
            continue
        dl_label, voxel_mm3 = dl_loaded
        ml_per_voxel = voxel_mm3 / 1000.0

        for kind in COMPARATOR_DISPLAY_ORDER:
            comp_path = derivatives_root / f"tp-{tp}" / f"tp-{tp}_{kind}_to_ref_segmentation.nii.gz"
            comp_loaded = _load_label_map(comp_path)
            if comp_loaded is None:
                continue
            comp_label, _ = comp_loaded
            if comp_label.shape != dl_label.shape:
                # Should never happen on a healthy registration tree, but
                # surface it rather than silently mis-score.
                out.setdefault(kind, {}).setdefault(tp, {})["__shape_mismatch__"] = {
                    "dl_shape": list(dl_label.shape),
                    "comparator_shape": list(comp_label.shape),
                }
                continue
            for comp_name in COMPARTMENTS_ORDER:
                comp_mask = _comparator_compartment_mask(comp_label, kind, comp_name)
                if comp_mask is None:
                    continue  # comparator does not resolve this compartment
                dl_mask = _dl_compartment_mask(dl_label, comp_name)
                vol_dl_ml = round(float(dl_mask.sum()) * ml_per_voxel, 4)
                vol_comp_ml = round(float(comp_mask.sum()) * ml_per_voxel, 4)
                out.setdefault(kind, {}).setdefault(tp, {})[comp_name] = {
                    "dice": _dice(dl_mask, comp_mask),
                    "jaccard": _jaccard(dl_mask, comp_mask),
                    "vol_dl_ml": vol_dl_ml,
                    "vol_comp_ml": vol_comp_ml,
                    "vol_diff_ml": round(vol_dl_ml - vol_comp_ml, 4),
                }
    return out


# ---------------------------------------------------------------------------
# Stdout printers.
# ---------------------------------------------------------------------------


def _fmt_or_dash(v: Any, fmt: str) -> str:
    if v is None:
        return "-"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _print_per_tp(rows: list[dict[str, Any]]) -> None:
    print("=== Per-timepoint segmentation QC ===")
    print(
        f"{'timepoint':<14}  {'RANO':<8}  {'days':>5}  {'vox':>5}  "
        f"{'WT(mL)':>8}  {'TC(mL)':>8}  {'ET(mL)':>8}  "
        f"{'NCR(mL)':>8}  {'ED(mL)':>8}  {'t(s)':>6}  status"
    )
    print("-" * 105)
    for r in rows:
        ax = "/".join(r["output_orientation"]) if r.get("output_orientation") else "-"
        print(
            f"{r['timepoint']:<14}  {r['rano']:<8}  "
            f"{_fmt_or_dash(r['days_since_baseline'], '>5.0f')}  "
            f"{_fmt_or_dash(r['voxel_volume_mm3'], '>5.2f')}  "
            f"{_fmt_or_dash(r['vol_WT_ml'], '>8.2f')}  "
            f"{_fmt_or_dash(r['vol_TC_ml'], '>8.2f')}  "
            f"{_fmt_or_dash(r['vol_ET_ml'], '>8.2f')}  "
            f"{_fmt_or_dash(r['vol_NCR_ml'], '>8.2f')}  "
            f"{_fmt_or_dash(r['vol_ED_ml'], '>8.2f')}  "
            f"{_fmt_or_dash(r.get('elapsed_s'), '>6.1f')}  "
            f"{r.get('status', '?')}  [{ax}]"
        )
    print()


def _print_agreement(
    agreement: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    tp_order: list[str],
) -> None:
    if not agreement:
        print("=== Comparator agreement: no comparator masks on disk ===\n")
        return
    print("=== Comparator agreement (DL vs comparator, Dice / Jaccard, mL DL / mL comp) ===")
    print("  HD-GLIO (2-class) -> WT, ET only;  DeepBraTumIA (3-class) -> WT, TC, ET")
    for kind in COMPARATOR_DISPLAY_ORDER:
        per_tp = agreement.get(kind)
        if not per_tp:
            continue
        print(f"-- DL vs {kind} --")
        comps_in_scheme = [c for c in COMPARTMENTS_ORDER if c in COMPARATOR_LABELS[kind]]
        header_cells = "  ".join(f"{c}(Dice/JSC)".center(20) for c in comps_in_scheme)
        print(f"{'timepoint':<14}  {header_cells}")
        print("-" * (16 + 22 * len(comps_in_scheme)))
        for tp in tp_order:
            cells_for_tp = per_tp.get(tp, {})
            if "__shape_mismatch__" in cells_for_tp:
                print(
                    f"{tp:<14}  shape mismatch dl={cells_for_tp['__shape_mismatch__']['dl_shape']} "
                    f"comp={cells_for_tp['__shape_mismatch__']['comparator_shape']}"
                )
                continue
            parts: list[str] = []
            for comp in comps_in_scheme:
                cell = cells_for_tp.get(comp)
                if cell is None:
                    parts.append("-".center(20))
                    continue
                d = cell["dice"]
                j = cell["jaccard"]
                d_s = "n/a" if d is None else f"{d:.3f}"
                j_s = "n/a" if j is None else f"{j:.3f}"
                parts.append(f"{d_s}/{j_s}".center(20))
            print(f"{tp:<14}  " + "  ".join(parts))
        print()


# ---------------------------------------------------------------------------
# Markdown sidecar.
# ---------------------------------------------------------------------------


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# LUMIERE Patient-048 -- segmentation summary", ""]
    lines.append(f"- Generated (UTC): `{payload['ran_at_utc']}`")
    lines.append(f"- Cohort id: `{payload.get('cohort_id')}`")
    lines.append(f"- Derivatives root: `{payload['derivatives_root']}`")
    lines.append(f"- Reference timepoint: `{payload['reference_timepoint']}`")
    lines.append("")

    lines.append("## Per-timepoint segmentation QC")
    lines.append("")
    header_cols = (
        "timepoint",
        "RANO",
        "days",
        "vox (mm^3)",
        "WT (mL)",
        "TC (mL)",
        "ET (mL)",
        "NCR (mL)",
        "ED (mL)",
        "t (s)",
        "orientation",
        "device",
        "status",
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    for r in payload["per_timepoint"]:
        ax = "/".join(r["output_orientation"]) if r.get("output_orientation") else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    r["timepoint"],
                    r["rano"],
                    _fmt_or_dash(r["days_since_baseline"], ".0f"),
                    _fmt_or_dash(r["voxel_volume_mm3"], ".2f"),
                    _fmt_or_dash(r["vol_WT_ml"], ".2f"),
                    _fmt_or_dash(r["vol_TC_ml"], ".2f"),
                    _fmt_or_dash(r["vol_ET_ml"], ".2f"),
                    _fmt_or_dash(r["vol_NCR_ml"], ".2f"),
                    _fmt_or_dash(r["vol_ED_ml"], ".2f"),
                    _fmt_or_dash(r.get("elapsed_s"), ".1f"),
                    ax,
                    str(r.get("device", "-")),
                    str(r.get("status", "?")),
                ]
            )
            + " |"
        )
    lines.append("")

    agreement = payload.get("comparator_agreement") or {}
    if agreement:
        lines.append("## Comparator agreement (DL vs comparator)")
        lines.append("")
        lines.append(
            "Dice / Jaccard on boolean compartment masks at the registered "
            "Patient-048 reference grid. HD-GLIO is 2-class so only WT and "
            "ET are scorable; DeepBraTumIA is 3-class so WT, TC, and ET are "
            "all scorable. ``vol_diff_ml = vol_dl - vol_comp``."
        )
        lines.append("")
        for kind in COMPARATOR_DISPLAY_ORDER:
            per_tp = agreement.get(kind)
            if not per_tp:
                continue
            comps = [c for c in COMPARTMENTS_ORDER if c in COMPARATOR_LABELS[kind]]
            lines.append(f"### DL vs {kind}")
            lines.append("")
            head = (
                "| timepoint | "
                + " | ".join(
                    f"{c} Dice | {c} JSC | {c} dl mL | {c} comp mL | {c} diff mL" for c in comps
                )
                + " |"
            )
            sep = "|-----------|" + "|".join(["-----:" for _ in range(5 * len(comps))]) + "|"
            lines.append(head)
            lines.append(sep)
            for tp_row in payload["per_timepoint"]:
                tp = tp_row["timepoint"]
                cells = per_tp.get(tp, {})
                row_cells: list[str] = [tp]
                for comp in comps:
                    cell = cells.get(comp, {})
                    d = cell.get("dice")
                    j = cell.get("jaccard")
                    row_cells.extend(
                        [
                            "n/a" if d is None else f"{d:.3f}",
                            "n/a" if j is None else f"{j:.3f}",
                            _fmt_or_dash(cell.get("vol_dl_ml"), ".2f"),
                            _fmt_or_dash(cell.get("vol_comp_ml"), ".2f"),
                            _fmt_or_dash(cell.get("vol_diff_ml"), "+.2f"),
                        ]
                    )
                lines.append("| " + " | ".join(row_cells) + " |")
            lines.append("")
    else:
        lines.append("## Comparator agreement")
        lines.append("")
        lines.append("_No comparator masks on disk; skipped._")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def summarize(
    *,
    derivatives_root: Path,
    summary_csv: Path,
) -> dict[str, Any]:
    run_manifest = _load_run_manifest(derivatives_root)
    csv_rows = _load_summary_csv(summary_csv)
    rows = per_tp_segmentation_table(
        run_manifest=run_manifest,
        derivatives_root=derivatives_root,
        csv_rows=csv_rows,
    )
    agreement = comparator_agreement(
        run_manifest=run_manifest,
        derivatives_root=derivatives_root,
    )

    tp_order = [tp_meta["name"] for tp_meta in run_manifest["timepoints"]]
    _print_per_tp(rows)
    _print_agreement(agreement, tp_order=tp_order)

    return {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": run_manifest.get("cohort_id"),
        "derivatives_root": str(derivatives_root),
        "summary_csv": str(summary_csv),
        "reference_timepoint": run_manifest["reference_timepoint"],
        "channels": run_manifest["channels"],
        "per_timepoint": rows,
        "comparator_agreement": agreement,
        "comparator_label_schemes": {
            k: {comp: list(codes) for comp, codes in v.items()}
            for k, v in COMPARATOR_LABELS.items()
        },
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Where the segmentation outputs live. Default: "
            f"{DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=(
            "Per-tp segmentation_summary.csv produced by "
            "scripts/segment_lumiere_p048.py. Default: "
            f"{DEFAULT_SUMMARY_CSV.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON sidecar to write the full summary into.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Optional Markdown sidecar mirroring the JSON for human readers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    derivs = args.derivatives_root.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()
    if not derivs.is_dir():
        print(f"[err] derivatives-root not found: {derivs}", file=sys.stderr)
        return 2
    payload = summarize(derivatives_root=derivs, summary_csv=summary_csv)
    if args.json_out is not None:
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[ok] wrote JSON summary -> {out}")
    if args.md_out is not None:
        out_md = args.md_out.expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(payload))
        print(f"[ok] wrote Markdown summary -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
