"""Aggregate registration QC for the LUMIERE Patient-048 derivatives tree.

Reads every ``registration_qc.json`` produced by
``scripts/register_lumiere_p048.py`` and computes three views that together
answer the question "is the longitudinal coregistration healthy enough to
hand off to segmentation?":

1. **Per-tp registration QC table** -- moving channel, moving-to-ref Pearson
   correlation, and a "all channels share ref grid" check, straight from
   each ``registration_qc.json``.
2. **Cross-tp same-channel correlations** -- for each channel, Pearson
   correlation of every tp's warped channel against the reference tp's same
   channel. Drops near 0 would indicate that LUMIERE did not pre-coregister
   that channel within-tp (which the rigid T1->T1 transform cannot fix).
3. **Legacy-segmentation sanity** (only when present) -- label-preservation
   under nearest-neighbour warp + fraction of warped non-zero voxels that
   lie inside the reference T1 brain mask, broken down by label.

Usage
-----

    uv run python scripts/summarize_lumiere_p048_registration.py

    uv run python scripts/summarize_lumiere_p048_registration.py \
        --derivatives-root ./data/lumiere_p048/derivatives \
        --json-out ./outputs/inventory/lumiere_p048_registration_summary.json

Side-effects: prints to stdout. Writes a JSON sidecar only with ``--json-out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DERIVATIVES = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"
LABEL_NAMES_DEEPBRATUMIA = {0: "background", 1: "enhT", 2: "necrosis", 3: "edema"}
LABEL_NAMES_HDGLIO = {0: "background", 1: "necrosis_or_edema", 2: "enhT_or_tumor_core"}
DEFAULT_BRAIN_THRESHOLD = 0.05  # T1 > 5% * max  -> brain mask


def _load_run_manifest(derivatives_root: Path) -> dict[str, Any]:
    path = derivatives_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No manifest.json under {derivatives_root}; "
            "did you run scripts/register_lumiere_p048.py?"
        )
    return json.loads(path.read_text())


def _load_qc(derivatives_root: Path, tp_name: str) -> dict[str, Any]:
    return json.loads((derivatives_root / f"tp-{tp_name}" / "registration_qc.json").read_text())


def _ravel(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).ravel()


def _array(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _voxel_volume_ml(path: Path) -> float:
    zooms = nib.load(str(path)).header.get_zooms()[:3]
    return float(np.prod(zooms)) / 1000.0


def per_tp_table(*, run_manifest: dict[str, Any], derivatives_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tp_meta in run_manifest["timepoints"]:
        tp = tp_meta["name"]
        qc = _load_qc(derivatives_root, tp)
        chans = qc["qc"]["channels"]
        all_match = all(info["grid_matches_reference"] for info in chans.values())
        rows.append(
            {
                "timepoint": tp,
                "is_reference": qc["is_reference"],
                "moving_channel": qc["moving_channel"] or "-",
                "moving_to_ref_correlation": qc["qc"].get("moving_to_ref_correlation"),
                "all_channels_match_ref_grid": all_match,
            }
        )
    return rows


def cross_tp_correlations(
    *, run_manifest: dict[str, Any], derivatives_root: Path
) -> dict[str, dict[str, float]]:
    channels = run_manifest["channels"]
    ref_tp = run_manifest["reference_timepoint"]
    ref_arrays = {
        ch: _ravel(derivatives_root / f"tp-{ref_tp}" / f"tp-{ref_tp}_{ch}_to_ref.nii.gz")
        for ch in channels
    }
    out: dict[str, dict[str, float]] = {}
    for tp_meta in run_manifest["timepoints"]:
        tp = tp_meta["name"]
        out[tp] = {}
        for ch in channels:
            arr = _ravel(derivatives_root / f"tp-{tp}" / f"tp-{tp}_{ch}_to_ref.nii.gz")
            ref = ref_arrays[ch]
            if ref.std() == 0 or arr.std() == 0:
                out[tp][ch] = float("nan")
            else:
                out[tp][ch] = float(np.corrcoef(ref, arr)[0, 1])
    return out


def legacy_seg_diagnostics(
    *,
    run_manifest: dict[str, Any],
    stage_root: Path,
    derivatives_root: Path,
    brain_threshold: float = DEFAULT_BRAIN_THRESHOLD,
) -> dict[str, Any] | None:
    """Per-tp legacy-seg: label preservation + per-label fraction inside ref brain mask."""
    if not run_manifest.get("include_legacy_seg"):
        return None
    ref_tp = run_manifest["reference_timepoint"]
    ref_t1 = _array(derivatives_root / f"tp-{ref_tp}" / f"tp-{ref_tp}_T1_to_ref.nii.gz")
    ref_brain = ref_t1 > (ref_t1.max() * brain_threshold)
    out: dict[str, Any] = {
        "brain_threshold": brain_threshold,
        "reference_brain_voxels": int(ref_brain.sum()),
        "timepoints": {},
    }
    seg_specs = (
        ("hdglio", "hdglio_registered_segmentation.nii.gz", "hdglio_to_ref_segmentation.nii.gz"),
        (
            "deepbratumia",
            "deepbratumia_native_segmentation.nii.gz",
            "deepbratumia_to_ref_segmentation.nii.gz",
        ),
    )
    for tp_meta in run_manifest["timepoints"]:
        tp = tp_meta["name"]
        out["timepoints"][tp] = {}
        for kind, src_suffix, dst_suffix in seg_specs:
            src = stage_root / f"tp-{tp}" / f"tp-{tp}_{src_suffix}"
            dst = derivatives_root / f"tp-{tp}" / f"tp-{tp}_{dst_suffix}"
            if not (src.is_file() and dst.is_file()):
                out["timepoints"][tp][kind] = {"status": "missing"}
                continue
            s = _array(src)
            d = _array(dst)
            s_voxml = _voxel_volume_ml(src)
            d_voxml = _voxel_volume_ml(dst)
            s_labels = sorted(int(v) for v in np.unique(s))
            d_labels = sorted(int(v) for v in np.unique(d))
            per_label = {}
            for lbl in d_labels:
                if lbl == 0:
                    continue
                mask = d == lbl
                n = int(mask.sum())
                inside = int((mask & ref_brain).sum())
                per_label[str(lbl)] = {
                    "voxels": n,
                    "volume_ml": round(n * d_voxml, 4),
                    "fraction_inside_ref_brain": (inside / n if n else None),
                }
            out["timepoints"][tp][kind] = {
                "status": "ok",
                "labels_preserved": s_labels == d_labels,
                "src_labels": s_labels,
                "dst_labels": d_labels,
                "src_volume_ml": round(int((s != 0).sum()) * s_voxml, 4),
                "dst_volume_ml": round(int((d != 0).sum()) * d_voxml, 4),
                "per_label": per_label,
            }
    return out


def _print_per_tp(rows: list[dict[str, Any]]) -> None:
    print("=== Per-timepoint registration QC ===")
    print(
        f"{'timepoint':<14}  {'role':<7}  {'moving':<6}  {'moving->ref corr':>17}  "
        f"all_chans_grid_OK"
    )
    print("-" * 72)
    for row in rows:
        role = "REF" if row["is_reference"] else "moving"
        corr = row["moving_to_ref_correlation"]
        corr_s = f"{corr:.3f}" if isinstance(corr, float) else "n/a"
        ok = "yes" if row["all_channels_match_ref_grid"] else "NO"
        print(f"{row['timepoint']:<14}  {role:<7}  {row['moving_channel']:<6}  {corr_s:>17}  {ok}")
    print()


def _print_cross_tp(corr: dict[str, dict[str, float]], channels: list[str]) -> None:
    print("=== Cross-timepoint same-channel correlation (vs reference) ===")
    print(f"{'timepoint':<14}  " + "  ".join(f"{ch:>7}" for ch in channels))
    print("-" * (16 + 9 * len(channels)))
    for tp, by_ch in corr.items():
        cells = "  ".join(f"{by_ch[ch]:>7.3f}" for ch in channels)
        print(f"{tp:<14}  {cells}")
    print()


def _print_legacy(diag: dict[str, Any] | None) -> None:
    if diag is None:
        print("=== Legacy segmentations: not requested (run with --include-legacy-seg) ===\n")
        return
    print("=== Legacy segmentations: label preservation + per-label inside-brain fraction ===")
    print(f"(brain mask = T1 > {diag['brain_threshold'] * 100:.0f}% * max)")
    for tp, kinds in diag["timepoints"].items():
        print(f"-- {tp} --")
        for kind, info in kinds.items():
            if info.get("status") != "ok":
                print(f"  {kind:<13}: {info.get('status')}")
                continue
            label_names = LABEL_NAMES_HDGLIO if kind == "hdglio" else LABEL_NAMES_DEEPBRATUMIA
            preserved = (
                "OK"
                if info["labels_preserved"]
                else (f"DRIFT src={info['src_labels']} dst={info['dst_labels']}")
            )
            print(
                f"  {kind:<13}: labels={preserved}  "
                f"src={info['src_volume_ml']:.2f} mL  dst={info['dst_volume_ml']:.2f} mL"
            )
            for lbl_str, per in info["per_label"].items():
                lbl = int(lbl_str)
                name = label_names.get(lbl, str(lbl))
                frac = per["fraction_inside_ref_brain"]
                frac_s = "n/a" if frac is None else f"{frac * 100:>5.1f}%"
                print(
                    f"      label {lbl} ({name:<18}): "
                    f"{per['volume_ml']:>7.2f} mL, inside_brain={frac_s}"
                )
    print()


def summarize(
    *,
    derivatives_root: Path,
    stage_root: Path | None,
) -> dict[str, Any]:
    run_manifest = _load_run_manifest(derivatives_root)
    if stage_root is None:
        stage_root = Path(run_manifest["stage_root"])

    rows = per_tp_table(run_manifest=run_manifest, derivatives_root=derivatives_root)
    corr = cross_tp_correlations(run_manifest=run_manifest, derivatives_root=derivatives_root)
    legacy = legacy_seg_diagnostics(
        run_manifest=run_manifest,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
    )

    _print_per_tp(rows)
    _print_cross_tp(corr, run_manifest["channels"])
    _print_legacy(legacy)

    return {
        "schema_version": "1.0",
        "cohort_id": run_manifest.get("cohort_id"),
        "derivatives_root": str(derivatives_root),
        "stage_root": str(stage_root),
        "reference_timepoint": run_manifest["reference_timepoint"],
        "channels": run_manifest["channels"],
        "per_timepoint": rows,
        "cross_tp_correlations": corr,
        "legacy_seg_diagnostics": legacy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Where the registration outputs live. Default: "
            f"{DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help=("Where the staged inputs live. Default: read from manifest.json's stage_root field."),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON sidecar to write the full summary into.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    derivs = args.derivatives_root.expanduser().resolve()
    stage = args.stage_root.expanduser().resolve() if args.stage_root else None
    if not derivs.is_dir():
        print(f"[err] derivatives-root not found: {derivs}", file=sys.stderr)
        return 2
    payload = summarize(derivatives_root=derivs, stage_root=stage)
    if args.json_out:
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[ok] wrote summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
