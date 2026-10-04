"""Build Figure 11 (R3): LUMIERE Patient-048 longitudinal Hit-Plot.

Reads the per-tp DL Hit-Plot CSVs produced by
``scripts/compute_hitplot_lumiere_p048.py`` (Step G.3) and renders a
two-panel heatmap that closes the round-3 loop opened by Step G.2 (the
real FreeSurfer 8.2.0 ``recon-all-clinical`` parcellation): the same
anatomy-aware Hit-Plot framework that powers the cohort-level
DL-vs-GT agreement panel (Figure~10) is now visualised across all six
Patient-048 timepoints, exposing how the WT and ET tumour burden
redistributes through Pre-Op -> Post-Op resection -> SD -> CR ->
recurrence -> terminal PD.

This figure replaces the v1 manual Freeview snapshot
(``LUMIERE-Patient-048-hd-glio-synthseg.png``, retired in favour of a
fully reproducible build), which used FreeSurfer 7.4.1 + HD-GLIO-AUTO
on hand-curated overlays.

Figure design
-------------
Two stacked sub-panels, ~9 x 8 in:

* Panel (a) -- WT (whole tumour). Heatmap of the top-K wmparc regions
  (ranked by max overlap volume across the 6 timepoints) on the y-axis,
  six timepoints on the x-axis. Cell colour and annotation = overlap
  volume in mL.
* Panel (b) -- ET (enhancing tumour). Same axes; the recurrence
  trajectory is dominated by ET so this is the clinically critical view.

X-axis tick labels carry the timepoint name, days-since-baseline, and
the manifest RANO label so the heatmap is self-explanatory.

Inputs (consumed; never modified)
---------------------------------
* ``data/lumiere_p048/derivatives/tp-<week>/hitplot/tp-<week>_hitplot_dl.csv``
  -- one per timepoint (Step G.3 output).
* ``configs/lumiere_p048_timepoints.yaml`` -- canonical ordering + RANO
  label for every timepoint.

Outputs (under ``--out-dir``, default ``outputs/figures/``)
----------------------------------------------------------
* ``fig11_lumiere_p048_hitplot.pdf`` -- vector primary asset.
* ``fig11_lumiere_p048_hitplot.png`` -- raster mirror.
* ``fig11_lumiere_p048_hitplot.json`` -- provenance + the matrices the
  caption auto-cites (top-K region order, per-cell overlap volumes,
  per-tp WT/ET totals, RANO labels).

Usage
-----

    uv run python scripts/build_figure11_lumiere_p048_hitplot.py
    uv run python scripts/build_figure11_lumiere_p048_hitplot.py --top-k 10
    uv run python scripts/build_figure11_lumiere_p048_hitplot.py \\
        --hitplot-root path/to/derivatives \\
        --out-dir outputs/figures

Exit codes
----------
0   figure written successfully
1   could not read inputs (manifest, CSVs, voxel volume)
2   inputs were valid but malformed (e.g. zero non-zero overlap rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HITPLOT_ROOT = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "figures"
DEFAULT_BASENAME = "fig11_lumiere_p048_hitplot"

FIGURE_SCHEMA_VERSION = "1.0"
DEFAULT_TOP_K = 12

# Matched to the WT (steel-blue) / ET (crimson) legend used in Figure 6
# so reviewers immediately read the panels as the same compartment.
WT_CMAP = LinearSegmentedColormap.from_list(
    "wt_blues", ["#f6fafc", "#cce0ec", "#5b9bd1", "#1f5d8b", "#0a3554"]
)
ET_CMAP = LinearSegmentedColormap.from_list(
    "et_reds", ["#fdf3f3", "#f7c9c9", "#e07474", "#a82828", "#5c0e0e"]
)


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _hitplot_csv_path(hitplot_root: Path, tp: str) -> Path:
    return hitplot_root / f"tp-{tp}" / "hitplot" / f"tp-{tp}_hitplot_dl.csv"


def _load_hitplot_csv(path: Path) -> list[dict[str, Any]]:
    """Read one per-tp DL Hit-Plot CSV; coerce numeric columns to float/int."""
    with path.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        rows = list(reader)
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            out.append(
                {
                    "label": int(r["label"]),
                    "name": r["name"],
                    "compartment": r["compartment"],
                    "parcel_voxels": int(r["parcel_voxels"]),
                    "overlap_voxels": int(r["overlap_voxels"]),
                    "overlap_volume_mm3": float(r["overlap_volume_mm3"]),
                    "pct_of_parcel": float(r["pct_of_parcel"]),
                    "pct_of_compartment": float(r["pct_of_compartment"]),
                }
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}: malformed row {r!r}: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# Heatmap matrix builders.
# ---------------------------------------------------------------------------


def _build_compartment_matrix(
    *,
    rows_per_tp: dict[str, list[dict[str, Any]]],
    tp_order: list[str],
    compartment: str,
    top_k: int,
) -> tuple[list[str], np.ndarray]:
    """Return ``(region_order, matrix)`` for one compartment.

    ``matrix`` is shape ``(top_k, n_tp)`` with cells = overlap volume in mL.
    Region ordering is by max overlap volume across the timepoints, descending.
    Regions named "Unknown" (label 0) are excluded -- they aggregate
    everything outside the parcellation (resection cavity, mass-effect
    distortion, etc.) and would dominate the heatmap with a value that
    has no anatomical meaning.
    """
    # Aggregate per-(region) max overlap mL across all tp.
    per_region_max: dict[str, float] = {}
    per_region_per_tp: dict[str, dict[str, float]] = {}
    for tp in tp_order:
        for r in rows_per_tp[tp]:
            if r["compartment"] != compartment:
                continue
            name = r["name"]
            if name == "Unknown":
                continue
            vol_ml = r["overlap_volume_mm3"] / 1000.0  # mm^3 -> mL
            cur = per_region_max.get(name, 0.0)
            if vol_ml > cur:
                per_region_max[name] = vol_ml
            per_region_per_tp.setdefault(name, {})[tp] = vol_ml

    if not per_region_max:
        raise ValueError(f"compartment {compartment!r} has no non-Unknown overlap rows in any tp")

    region_order = sorted(per_region_max, key=lambda r: per_region_max[r], reverse=True)[:top_k]
    matrix = np.zeros((len(region_order), len(tp_order)), dtype=np.float64)
    for i, region in enumerate(region_order):
        per_tp = per_region_per_tp.get(region, {})
        for j, tp in enumerate(tp_order):
            matrix[i, j] = per_tp.get(tp, 0.0)
    return region_order, matrix


def _per_tp_compartment_total(
    *,
    rows_per_tp: dict[str, list[dict[str, Any]]],
    tp_order: list[str],
    compartment: str,
    include_unknown: bool = True,
) -> dict[str, float]:
    """Per-tp total overlap mL (across all parcels) for one compartment."""
    out: dict[str, float] = {}
    for tp in tp_order:
        total = 0.0
        for r in rows_per_tp[tp]:
            if r["compartment"] != compartment:
                continue
            if not include_unknown and r["name"] == "Unknown":
                continue
            total += r["overlap_volume_mm3"] / 1000.0
        out[tp] = total
    return out


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _xtick_label(tp_meta: dict[str, Any]) -> str:
    """Pretty x-axis label: e.g. ``week-013\\n(91 d, SD)``."""
    name = tp_meta["name"]
    days = int(tp_meta.get("days_since_baseline", 0))
    rano = tp_meta.get("rano") or "-"
    return f"{name}\n({days} d, {rano})"


def _short_region_name(name: str) -> str:
    """Shrink long FS names so the y-axis ticks read cleanly."""
    repl = {
        "Right-": "R-",
        "Left-": "L-",
        "Cerebral-": "",
        "Cerebellum-": "Cb-",
        "UnsegmentedWhiteMatter": "UnsegWM",
        "ctx-rh-": "ctx-R-",
        "ctx-lh-": "ctx-L-",
        "wm-rh-": "wm-R-",
        "wm-lh-": "wm-L-",
        "rostralmiddlefrontal": "rostMidFront",
        "lateralorbitofrontal": "latOrbFront",
        "medialorbitofrontal": "medOrbFront",
        "caudalmiddlefrontal": "caudMidFront",
        "parstriangularis": "parsTri",
        "parsopercularis": "parsOper",
        "parsorbitalis": "parsOrb",
        "superiorfrontal": "supFront",
        "inferiorfrontal": "infFront",
        "supramarginal": "supraMarg",
    }
    short = name
    for k, v in repl.items():
        short = short.replace(k, v)
    return short


def _draw_heatmap_panel(
    *,
    ax,
    matrix: np.ndarray,
    region_order: list[str],
    tp_xlabels: list[str],
    cmap,
    panel_title: str,
    cbar_label: str,
    annotate_threshold: float,
) -> None:
    """Render one compartment's heatmap with annotations + colour bar."""
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(tp_xlabels, fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [_short_region_name(r) for r in region_order],
        fontsize=8.5,
    )
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate each cell with the value in mL when > threshold (else skip
    # to keep the panel readable).
    vmax = float(matrix.max()) if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v < annotate_threshold:
                continue
            # White text on dark cells, black on light cells.
            text_color = "white" if (vmax > 0 and v / vmax > 0.55) else "#222222"
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    ax.set_title(panel_title, fontsize=11, loc="left", pad=6)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _build_figure(
    *,
    rows_per_tp: dict[str, list[dict[str, Any]]],
    timepoints_meta: list[dict[str, Any]],
    top_k: int,
) -> tuple[plt.Figure, dict[str, Any]]:
    tp_order = [tp["name"] for tp in timepoints_meta]
    tp_xlabels = [_xtick_label(tp) for tp in timepoints_meta]

    wt_regions, wt_matrix = _build_compartment_matrix(
        rows_per_tp=rows_per_tp, tp_order=tp_order, compartment="WT", top_k=top_k
    )
    et_regions, et_matrix = _build_compartment_matrix(
        rows_per_tp=rows_per_tp, tp_order=tp_order, compartment="ET", top_k=top_k
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.6),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.42},
    )

    wt_title = f"(a) WT (whole tumour) -- top-{len(wt_regions)} wmparc regions by max overlap"
    et_title = f"(b) ET (enhancing tumour) -- top-{len(et_regions)} wmparc regions by max overlap"
    _draw_heatmap_panel(
        ax=axes[0],
        matrix=wt_matrix,
        region_order=wt_regions,
        tp_xlabels=tp_xlabels,
        cmap=WT_CMAP,
        panel_title=wt_title,
        cbar_label="overlap volume (mL)",
        annotate_threshold=0.5,
    )
    _draw_heatmap_panel(
        ax=axes[1],
        matrix=et_matrix,
        region_order=et_regions,
        tp_xlabels=tp_xlabels,
        cmap=ET_CMAP,
        panel_title=et_title,
        cbar_label="overlap volume (mL)",
        annotate_threshold=0.1,
    )

    fig.suptitle(
        "LUMIERE Patient-048 longitudinal Hit-Plot",
        fontsize=12,
        y=0.995,
    )

    # Per-tp totals (excluding the BG/Unknown bucket) for the JSON sidecar.
    wt_totals = _per_tp_compartment_total(
        rows_per_tp=rows_per_tp,
        tp_order=tp_order,
        compartment="WT",
        include_unknown=False,
    )
    et_totals = _per_tp_compartment_total(
        rows_per_tp=rows_per_tp,
        tp_order=tp_order,
        compartment="ET",
        include_unknown=False,
    )

    summary = {
        "tp_order": tp_order,
        "wt_top_regions": wt_regions,
        "wt_overlap_ml_matrix": wt_matrix.tolist(),
        "et_top_regions": et_regions,
        "et_overlap_ml_matrix": et_matrix.tolist(),
        "per_tp_wt_total_ml_excl_unknown": wt_totals,
        "per_tp_et_total_ml_excl_unknown": et_totals,
    }
    return fig, summary


# ---------------------------------------------------------------------------
# Driver.
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
        "--hitplot-root",
        type=Path,
        default=DEFAULT_HITPLOT_ROOT,
        help=(
            "Root containing the per-tp Hit-Plot CSVs. Default: "
            f"{DEFAULT_HITPLOT_ROOT.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=DEFAULT_BASENAME,
        help=f"Basename for the three output files. Default: {DEFAULT_BASENAME}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K regions per compartment to display. Default: {DEFAULT_TOP_K}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    hitplot_root = args.hitplot_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not manifest_path.is_file():
        print(f"[err] manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not hitplot_root.is_dir():
        print(f"[err] hitplot-root not found: {hitplot_root}", file=sys.stderr)
        return 1

    try:
        manifest = _load_manifest(manifest_path)
    except (OSError, yaml.YAMLError) as exc:
        print(f"[err] could not load manifest: {exc}", file=sys.stderr)
        return 1
    timepoints_meta = manifest["timepoints"]
    if not timepoints_meta:
        print("[err] manifest has zero timepoints", file=sys.stderr)
        return 2

    rows_per_tp: dict[str, list[dict[str, Any]]] = {}
    for tp_meta in timepoints_meta:
        tp = tp_meta["name"]
        csv_path = _hitplot_csv_path(hitplot_root, tp)
        if not csv_path.is_file():
            print(
                f"[err] missing per-tp Hit-Plot CSV: {csv_path}\n"
                "  Run scripts/compute_hitplot_lumiere_p048.py (Step G.3) first.",
                file=sys.stderr,
            )
            return 1
        try:
            rows_per_tp[tp] = _load_hitplot_csv(csv_path)
        except ValueError as exc:
            print(f"[err] {exc}", file=sys.stderr)
            return 2

    try:
        fig, summary = _build_figure(
            rows_per_tp=rows_per_tp,
            timepoints_meta=timepoints_meta,
            top_k=args.top_k,
        )
    except ValueError as exc:
        print(f"[err] {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.basename}.pdf"
    png_path = out_dir / f"{args.basename}.png"
    json_path = out_dir / f"{args.basename}.json"

    fig.savefig(str(pdf_path), bbox_inches="tight")
    fig.savefig(str(png_path), bbox_inches="tight", dpi=200)
    plt.close(fig)

    payload = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_id": args.basename,
        "title": "LUMIERE Patient-048 longitudinal Hit-Plot",
        "caption_disclosure": (
            "LUMIERE Patient-048 timepoint identifiers report the per-week-floor"
            " anonymisation provided by the LUMIERE consortium; the days-since-baseline"
            " axis is the manifest-canonical (0, 3, 91, 161, 315, 343) sequence."
        ),
        "cohort_id": manifest.get("cohort_id"),
        "patient_id": manifest.get("patient_id"),
        "manifest_path": (
            str(manifest_path.relative_to(REPO_ROOT))
            if manifest_path.is_relative_to(REPO_ROOT)
            else str(manifest_path)
        ),
        "hitplot_root": str(hitplot_root),
        "top_k": args.top_k,
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "panels": {
            "a_wt": {
                "compartment": "WT",
                "n_regions": len(summary["wt_top_regions"]),
                "regions_top_to_bottom": summary["wt_top_regions"],
                "matrix_overlap_ml": summary["wt_overlap_ml_matrix"],
                "tp_order_left_to_right": summary["tp_order"],
                "per_tp_total_ml_excl_unknown": summary["per_tp_wt_total_ml_excl_unknown"],
            },
            "b_et": {
                "compartment": "ET",
                "n_regions": len(summary["et_top_regions"]),
                "regions_top_to_bottom": summary["et_top_regions"],
                "matrix_overlap_ml": summary["et_overlap_ml_matrix"],
                "tp_order_left_to_right": summary["tp_order"],
                "per_tp_total_ml_excl_unknown": summary["per_tp_et_total_ml_excl_unknown"],
            },
        },
        "outputs": {
            "pdf": str(pdf_path.relative_to(REPO_ROOT))
            if pdf_path.is_relative_to(REPO_ROOT)
            else str(pdf_path),
            "png": str(png_path.relative_to(REPO_ROOT))
            if png_path.is_relative_to(REPO_ROOT)
            else str(png_path),
            "json": str(json_path.relative_to(REPO_ROOT))
            if json_path.is_relative_to(REPO_ROOT)
            else str(json_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"[ok] wrote {pdf_path}")
    print(f"[ok] wrote {png_path}")
    print(f"[ok] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
