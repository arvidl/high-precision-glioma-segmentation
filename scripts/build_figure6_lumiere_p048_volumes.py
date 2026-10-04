"""Build the round-3 Figure 6: LUMIERE Patient-048 longitudinal tumour volumes.

This is the *replacement* producer for the old BGO-D Fig. 7 (which lived in
the unreproducible ``glioma-recurrence`` repo on REK-restricted data). The
round-3 strategic pivot keeps Patient-048 -- the openly-shareable LUMIERE
case already used as Fig. 6's anatomical overlay -- as the longitudinal
illustrative subject, but uses the unified MONAI Bundle BraTS three-class
SegResNet (the same engine evaluated on UCSF-PDGM in Section 4) so the
end-to-end pipeline is fully reproducible from a public archive.

Inputs (consumed but not modified)
----------------------------------
- ``data/lumiere_p048/derivatives/segmentation_summary.csv`` -- one row per
  timepoint (produced by ``scripts/segment_lumiere_p048.py``).
- ``configs/lumiere_p048_timepoints.yaml`` -- canonical day-axis + RANO
  labels (single source of truth; the CSV's ``rano`` column is used only
  as a fallback).

Outputs (under ``--out-dir``, default ``outputs/figures/``)
----------------------------------------------------------
- ``fig6_lumiere_p048_volumes.pdf`` -- the figure (vector; primary asset
  consumed by ``paper/main.tex``).
- ``fig6_lumiere_p048_volumes.png`` -- raster mirror for previews / web.
- ``fig6_lumiere_p048_volumes.json`` -- provenance + the summary numbers
  the caption auto-cites (peak WT, post-op ET drop, recurrence ET surge).

Figure design (R3 spec)
-----------------------
Single panel, ~9x5 in, paper-quality matplotlib::

    y: tumour volume (mL), linear
    x: days_since_baseline, continuous, anchored at the manifest

    WT  (whole tumour)     -- steel-blue,   thick line + filled circles
    TC  (tumour core)      -- orange,        thick line + filled triangles
    ET  (enhancing tumour) -- crimson,       thick line + filled squares
    ED  (peritumoral edema) -- light-green,  thin line + open circles
                              (shown for context; ED = WT \\ TC)

Each timepoint is annotated with its RANO label above the marker; the
two clinically critical events -- the post-op ET drop and the recurrence
ET surge at the last timepoint -- get a callout. The caption-disclosure
sentence about the LUMIERE week-floor anonymisation is included in the
JSON sidecar (the manuscript caption picks it up verbatim).

Usage
-----

    # Default: read summary.csv + manifest, write PDF/PNG/JSON to
    # outputs/figures/, ready for `make sync`.
    uv run python scripts/build_figure6_lumiere_p048_volumes.py

    # Point at a different summary (e.g. a re-run with --backend dummy):
    uv run python scripts/build_figure6_lumiere_p048_volumes.py \\
        --summary-csv path/to/segmentation_summary.csv

Exit codes
----------
0   figure written successfully
1   could not read inputs (CSV or manifest)
2   inputs were valid but malformed (e.g. zero rows, missing columns)
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "data" / "lumiere_p048" / "derivatives" / "segmentation_summary.csv"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "figures"
DEFAULT_BASENAME = "fig6_lumiere_p048_volumes"

FIGURE_SCHEMA_VERSION = "1.0"

REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "timepoint",
    "days_since_baseline",
    "rano",
    "vol_WT_ml",
    "vol_TC_ml",
    "vol_ET_ml",
    "vol_NCR_ml",
    "vol_ED_ml",
)

# Compartment styling (BraTS WT/TC/ET are the headline curves; ED is shown
# for context as a thinner secondary line).
COMPARTMENT_STYLE: dict[str, dict[str, Any]] = {
    "WT": {
        "label": "WT (whole tumour)",
        "color": "#1f77b4",
        "marker": "o",
        "lw": 2.2,
        "zorder": 4,
    },
    "TC": {
        "label": "TC (tumour core)",
        "color": "#ff7f0e",
        "marker": "^",
        "lw": 2.0,
        "zorder": 5,
    },
    "ET": {
        "label": "ET (enhancing tumour)",
        "color": "#d62728",
        "marker": "s",
        "lw": 2.0,
        "zorder": 6,
    },
    "ED": {
        "label": "ED (peritumoral edema)",
        "color": "#2ca02c",
        "marker": "o",
        "lw": 1.2,
        "alpha": 0.7,
        "linestyle": "--",
        "markerfacecolor": "white",
        "zorder": 3,
    },
}
COMPARTMENTS_PRIMARY: tuple[str, ...] = ("WT", "TC", "ET")
COMPARTMENTS_SECONDARY: tuple[str, ...] = ("ED",)


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def _load_summary_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read segmentation_summary.csv, validate required columns, sort by day."""
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header")
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path}: zero rows")
    parsed: list[dict[str, Any]] = []
    for r in rows:
        try:
            parsed.append(
                {
                    "timepoint": r["timepoint"],
                    "days_since_baseline": float(r["days_since_baseline"]),
                    "rano": (r.get("rano") or "").strip(),
                    "vol_WT_ml": float(r["vol_WT_ml"]),
                    "vol_TC_ml": float(r["vol_TC_ml"]),
                    "vol_ET_ml": float(r["vol_ET_ml"]),
                    "vol_NCR_ml": float(r["vol_NCR_ml"]),
                    "vol_ED_ml": float(r["vol_ED_ml"]),
                }
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{csv_path}: malformed row {r!r}: {exc}") from exc
    parsed.sort(key=lambda x: x["days_since_baseline"])
    return parsed


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _merge_rano_from_manifest(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Manifest is the single source of truth for RANO; CSV is the fallback."""
    by_name = {tp["name"]: tp for tp in manifest.get("timepoints", [])}
    for r in rows:
        meta = by_name.get(r["timepoint"])
        if meta is not None:
            rano_meta = meta.get("rano")
            if rano_meta:
                r["rano"] = str(rano_meta).strip()
            r["rano_rationale"] = (meta.get("rano_rationale") or "").strip()
        else:
            r.setdefault("rano_rationale", "")
    return rows


# ---------------------------------------------------------------------------
# Caption-summary numbers (small, JSON-serializable).
# ---------------------------------------------------------------------------


def _summarize_curves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the caption-cited landmarks: peak WT, post-op ET drop, ET surge."""
    if not rows:
        return {}
    days = [r["days_since_baseline"] for r in rows]
    out: dict[str, Any] = {
        "n_timepoints": len(rows),
        "day_axis": {"min": min(days), "max": max(days)},
        "peaks": {},
        "events": {},
    }
    for comp in ("WT", "TC", "ET", "ED", "NCR"):
        col = f"vol_{comp}_ml"
        peak_row = max(rows, key=lambda r, c=col: r[c])
        out["peaks"][comp] = {
            "timepoint": peak_row["timepoint"],
            "days_since_baseline": peak_row["days_since_baseline"],
            "rano": peak_row["rano"],
            "value_ml": round(peak_row[col], 4),
        }
    pre_op = next((r for r in rows if r["rano"].lower() == "pre-op"), None)
    post_op = next((r for r in rows if r["rano"].lower() == "post-op"), None)
    if pre_op is not None and post_op is not None:
        out["events"]["postop_ET_drop_ml"] = round(pre_op["vol_ET_ml"] - post_op["vol_ET_ml"], 4)
        out["events"]["postop_WT_change_ml"] = round(post_op["vol_WT_ml"] - pre_op["vol_WT_ml"], 4)
    pd_rows = [r for r in rows if r["rano"].lower() == "pd"]
    if pd_rows:
        last_pd = pd_rows[-1]
        # Compare ET at last PD against the minimum ET seen between baseline
        # post-op and the first PD timepoint -- the "trough" the recurrence
        # rises from.
        first_pd_day = pd_rows[0]["days_since_baseline"]
        trough_pool = [
            r
            for r in rows
            if (post_op is None or r["days_since_baseline"] >= post_op["days_since_baseline"])
            and r["days_since_baseline"] < first_pd_day
        ]
        if trough_pool:
            trough = min(trough_pool, key=lambda r: r["vol_ET_ml"])
            out["events"]["recurrence_ET_trough_ml"] = round(trough["vol_ET_ml"], 4)
            out["events"]["recurrence_ET_trough_timepoint"] = trough["timepoint"]
        out["events"]["recurrence_ET_at_last_PD_ml"] = round(last_pd["vol_ET_ml"], 4)
        out["events"]["recurrence_ET_last_PD_timepoint"] = last_pd["timepoint"]
    return out


# ---------------------------------------------------------------------------
# Figure rendering.
# ---------------------------------------------------------------------------


def _plot_compartment_curves(ax: Any, rows: list[dict[str, Any]], days: list[float]) -> None:
    """Plot ED first so the headline WT/TC/ET sit on top."""
    for comp in (*COMPARTMENTS_SECONDARY, *COMPARTMENTS_PRIMARY):
        style = COMPARTMENT_STYLE[comp]
        col = f"vol_{comp}_ml"
        y = [r[col] for r in rows]
        ax.plot(
            days,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=7 if comp in COMPARTMENTS_PRIMARY else 5,
            alpha=style.get("alpha", 1.0),
            linestyle=style.get("linestyle", "-"),
            markerfacecolor=style.get("markerfacecolor", style["color"]),
            markeredgecolor=style["color"],
            zorder=style["zorder"],
        )


def _annotate_rano_labels(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    band_low: float,
    band_high: float,
    collision_eps_days: float,
) -> None:
    """RANO labels above the curves; adjacent tps stagger onto two rows."""
    use_high_row = False
    prev_day: float | None = None
    for r in rows:
        d = r["days_since_baseline"]
        if prev_day is not None and (d - prev_day) < collision_eps_days:
            use_high_row = not use_high_row
        else:
            use_high_row = False
        label_y = band_high if use_high_row else band_low
        ax.annotate(
            r["rano"],
            xy=(d, label_y),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "lightgray",
                "lw": 0.5,
            },
            zorder=10,
        )
        if use_high_row:
            ax.plot(
                [d, d],
                [band_low * 0.985, label_y * 0.965],
                color="lightgray",
                lw=0.6,
                ls=":",
                zorder=2,
            )
        prev_day = d


def _annotate_events(ax: Any, rows: list[dict[str, Any]], events: dict[str, Any]) -> None:
    """Clinical-event callouts: post-op resection drop and recurrence surge."""
    postop_drop = events.get("postop_ET_drop_ml")
    if postop_drop is not None and postop_drop > 0:
        post_op_row = next((r for r in rows if r["rano"].lower() == "post-op"), None)
        if post_op_row is not None:
            ax.annotate(
                f"resection:\nET -{postop_drop:.1f} mL",
                xy=(post_op_row["days_since_baseline"], post_op_row["vol_ET_ml"]),
                xytext=(
                    post_op_row["days_since_baseline"] + 30,
                    post_op_row["vol_ET_ml"] + 32,
                ),
                fontsize=8,
                color="#444444",
                arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 0.7},
            )
    last_pd_tp = events.get("recurrence_ET_last_PD_timepoint")
    trough_ml = events.get("recurrence_ET_trough_ml")
    last_pd_et = events.get("recurrence_ET_at_last_PD_ml")
    if last_pd_tp is not None and trough_ml is not None and last_pd_et is not None:
        last_pd_row = next((r for r in rows if r["timepoint"] == last_pd_tp), None)
        if last_pd_row is not None:
            delta = last_pd_et - trough_ml
            ax.annotate(
                f"recurrence:\nET +{delta:.1f} mL",
                xy=(last_pd_row["days_since_baseline"], last_pd_row["vol_ET_ml"]),
                xytext=(
                    last_pd_row["days_since_baseline"] - 110,
                    last_pd_row["vol_ET_ml"] + 22,
                ),
                fontsize=8,
                color="#444444",
                arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 0.7},
            )


def render_figure(
    rows: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
    summary: dict[str, Any],
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    days = [r["days_since_baseline"] for r in rows]
    day_span = max(days) - min(days) if len(days) > 1 else 1.0
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    for d in days:
        ax.axvline(d, color="lightgray", lw=0.7, ls=":", zorder=1)
    _plot_compartment_curves(ax, rows, days)

    y_max = max(r["vol_WT_ml"] for r in rows)
    band_low = y_max * 1.06
    band_high = y_max * 1.20
    _annotate_rano_labels(
        ax,
        rows,
        band_low=band_low,
        band_high=band_high,
        collision_eps_days=max(day_span * 0.05, 5.0),
    )
    _annotate_events(ax, rows, summary.get("events", {}))

    ax.set_xlabel("Days since baseline (Pre-Op)", fontsize=11)
    ax.set_ylabel("Tumour-compartment volume (mL)", fontsize=11)
    ax.set_title(
        "LUMIERE Patient-048 -- longitudinal tumour volumes\n"
        "(unified MONAI Bundle BraTS three-class SegResNet, n = 6 timepoints)",
        fontsize=11,
    )
    ax.set_xlim(min(days) - 15, max(days) + 25)
    # Reserve headroom for the (potentially staggered) RANO labels.
    ax.set_ylim(0, band_high * 1.06)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    # Legend below the axis to free the in-plot area for RANO labels +
    # clinical-event callouts.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Provenance JSON sidecar.
# ---------------------------------------------------------------------------


def _write_json_sidecar(
    *,
    out_json: Path,
    summary_csv: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    out_pdf: Path,
    out_png: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figure_id": "fig6_lumiere_p048_volumes",
        "title": (
            "LUMIERE Patient-048 -- longitudinal tumour volumes "
            "(unified MONAI Bundle BraTS three-class SegResNet, n = 6 timepoints)"
        ),
        "cohort_id": manifest.get("cohort_id"),
        "patient_id": manifest.get("patient_id"),
        "inputs": {
            "summary_csv": str(summary_csv),
            "manifest_yaml": str(manifest_path),
        },
        "outputs": {
            "pdf": str(out_pdf),
            "png": str(out_png),
        },
        "caption_disclosure": (
            manifest.get("time_axis", {}).get("caption_disclosure") or ""
        ).strip(),
        "summary": summary,
        "table": rows,
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Per-tp segmentation CSV. Default: {DEFAULT_SUMMARY.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Cohort manifest YAML. Default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}",
    )
    out_dir_default_rel = DEFAULT_OUT_DIR.relative_to(REPO_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write the figure + sidecar. Default: {out_dir_default_rel}",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_BASENAME,
        help="Output basename (without extension). Default: %(default)s",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_csv = args.summary_csv.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not summary_csv.is_file():
        print(f"[err] summary CSV not found: {summary_csv}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"[err] manifest YAML not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        rows = _load_summary_rows(summary_csv)
        manifest = _load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        print(f"[err] could not read inputs: {exc}", file=sys.stderr)
        return 2

    rows = _merge_rano_from_manifest(rows, manifest)
    summary = _summarize_curves(rows)

    out_pdf = out_dir / f"{args.basename}.pdf"
    out_png = out_dir / f"{args.basename}.png"
    out_json = out_dir / f"{args.basename}.json"

    render_figure(rows, out_pdf=out_pdf, out_png=out_png, summary=summary)
    _write_json_sidecar(
        out_json=out_json,
        summary_csv=summary_csv,
        manifest_path=manifest_path,
        rows=rows,
        summary=summary,
        manifest=manifest,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    rel = out_pdf.relative_to(REPO_ROOT) if out_pdf.is_relative_to(REPO_ROOT) else out_pdf
    print(f"[ok] wrote {rel}  (n_timepoints={summary.get('n_timepoints')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
