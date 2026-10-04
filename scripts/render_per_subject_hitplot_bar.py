"""Render the per-subject Hit-Plot bar chart (panel `i` of Figs. 4/5/12/13/14).

Consumes one Hit-Plot CSV (the long-format dataframe written by
``scripts/compute_legacy5_hitplot_gt.py`` or
``scripts/compute_hitplot_cohort50.py``) and emits a horizontal stacked
bar chart of the top-K wmparc regions ranked by total tumor-overlap
volume across all three compartments. Bars are stacked by compartment
in the canonical NCR/ED/ET order with the same palette as Fig. 3 panel
(e):

    NCR  red     (label 1)
    ED   green   (label 2)
    ET   yellow  (label 4)

Region labels read directly from the CSV's ``name`` column (FreeSurfer
``aparc+aseg`` / ``wmparc`` LUT names). Each bar is annotated to the
right with the per-region totals in millilitres.

Usage::

    uv run python scripts/render_per_subject_hitplot_bar.py \\
        --csv data/derivatives_legacy5/sub-0020/hitplot/sub-0020_hitplot_gt.csv \\
        --subject 0020 \\
        --source gt \\
        --top-k 12 \\
        --out-pdf outputs/figures/legacy5/sub-0020_panel_i_hitplot_gt.pdf \\
        --out-png outputs/figures/legacy5/sub-0020_panel_i_hitplot_gt.png

The default colour palette matches ``scripts/render_figure2_panels.py``
``TUMOR_COLORS`` so a reader scanning panels (a-h) of Fig. 3 finds the
same colour key on the new (i) panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Mirror of TUMOR_COLORS in scripts/render_figure2_panels.py (NCR=1 red,
# ED=2 green, ET=4 yellow). Kept literally identical so the manuscript
# panels share a single colour key across the (a-h) anatomy panels and
# the new (i) bar chart.
COMPARTMENT_COLORS: dict[str, tuple[float, float, float]] = {
    "NCR": (1.00, 0.235, 0.235),  # 255, 60, 60 / 255
    "ED": (0.235, 0.863, 0.314),  # 60, 220, 80 / 255
    "ET": (1.00, 0.843, 0.000),  # 255, 215, 0 / 255
}

# Disjoint BraTS-style compartments used in the manuscript captions.
# NCR + ED + ET = WT (the whole tumor) by construction. We always stack
# in this order so the colour key reads NCR-bottom, ED-middle, ET-top
# left-to-right when the bars are horizontal.
CANONICAL_COMPARTMENT_ORDER: tuple[str, ...] = ("NCR", "ED", "ET")
# Hierarchical BraTS subregions written by hpgs.hitplot.compartment_region_matrix
# (WT \supset TC \supset ET). These overlap by construction and CANNOT be
# stacked directly; we translate them to the disjoint canonical order
# inside aggregate_top_k.
HIERARCHICAL_COMPARTMENT_ORDER: tuple[str, ...] = ("WT", "TC", "ET")


def _hierarchical_to_disjoint(volumes: dict[str, float]) -> dict[str, float]:
    """Translate hierarchical (WT, TC, ET) volumes to disjoint (NCR, ED, ET).

    Per the BraTS schema:

        WT (whole tumor)       = labels {1, 2, 4} = NCR + ED + ET
        TC (tumor core)        = labels {1, 4}    = NCR + ET
        ET (enhancing tumor)   = labels {4}       = ET

    so the disjoint compartments are recovered by

        ET   = ET                     (unchanged)
        NCR  = TC - ET                (necroses + non-enhancing)
        ED   = WT - TC                (FLAIR-abnormal edema only)

    Negative differences (which can only arise from numerical noise in
    upstream voxel counts, since the underlying integer counts always
    satisfy the hierarchy) are clipped to zero so the bar geometry
    stays non-negative.
    """
    et = float(volumes.get("ET", 0.0))
    tc = float(volumes.get("TC", 0.0))
    wt = float(volumes.get("WT", 0.0))
    ncr = max(tc - et, 0.0)
    ed = max(wt - tc, 0.0)
    return {"NCR": ncr, "ED": ed, "ET": et}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", type=Path, required=True, help="Per-subject Hit-Plot CSV.")
    p.add_argument(
        "--subject",
        type=str,
        required=True,
        help="UCSF-PDGM subject id (used in title / filename only).",
    )
    p.add_argument(
        "--source",
        type=str,
        default="gt",
        help=(
            "Free-form source tag for the title (e.g. 'gt' for the dataset-provided "
            "segmentation, 'dl' for the unified DL output). Default: gt."
        ),
    )
    p.add_argument("--top-k", type=int, default=12, help="Number of regions to show.")
    p.add_argument("--out-pdf", type=Path, required=True)
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Suppress the figure title (useful when used as a sub-panel).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pure helpers (CSV in, dataframe out) -- keep ax-painting separate so tests
# don't need matplotlib displays.
# ---------------------------------------------------------------------------


def _detect_compartment_schema(df: pd.DataFrame) -> tuple[str, ...]:
    """Return the compartment schema present in ``df``.

    Returns either :data:`CANONICAL_COMPARTMENT_ORDER` (the disjoint
    NCR/ED/ET form, used by the manuscript captions) or
    :data:`HIERARCHICAL_COMPARTMENT_ORDER` (the BraTS hierarchical
    WT/TC/ET form written by :func:`hpgs.hitplot.compartment_region_matrix`).
    Any other shape raises :class:`ValueError`.
    """
    present = set(df["compartment"].unique())
    if {"NCR", "ED", "ET"} <= present:
        return CANONICAL_COMPARTMENT_ORDER
    if {"WT", "TC", "ET"} <= present:
        return HIERARCHICAL_COMPARTMENT_ORDER
    raise ValueError(
        f"compartment column has unexpected values: {sorted(present)}; "
        "expected superset of {NCR,ED,ET} or {WT,TC,ET}",
    )


def aggregate_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Aggregate a Hit-Plot CSV to the top-K regions by total overlap volume.

    Returns a wide-format dataframe indexed by region ``name`` with one
    column per *disjoint* compartment in the canonical NCR/ED/ET order
    (always; hierarchical WT/TC/ET CSVs are translated internally via
    :func:`_hierarchical_to_disjoint` so the bars are stackable and
    NCR + ED + ET = WT for every region row). Sorted by row sum
    (= WT volume per region) descending and truncated to the top-K rows.

    Region rows whose total overlap volume is exactly 0 across all
    three disjoint compartments are dropped before the top-K cut so the
    chart never shows empty bars. The ``Unknown`` (label 0) row is
    dropped because it represents background voxels and would visually
    drown out the anatomical regions a reader cares about.
    """
    if k <= 0:
        raise ValueError(f"top_k must be positive, got {k}")
    needed = {"label", "name", "compartment", "overlap_volume_mm3"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    schema = _detect_compartment_schema(df)

    # Drop background label 0 (FreeSurfer 'Unknown'); it is the largest
    # parcel by voxel count and would dominate every bar otherwise.
    df = df[df["label"] != 0]

    pivoted = df.pivot_table(
        index="name",
        columns="compartment",
        values="overlap_volume_mm3",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=list(schema), fill_value=0.0)

    if schema == HIERARCHICAL_COMPARTMENT_ORDER:
        # WT/TC/ET overlap by construction; translate to the disjoint
        # NCR/ED/ET form before stacking. Operate row-by-row so the
        # clipping in _hierarchical_to_disjoint runs on per-region
        # numbers (per-region clipping is the correct granularity).
        rows: list[dict[str, float]] = []
        for _, row in pivoted.iterrows():
            rows.append(_hierarchical_to_disjoint(row.to_dict()))
        wide = pd.DataFrame(rows, index=pivoted.index)
        wide = wide.reindex(columns=list(CANONICAL_COMPARTMENT_ORDER), fill_value=0.0)
    else:
        wide = pivoted

    wide = wide.assign(_total=wide.sum(axis=1))
    wide = wide[wide["_total"] > 0]
    wide = wide.sort_values("_total", ascending=False).head(k)
    wide = wide.drop(columns="_total")
    return wide


def _format_total_label(row_volumes_mm3: pd.Series) -> str:
    """Return the per-region annotation string (mL totals per compartment)."""
    parts: list[str] = []
    for comp, vol in row_volumes_mm3.items():
        if vol <= 0:
            continue
        parts.append(f"{comp} {vol / 1000.0:.1f}\u202fmL")
    return "  ".join(parts) if parts else "0"


def render_bar_chart(
    df: pd.DataFrame,
    *,
    subject_id: str,
    source: str,
    title: str | None,
) -> plt.Figure:
    """Render the horizontal stacked bar chart from an aggregated dataframe.

    ``df`` is the wide-format frame returned by :func:`aggregate_top_k`
    (rows = region names, columns = compartments). Returns the matplotlib
    Figure; caller is responsible for saving + closing it.
    """
    if df.empty:
        raise ValueError(f"sub-{subject_id} [{source}]: no non-zero region rows to plot")
    comp_order = list(df.columns)
    # Display top region at the top of the chart (matplotlib horizontal
    # bars stack bottom-up, so reverse).
    df_display = df.iloc[::-1]

    n_rows = len(df_display)
    fig_h = max(2.6, 0.34 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))

    y_pos = range(n_rows)
    left = [0.0] * n_rows
    bar_handles = []
    for comp in comp_order:
        vols_ml = (df_display[comp] / 1000.0).to_numpy()
        bar = ax.barh(
            y_pos,
            vols_ml,
            left=left,
            color=COMPARTMENT_COLORS[comp],
            edgecolor="black",
            linewidth=0.4,
            label=comp,
        )
        bar_handles.append(bar)
        left = [a + b for a, b in zip(left, vols_ml, strict=False)]

    # Per-region total annotation to the right of each bar.
    max_total = max(left) if left else 0.0
    pad = max_total * 0.012 if max_total > 0 else 0.05
    text_artists: list[plt.Text] = []
    for idx, name in enumerate(df_display.index):
        annotation = _format_total_label(df_display.loc[name])
        t = ax.text(
            left[idx] + pad,
            idx,
            annotation,
            va="center",
            ha="left",
            fontsize=8,
        )
        text_artists.append(t)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(df_display.index), fontsize=9)
    ax.set_xlabel("Tumor overlap volume (mL)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)

    fallback_xmax = max_total * 1.30 if max_total > 0 else 1.0
    ax.set_xlim(0, fallback_xmax)

    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        title="Compartment",
        title_fontsize=9,
    )

    if title is not None:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()

    # Size the x-axis so that the per-row right-side annotation strings
    # fit inside the axes frame. Done AFTER tight_layout (so the axes
    # box is at its final pixel width) and iterated to convergence:
    # expanding xlim stretches the data range, which means the
    # fixed-pixel text occupies more data units than it did before the
    # expansion, so a single-pass measurement undershoots. Fall back to
    # the fixed 1.30x heuristic in headless environments where the
    # canvas may not initialise.
    margin_data = (max_total if max_total > 0 else 1.0) * 0.04
    try:
        for _ in range(6):
            fig.canvas.draw()
            inv = ax.transData.inverted()
            right_edges_data: list[float] = []
            for t in text_artists:
                bbox_disp = t.get_window_extent(renderer=fig.canvas.get_renderer())
                x_right_data, _ = inv.transform((bbox_disp.x1, 0))
                right_edges_data.append(float(x_right_data))
            text_xmax = max(right_edges_data) if right_edges_data else 0.0
            target_xmax = max(fallback_xmax, text_xmax + margin_data)
            cur_xmax = ax.get_xlim()[1]
            if abs(target_xmax - cur_xmax) <= 0.005 * (max_total if max_total > 0 else 1.0):
                break
            ax.set_xlim(0, target_xmax)
    except Exception:
        ax.set_xlim(0, fallback_xmax)

    return fig


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        sys.argv = ["render_per_subject_hitplot_bar.py", *argv]
    args = _parse_args()

    if not args.csv.is_file():
        print(f"[error] CSV not found: {args.csv}")
        return 2

    df = pd.read_csv(args.csv)
    aggregated = aggregate_top_k(df, args.top_k)

    title = (
        None
        if args.no_title
        else (
            f"UCSF-PDGM-{args.subject} \u2014 Hit-Plot (top {args.top_k}, "
            f"source = {args.source})"
        )
    )
    fig = render_bar_chart(
        aggregated,
        subject_id=args.subject,
        source=args.source,
        title=title,
    )

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] sub-{args.subject} [{args.source}]: top-{args.top_k} bars")
    print(f"     wrote {args.out_pdf}")
    print(f"     wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
