"""Supplementary 5\u00d71 DL-vs-GT per-subject Hit-Plot panel.

Builds the supplementary figure that complements the per-subject Hit-Plot
bar charts in the main text (Figs.~4, 5, 12, 13, 14). Those main-text
bars are computed from the dataset-provided segmentation
(``hitplot_gt.csv``) on the *legacy-5* detail subjects, which are
deliberately leak-unsafe for DL inference (4/5 are in the BraTS21
Segmentation *Training* cohort; 1/5 in *Validation*; see
``configs/cohort_legacy5.yaml``). To answer R2.8 (DL-vs-GT agreement at
the per-subject level) we must therefore use subjects from the
leak-safe ``configs/cohort_ucsfpdgm_n50.yaml`` evaluation cohort, where
DL inference is legitimate test-set inference.

Subject-selection rule (the **S-D rule**): pick the five n=50 cohort
subjects whose mean absolute per-cell residual (averaged across the
~516 (label \u00d7 compartment) cells of
``fig_hitplot_agreement_dl_vs_gt.csv``) is closest to the *cohort*
mean absolute per-cell residual. Ties (extremely unlikely given
floating-point sums over 516 cells) are broken by subject id ascending.
This produces a representative slice of the cohort agreement
distribution rather than a cherry-picked best/worst.

Per-subject sub-panel: top-K=8 regions (ranked by total tumor-overlap
volume averaged across DL and GT) with two horizontal stacked bars per
region --- DL on top (solid, NCR/ED/ET stacked left-to-right) and GT
underneath (hatched outlines of the same compartments). When DL = GT
the two bars are visually congruent; gaps and overshoots are the
per-subject DL-vs-GT residual.

Usage::

    uv run python scripts/build_legacy5_dlvsgt_panel.py \\
        --agreement-csv outputs/figures/fig_hitplot_agreement_dl_vs_gt.csv \\
        --hitplot-root data/derivatives_cohort50 \\
        --top-k 8 \\
        --out-pdf outputs/figures/supp_legacy5_dlvsgt_panel.pdf \\
        --out-png outputs/figures/supp_legacy5_dlvsgt_panel.png \\
        --out-json outputs/figures/supp_legacy5_dlvsgt_panel.json

Outputs:

* PDF + PNG --- the 5\u00d71 supplementary figure.
* JSON --- the deterministic record of the S-D pick (the 5 selected
  subjects, their per-subject mean |residual|, and the cohort mean
  |residual|) so the LaTeX caption quotes the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# scripts/ is not on sys.path when this module is loaded by pytest via
# importlib; ensure the sibling renderer is importable in both contexts
# (direct `python scripts/...` invocation already adds scripts/ to
# sys.path automatically).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from render_per_subject_hitplot_bar import (  # noqa: E402
    CANONICAL_COMPARTMENT_ORDER,
    COMPARTMENT_COLORS,
    aggregate_top_k,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--agreement-csv",
        type=Path,
        default=Path("outputs/figures/fig_hitplot_agreement_dl_vs_gt.csv"),
        help="Per-cell DL-vs-GT residual CSV (from build_hitplot_agreement_panel.py).",
    )
    p.add_argument(
        "--hitplot-root",
        type=Path,
        default=Path("data/derivatives_cohort50"),
        help="Root containing per-subject hitplot/sub-XXXX_hitplot_{dl,gt}.csv.",
    )
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-pdf", type=Path, required=True)
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# S-D subject picker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SDpick:
    """Result of the S-D subject-selection rule."""

    selected: tuple[str, ...]  # 5 subject ids in selection order
    cohort_mean_abs_residual: float  # mm^3 of |overlap_volume_diff|
    per_subject_mean_abs_residual: dict[str, float]


def select_representative_subjects(
    agreement_df: pd.DataFrame,
    *,
    n_pick: int = 5,
) -> SDpick:
    """Pick ``n_pick`` subjects whose mean |residual| is closest to cohort mean.

    The agreement CSV (one row per (subject, label, compartment) cell)
    carries an ``abs_diff_volume_mm3`` column. We compute, per subject,
    the mean of that column over all of the subject's cells; then we
    rank subjects by ``|per_subject_mean - cohort_mean|`` ascending and
    take the first ``n_pick``. Ties are broken by subject id (string
    ascending; deterministic).
    """
    needed = {"subject_id", "abs_diff_volume_mm3"}
    missing = needed - set(agreement_df.columns)
    if missing:
        raise ValueError(f"agreement CSV missing columns: {sorted(missing)}")

    per_subject = agreement_df.groupby("subject_id")["abs_diff_volume_mm3"].mean().sort_index()
    cohort_mean = float(per_subject.mean())

    distance = (per_subject - cohort_mean).abs().rename("dist")
    ranking = pd.DataFrame({"dist": distance, "sid": per_subject.index.astype(str)})
    ranking = ranking.sort_values(by=["dist", "sid"], ascending=[True, True])
    selected = tuple(str(s) for s in ranking.head(n_pick)["sid"].tolist())

    per_subject_dict = {str(k): float(v) for k, v in per_subject.items()}
    return SDpick(
        selected=selected,
        cohort_mean_abs_residual=cohort_mean,
        per_subject_mean_abs_residual=per_subject_dict,
    )


# ---------------------------------------------------------------------------
# Per-subject sub-panel rendering
# ---------------------------------------------------------------------------


def _aggregate_subject(
    hitplot_root: Path,
    subject_id: str,
    *,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dl_top_k, gt_top_k) wide-format dataframes aligned by region.

    The region set is ranked by mean tumour-overlap volume across the
    DL and GT views, so a region present in DL but absent in GT (or
    vice versa) still appears with a zero-length bar on the missing
    side.  Both returned dataframes are reindexed to the same
    ``MultiIndex``-free row label list, ordered top-to-bottom.
    """
    sub_dir = hitplot_root / f"sub-{subject_id}" / "hitplot"
    dl_csv = sub_dir / f"sub-{subject_id}_hitplot_dl.csv"
    gt_csv = sub_dir / f"sub-{subject_id}_hitplot_gt.csv"
    if not dl_csv.is_file():
        raise FileNotFoundError(dl_csv)
    if not gt_csv.is_file():
        raise FileNotFoundError(gt_csv)

    # Aggregate independently to NCR/ED/ET disjoint compartments.
    dl_full = aggregate_top_k(pd.read_csv(dl_csv), k=10_000)  # all non-zero
    gt_full = aggregate_top_k(pd.read_csv(gt_csv), k=10_000)

    # Union of region names, ranked by max(DL, GT) total mm^3 so a
    # region only seen by GT or only seen by DL still reaches the
    # leaderboard if it is large enough.
    regions = sorted(
        set(dl_full.index) | set(gt_full.index),
        key=lambda r: -max(
            dl_full.loc[r].sum() if r in dl_full.index else 0.0,
            gt_full.loc[r].sum() if r in gt_full.index else 0.0,
        ),
    )[:k]
    dl_top = dl_full.reindex(
        index=regions, columns=list(CANONICAL_COMPARTMENT_ORDER), fill_value=0.0
    )
    gt_top = gt_full.reindex(
        index=regions, columns=list(CANONICAL_COMPARTMENT_ORDER), fill_value=0.0
    )
    return dl_top, gt_top


def _draw_subject_subpanel(
    ax: plt.Axes,
    dl_top: pd.DataFrame,
    gt_top: pd.DataFrame,
    *,
    subject_id: str,
    per_subject_mean_residual: float,
) -> float:
    """Render one subject's sub-panel onto ``ax``; return the max x-extent.

    Each region row has *two* horizontal bars: DL on top (solid colours,
    edge black) and GT below (hatched, edge black). When DL = GT
    perfectly the two bars are visually identical apart from the hatch
    pattern.
    """
    n_rows = len(dl_top)
    if n_rows == 0:
        ax.text(0.5, 0.5, f"sub-{subject_id}: no non-zero regions", ha="center", va="center")
        ax.axis("off")
        return 1.0

    # Bar geometry. Each row spans 1 unit on the y-axis; we put the
    # DL bar at y = i + 0.20 and the GT bar at y = i - 0.20 (so a row
    # is fully populated within [i - 0.40, i + 0.40] and the next row
    # starts at i + 1.0 with comfortable padding).
    bar_height = 0.36

    region_names = list(dl_top.index)
    # Display top-1 at the top of the sub-panel.
    region_names = region_names[::-1]
    dl_disp = dl_top.iloc[::-1]
    gt_disp = gt_top.iloc[::-1]

    max_x = 0.0
    for i, region in enumerate(region_names):
        # DL bar (top half), solid colours stacked NCR -> ED -> ET.
        left = 0.0
        for comp in CANONICAL_COMPARTMENT_ORDER:
            v_ml = float(dl_disp.loc[region, comp]) / 1000.0
            if v_ml > 0:
                ax.barh(
                    i + 0.20,
                    v_ml,
                    left=left,
                    height=bar_height,
                    color=COMPARTMENT_COLORS[comp],
                    edgecolor="black",
                    linewidth=0.4,
                )
            left += v_ml
        max_x = max(max_x, left)

        # GT bar (bottom half), same colours but hatched and slightly transparent.
        left = 0.0
        for comp in CANONICAL_COMPARTMENT_ORDER:
            v_ml = float(gt_disp.loc[region, comp]) / 1000.0
            if v_ml > 0:
                ax.barh(
                    i - 0.20,
                    v_ml,
                    left=left,
                    height=bar_height,
                    color=COMPARTMENT_COLORS[comp],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.55,
                    hatch="///",
                )
            left += v_ml
        max_x = max(max_x, left)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(region_names, fontsize=7)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_xlim(0, max_x * 1.08 if max_x > 0 else 1.0)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        f"sub-{subject_id}  (mean |residual| = {per_subject_mean_residual:.1f} mm\u00b3)",
        fontsize=9,
        loc="left",
    )
    return max_x


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


def render_panel(
    pick: SDpick,
    hitplot_root: Path,
    *,
    top_k: int,
) -> plt.Figure:
    """Compose the 5\u00d71 supplementary figure from the S-D pick."""
    fig, axes = plt.subplots(
        nrows=len(pick.selected),
        ncols=1,
        figsize=(8.4, 2.6 * len(pick.selected)),
        sharex=False,
    )
    if len(pick.selected) == 1:
        axes = [axes]

    for ax, sid in zip(axes, pick.selected, strict=False):
        dl_top, gt_top = _aggregate_subject(hitplot_root, sid, k=top_k)
        _draw_subject_subpanel(
            ax,
            dl_top,
            gt_top,
            subject_id=sid,
            per_subject_mean_residual=pick.per_subject_mean_abs_residual[sid],
        )

    # X-label on the bottom subplot only.
    axes[-1].set_xlabel("Tumor overlap volume per region (mL)", fontsize=10)

    # Single shared legend at the top: explains both the colour key
    # (NCR/ED/ET) and the DL-vs-GT encoding (solid vs hatched).
    handles = []
    for comp in CANONICAL_COMPARTMENT_ORDER:
        handles.append(
            mpatches.Patch(
                facecolor=COMPARTMENT_COLORS[comp], edgecolor="black", linewidth=0.4, label=comp
            ),
        )
    handles.append(
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.4, label="DL (top, solid)"
        ),
    )
    handles.append(
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            hatch="///",
            label="GT (bottom, hatched)",
        ),
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.005),
    )

    fig.suptitle(
        "Per-subject DL-vs-GT Hit-Plot agreement (S-D rule, n=5 representative cohort subjects)",
        fontsize=11,
        y=1.022,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        sys.argv = ["build_legacy5_dlvsgt_panel.py", *argv]
    args = _parse_args()

    if not args.agreement_csv.is_file():
        print(f"[error] agreement CSV not found: {args.agreement_csv}")
        return 2

    # Force ``subject_id`` to string so 4-digit ids like "0005" survive
    # the round-trip (pandas would otherwise read "0005" as integer 5
    # and downstream sub-XXXX path joins would break with "sub-5").
    agreement_df = pd.read_csv(args.agreement_csv, dtype={"subject_id": str})
    pick = select_representative_subjects(agreement_df, n_pick=5)

    fig = render_panel(pick, args.hitplot_root, top_k=args.top_k)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(args.out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    args.out_json.write_text(
        json.dumps(
            {
                "selected_subjects": list(pick.selected),
                "selection_rule": "S-D: per-subject mean |residual| closest to cohort mean",
                "cohort_mean_abs_residual_mm3": pick.cohort_mean_abs_residual,
                "per_subject_mean_abs_residual_mm3": pick.per_subject_mean_abs_residual,
                "top_k": args.top_k,
                "agreement_csv": str(args.agreement_csv),
            },
            indent=2,
            sort_keys=False,
        ),
    )

    print("=" * 84)
    print(f"PR-PHASEC  supp DL-vs-GT panel  selected={list(pick.selected)}")
    print(f"   cohort mean |residual| = {pick.cohort_mean_abs_residual:.2f} mm\u00b3")
    for sid in pick.selected:
        print(
            f"   sub-{sid}  mean |residual| = "
            f"{pick.per_subject_mean_abs_residual[sid]:.2f} mm\u00b3"
        )
    print("=" * 84)
    print(f"  wrote {args.out_pdf}")
    print(f"  wrote {args.out_png}")
    print(f"  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
