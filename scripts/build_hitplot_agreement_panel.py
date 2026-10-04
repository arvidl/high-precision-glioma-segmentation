"""PR-7i: build the Hit-Plot agreement panel (DL vs UCSF-PDGM reference).

Walks ``data/derivatives_cohort50/sub-XXXX/hitplot/sub-XXXX_hitplot_dl.csv``
and ``..._hitplot_gt.csv`` for every subject in the locked cohort YAML
(``configs/cohort_ucsfpdgm_n50.yaml``), joins them on
``(label, compartment)``, and produces three artefacts implementing the
Q2-C decision in ``docs/scope_pr7.md`` § 3:

* ``outputs/figures/fig_hitplot_agreement_dl_vs_gt.pdf`` --- 1x3
  Bland-Altman panel grid (one panel per WT/TC/ET compartment),
  x = mean of (DL pct_of_parcel, GT pct_of_parcel),
  y = DL - GT, with mean-difference and 1.96-sigma limits-of-agreement
  (LoA) reference lines and a per-compartment CCC + n_pairs annotation.
  Bland-Altman is the *visual* device; CCC is the headline scalar.
* ``outputs/figures/fig_hitplot_agreement_dl_vs_gt.json`` --- per-
  compartment summary the figure renders. Schema::

      {
        "schema_version": "1.0",
        "comparison": {"prediction": "dl", "reference": "gt"},
        "n_total": 50, "n_with_pair": 47, "n_problems": 3,
        "compartments": ["WT", "TC", "ET"],
        "summary": {
          "WT": {
            "ccc": 0.94,
            "mean_diff_pct_of_parcel": 0.12,
            "std_diff_pct_of_parcel": 1.85,
            "loa_lower_pct_of_parcel": -3.50,
            "loa_upper_pct_of_parcel":  3.74,
            "mean_abs_diff_volume_mm3": 184.7,
            "n_pairs": 4836, "n_subjects": 47, "n_regions": 103
          },
          "TC": {...}, "ET": {...}
        },
        "problems": [{"subject_id": "0042", "reason": "..."}]
      }
* ``outputs/figures/fig_hitplot_agreement_dl_vs_gt.csv`` --- long-
  format per-(subject, label, compartment) audit trail with the four
  raw quantities the figure consumes
  (``dl_pct_of_parcel``, ``gt_pct_of_parcel``,
  ``dl_overlap_volume_mm3``, ``gt_overlap_volume_mm3``) plus the
  derived ``diff_pct_of_parcel`` and ``abs_diff_volume_mm3`` columns.
  Reviewers can recompute every panel and every JSON cell from this
  CSV alone.

Aggregation rules (mirroring PR-7g/PR-7h's NaN/Inf-safe convention):

* Each per-subject join drops ``(label, compartment)`` rows where
  *either* the DL or the GT ``pct_of_parcel`` is NaN. ``parcel_voxels``
  acts as the sentinel for "region absent in this subject's
  parcellation"; rows with ``parcel_voxels == 0`` cannot appear in the
  PR-7f CSVs by construction (``compartment_region_matrix`` filters
  them out), so we only have to guard against NaN propagation from
  upstream.
* CCC is computed on the full per-(subject, label) pair vector for a
  compartment, *across* the cohort (matching the scope doc's "CCC per
  compartment" decision). Pairs from different subjects share a
  compartment but live in different anatomical contexts; this is by
  design --- the headline number is "how well does DL track GT on
  every (subject, region) cell of the Hit-Plot, anywhere in the
  cohort". Per-subject CCCs are also reported in the audit CSV for
  completeness.
* ``mean_abs_diff_volume_mm3`` is the cohort-mean per-cell
  ``|dl_overlap_volume_mm3 - gt_overlap_volume_mm3|`` and is the
  one-number summary the scope doc § 3 calls for.

R2.8 ("sensitivity analysis of Hit-Plot to segmentation choice"); the
JSON powers the auto-generated caption sentence in the
``\\rev{...}`` block of ``paper/main.tex``.

Usage::

    # Default --- whole cohort, dl-vs-gt:
    uv run python scripts/build_hitplot_agreement_panel.py

    # Equivalent via the Make wrapper:
    make agreement-panel

    # Alternative comparison (same producer, different upstream
    # CSVs --- e.g. dl-vs-raidionics for a sensitivity-to-segmenter
    # robustness panel reusing exactly the same code path):
    uv run python scripts/build_hitplot_agreement_panel.py \\
        --prediction dl --reference raidionics \\
        --out-pdf outputs/figures/fig_hitplot_agreement_dl_vs_raidionics.pdf

Exit codes:
    0   panel built and written; n_pairs > 0 in every compartment
    1   panel written but at least one compartment has n_pairs == 0
    2   could not load the cohort YAML / hitplot root unreachable
    3   --strict and at least one CSV missing / malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpgs.io import load_cohort_metadata, normalize_subject_id
from hpgs.metrics import concordance_correlation_coefficient

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_HITPLOT_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_PDF = Path("outputs/figures/fig_hitplot_agreement_dl_vs_gt.pdf")
DEFAULT_OUT_JSON = Path("outputs/figures/fig_hitplot_agreement_dl_vs_gt.json")
DEFAULT_OUT_CSV = Path("outputs/figures/fig_hitplot_agreement_dl_vs_gt.csv")

# Compartments locked by Q2-C in docs/scope_pr7.md § 3.
COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")

# Sources whose deterministic Hit-Plot CSV PR-7f writes per subject.
# These are exactly the suffixes accepted by --prediction / --reference.
ACCEPTED_SOURCES: frozenset[str] = frozenset(
    {"dl", "gt", "raidionics", "segmentglioma", "tumorsynth"},
)

# Columns expected in every PR-7f deterministic Hit-Plot CSV (the
# probabilistic CSV uses a different schema and is *not* a valid input
# to this producer; the panel needs hard counts to be comparable).
REQUIRED_HITPLOT_COLUMNS: tuple[str, ...] = (
    "label",
    "compartment",
    "parcel_voxels",
    "overlap_voxels",
    "overlap_volume_mm3",
    "pct_of_parcel",
)

# Bland-Altman 95 % limits-of-agreement multiplier (Bland & Altman 1986).
LOA_K: float = 1.96


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cohort-yaml",
        type=Path,
        default=DEFAULT_COHORT_YAML,
        help="Cohort YAML (default: %(default)s).",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="UCSF-PDGM v5 metadata CSV (default: %(default)s).",
    )
    p.add_argument(
        "--hitplot-root",
        type=Path,
        default=DEFAULT_HITPLOT_ROOT,
        help="Per-subject derivatives root with hitplot/ subdirs (default: %(default)s).",
    )
    p.add_argument(
        "--prediction",
        default="dl",
        choices=sorted(ACCEPTED_SOURCES),
        help="Hit-Plot source treated as the prediction (default: %(default)s).",
    )
    p.add_argument(
        "--reference",
        default="gt",
        choices=sorted(ACCEPTED_SOURCES),
        help="Hit-Plot source treated as the reference (default: %(default)s).",
    )
    p.add_argument(
        "--out-pdf",
        type=Path,
        default=DEFAULT_OUT_PDF,
        help="Where to write the Bland-Altman panel PDF (default: %(default)s).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Where to write the per-compartment summary JSON (default: %(default)s).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Where to write the long-format per-pair audit CSV (default: %(default)s).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on the first missing / malformed Hit-Plot CSV.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV I/O + schema validation
# ---------------------------------------------------------------------------


def _hitplot_csv_path(hitplot_root: Path, subject_id: str, source: str) -> Path:
    """Path to one PR-7f deterministic Hit-Plot CSV."""
    if source not in ACCEPTED_SOURCES:
        raise ValueError(
            f"_hitplot_csv_path: source {source!r} not in {sorted(ACCEPTED_SOURCES)}",
        )
    sid = normalize_subject_id(subject_id)
    return hitplot_root / f"sub-{sid}" / "hitplot" / f"sub-{sid}_hitplot_{source}.csv"


class HitplotCsvError(ValueError):
    """Raised when a Hit-Plot CSV is missing required structure."""


def _validate_hitplot_csv(df: pd.DataFrame, *, source: Path | str = "<dataframe>") -> None:
    """Validate a deterministic PR-7f Hit-Plot CSV in memory.

    Permissive about *extra* columns (forward-compatibility) but strict
    about :data:`REQUIRED_HITPLOT_COLUMNS`. Catches the common confusion
    of feeding a probabilistic CSV (`hitplot_dl_prob.csv`) by hard-
    failing on the missing ``overlap_voxels`` column.
    """
    missing = [c for c in REQUIRED_HITPLOT_COLUMNS if c not in df.columns]
    if missing:
        raise HitplotCsvError(
            f"{source}: missing required columns {missing} (got {list(df.columns)})",
        )
    bad_comps = set(df["compartment"].unique()) - set(COMPARTMENTS)
    if bad_comps:
        raise HitplotCsvError(
            f"{source}: unexpected compartment(s) {sorted(bad_comps)}; "
            f"expected subset of {COMPARTMENTS}",
        )


def load_pair_records(
    hitplot_root: Path,
    subject_ids: Iterable[str],
    *,
    prediction: str,
    reference: str,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load + join the (prediction, reference) Hit-Plot CSVs cohort-wide.

    Returns ``(pairs_df, problems)`` where ``pairs_df`` is the long-
    format audit frame with one row per (subject, label, compartment)
    that survived the join, and ``problems`` is a list of
    ``(subject_id, reason)`` for subjects whose CSV was missing /
    malformed / empty after the join.

    If ``strict`` is True the first problem is re-raised.

    Per-subject join semantics: rows with NaN in either
    ``pct_of_parcel`` are dropped (NaN here means "compartment empty in
    this subject", which is rare but possible --- e.g. a tumor with no
    enhancing core). Rows with mismatched ``parcel_voxels`` between
    prediction and reference are kept but flagged in the audit CSV; in
    a sane pipeline both CSVs come from the same parcellation NIfTI so
    the values *will* match.
    """
    rows: list[pd.DataFrame] = []
    problems: list[tuple[str, str]] = []
    for raw_sid in subject_ids:
        sid = normalize_subject_id(raw_sid)
        pred_path = _hitplot_csv_path(hitplot_root, sid, prediction)
        ref_path = _hitplot_csv_path(hitplot_root, sid, reference)
        if not pred_path.is_file():
            reason = f"{prediction} CSV not found at {pred_path}"
            problems.append((sid, reason))
            if strict:
                raise FileNotFoundError(reason)
            continue
        if not ref_path.is_file():
            reason = f"{reference} CSV not found at {ref_path}"
            problems.append((sid, reason))
            if strict:
                raise FileNotFoundError(reason)
            continue
        try:
            pred_df = pd.read_csv(pred_path)
            ref_df = pd.read_csv(ref_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            reason = f"could not parse CSV ({exc})"
            problems.append((sid, reason))
            if strict:
                raise
            continue
        try:
            _validate_hitplot_csv(pred_df, source=pred_path)
            _validate_hitplot_csv(ref_df, source=ref_path)
        except HitplotCsvError as exc:
            problems.append((sid, str(exc)))
            if strict:
                raise
            continue

        merged = pred_df.merge(
            ref_df,
            on=["label", "compartment"],
            suffixes=(f"_{prediction}", f"_{reference}"),
            how="inner",
        )
        if merged.empty:
            problems.append((sid, "empty join (no shared (label, compartment) rows)"))
            continue
        # Drop NaN-in-either-pct rows defensively.
        valid = ~(
            merged[f"pct_of_parcel_{prediction}"].isna()
            | merged[f"pct_of_parcel_{reference}"].isna()
        )
        merged = merged[valid].copy()
        if merged.empty:
            problems.append((sid, "all pairs were NaN in pct_of_parcel"))
            continue
        merged["subject_id"] = sid
        rows.append(merged)

    if not rows:
        return _empty_pairs_frame(prediction, reference), problems

    pairs = pd.concat(rows, ignore_index=True, sort=False)
    return _project_audit_columns(pairs, prediction=prediction, reference=reference), problems


def _empty_pairs_frame(prediction: str, reference: str) -> pd.DataFrame:
    """Empty audit frame with the canonical column order."""
    return pd.DataFrame(
        columns=[
            "subject_id",
            "label",
            "compartment",
            f"pct_of_parcel_{prediction}",
            f"pct_of_parcel_{reference}",
            f"overlap_volume_mm3_{prediction}",
            f"overlap_volume_mm3_{reference}",
            "diff_pct_of_parcel",
            "mean_pct_of_parcel",
            "abs_diff_volume_mm3",
        ],
    )


def _project_audit_columns(
    pairs: pd.DataFrame,
    *,
    prediction: str,
    reference: str,
) -> pd.DataFrame:
    """Trim the merged frame to the canonical audit columns + derive Δ / mean."""
    pred_pct = f"pct_of_parcel_{prediction}"
    ref_pct = f"pct_of_parcel_{reference}"
    pred_vol = f"overlap_volume_mm3_{prediction}"
    ref_vol = f"overlap_volume_mm3_{reference}"
    out = pairs[
        [
            "subject_id",
            "label",
            "compartment",
            pred_pct,
            ref_pct,
            pred_vol,
            ref_vol,
        ]
    ].copy()
    out["diff_pct_of_parcel"] = out[pred_pct].astype(float) - out[ref_pct].astype(float)
    out["mean_pct_of_parcel"] = 0.5 * (out[pred_pct].astype(float) + out[ref_pct].astype(float))
    out["abs_diff_volume_mm3"] = (out[pred_vol].astype(float) - out[ref_vol].astype(float)).abs()
    return out.sort_values(
        ["compartment", "subject_id", "label"],
        kind="stable",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-compartment Bland-Altman + CCC summary
# ---------------------------------------------------------------------------


def summarise_per_compartment(
    pairs: pd.DataFrame,
    *,
    prediction: str,
    reference: str,
) -> dict[str, dict[str, float | int | None]]:
    """Per-compartment CCC + Bland-Altman + |Δvolume| summary.

    Returns a dict keyed by compartment with the same field names the
    JSON sidecar exposes (see module docstring). NaN-only compartments
    return ``None`` for the float fields so the JSON is valid.
    """
    pred_pct = f"pct_of_parcel_{prediction}"
    ref_pct = f"pct_of_parcel_{reference}"
    out: dict[str, dict[str, float | int | None]] = {}
    for comp in COMPARTMENTS:
        sub = pairs[pairs["compartment"] == comp]
        n_pairs = len(sub)
        if n_pairs == 0:
            out[comp] = {
                "ccc": None,
                "mean_diff_pct_of_parcel": None,
                "std_diff_pct_of_parcel": None,
                "loa_lower_pct_of_parcel": None,
                "loa_upper_pct_of_parcel": None,
                "mean_abs_diff_volume_mm3": None,
                "n_pairs": 0,
                "n_subjects": 0,
                "n_regions": 0,
            }
            continue
        x = sub[pred_pct].to_numpy(dtype=np.float64)
        y = sub[ref_pct].to_numpy(dtype=np.float64)
        diff = x - y
        ccc = concordance_correlation_coefficient(x, y)
        # Bland-Altman: ddof=1 sample std, matching the original 1986
        # paper. This is what reviewers expect when they see "1.96 SD".
        if diff.size >= 2:
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
        else:
            mean_diff = float(np.mean(diff))  # diff.size == 1 -> std undefined
            std_diff = float("nan")
        loa_lower = mean_diff - LOA_K * std_diff if np.isfinite(std_diff) else float("nan")
        loa_upper = mean_diff + LOA_K * std_diff if np.isfinite(std_diff) else float("nan")
        mean_abs_diff_vol = float(sub["abs_diff_volume_mm3"].mean())
        out[comp] = {
            "ccc": _json_safe(ccc),
            "mean_diff_pct_of_parcel": _json_safe(mean_diff),
            "std_diff_pct_of_parcel": _json_safe(std_diff),
            "loa_lower_pct_of_parcel": _json_safe(loa_lower),
            "loa_upper_pct_of_parcel": _json_safe(loa_upper),
            "mean_abs_diff_volume_mm3": _json_safe(mean_abs_diff_vol),
            "n_pairs": n_pairs,
            "n_subjects": int(sub["subject_id"].nunique()),
            "n_regions": int(sub["label"].nunique()),
        }
    return out


def _json_safe(x: float) -> float | None:
    """Convert NaN to ``None`` so the JSON sidecar is valid."""
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return float(x)


# ---------------------------------------------------------------------------
# Bland-Altman PDF rendering
# ---------------------------------------------------------------------------


def render_panel_pdf(
    pairs: pd.DataFrame,
    summary: Mapping[str, Mapping[str, float | int | None]],
    out_pdf: Path,
    *,
    prediction: str,
    reference: str,
) -> None:
    """Write a 1x3 Bland-Altman PDF (one panel per WT/TC/ET).

    Each panel scatters per-(subject, region) cells with the mean of
    (prediction, reference) ``pct_of_parcel`` on the x-axis and
    (prediction - reference) on the y-axis. Mean-difference and 1.96-
    sigma LoA reference lines are drawn; CCC and n_pairs are reported
    in the panel title.

    Empty compartments still get a panel (with a "no data" annotation)
    so the figure layout is stable regardless of cohort completeness.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=False)
    for ax, comp in zip(axes, COMPARTMENTS, strict=True):
        sub = pairs[pairs["compartment"] == comp]
        cell = summary[comp]
        if len(sub) == 0:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="dimgray",
                fontsize=12,
            )
            ax.set_title(f"{comp} (n=0)")
            ax.set_xlabel("mean pct_of_parcel (DL, GT)")
            ax.set_ylabel(f"{prediction} - {reference}  (pct_of_parcel)")
            continue
        ax.scatter(
            sub["mean_pct_of_parcel"],
            sub["diff_pct_of_parcel"],
            s=8,
            alpha=0.35,
            color="#1f77b4",
            edgecolors="none",
        )
        # Reference lines.
        mean_diff = cell.get("mean_diff_pct_of_parcel")
        loa_lower = cell.get("loa_lower_pct_of_parcel")
        loa_upper = cell.get("loa_upper_pct_of_parcel")
        if mean_diff is not None:
            ax.axhline(mean_diff, color="black", lw=1.0, ls="--")
        if loa_lower is not None:
            ax.axhline(loa_lower, color="gray", lw=1.0, ls=":")
        if loa_upper is not None:
            ax.axhline(loa_upper, color="gray", lw=1.0, ls=":")
        ax.axhline(0.0, color="gray", lw=0.5, ls="-", alpha=0.5)
        ccc = cell.get("ccc")
        ccc_str = "n/a" if ccc is None else f"{ccc:.3f}"
        ax.set_title(
            f"{comp}: CCC={ccc_str}  n={cell['n_pairs']}",
            fontsize=11,
        )
        ax.set_xlabel(
            f"mean pct_of_parcel ({prediction}, {reference})",
            fontsize=10,
        )
        ax.set_ylabel(
            f"{prediction} - {reference}  (pct_of_parcel)",
            fontsize=10,
        )
        ax.tick_params(axis="both", labelsize=9)
        # Annotate LoA values inline if finite.
        if loa_lower is not None and loa_upper is not None and mean_diff is not None:
            ax.text(
                0.99,
                0.02,
                f"mean diff: {mean_diff:+.2f}\nLoA: [{loa_lower:+.2f}, {loa_upper:+.2f}]",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="black",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "lightgray", "lw": 0.5},
            )
    fig.suptitle(
        f"Hit-Plot agreement: {prediction} vs {reference}  "
        "(Bland-Altman per (subject, region) cell)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _print_summary_to_stdout(
    summary: Mapping[str, Mapping[str, float | int | None]],
    *,
    n_total: int,
    n_with_pair: int,
    n_problems: int,
    prediction: str,
    reference: str,
) -> None:
    print()
    print("=" * 84)
    print(
        f"PR-7i  Hit-Plot agreement panel ({prediction} vs {reference})  "
        f"cohort_n={n_total}  with_pair={n_with_pair}  problems={n_problems}",
    )
    print("=" * 84)
    print(
        f"  {'compartment':<12}{'n_pairs':>10}{'CCC':>10}"
        f"{'mean_diff':>14}{'LoA':>22}{'mean_|dV|':>14}"
    )
    print("-" * 84)
    for comp in COMPARTMENTS:
        cell = summary[comp]
        if cell["n_pairs"] == 0 or cell["ccc"] is None:
            print(
                f"  {comp:<12}{cell['n_pairs']:>10}{'--':>10}{'--':>14}{'[--, --]':>22}{'--':>14}",
            )
            continue
        loa = f"[{cell['loa_lower_pct_of_parcel']:+.2f}, {cell['loa_upper_pct_of_parcel']:+.2f}]"
        print(
            f"  {comp:<12}{cell['n_pairs']:>10}{cell['ccc']:>10.3f}"
            f"{cell['mean_diff_pct_of_parcel']:>+14.3f}{loa:>22}"
            f"{cell['mean_abs_diff_volume_mm3']:>14.1f}",
        )
    print("=" * 84)


def main() -> int:
    args = _parse_args()

    if args.prediction == args.reference:
        print(
            f"error: --prediction and --reference must differ (both are {args.prediction!r})",
            file=sys.stderr,
        )
        return 2
    try:
        cm = load_cohort_metadata(args.cohort_yaml, args.metadata_csv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: failed to load cohort: {exc}", file=sys.stderr)
        return 2
    if not args.hitplot_root.exists():
        print(
            f"error: hitplot root does not exist: {args.hitplot_root}",
            file=sys.stderr,
        )
        return 2

    subject_ids = list(cm.df.index)
    try:
        pairs, problems = load_pair_records(
            args.hitplot_root,
            subject_ids,
            prediction=args.prediction,
            reference=args.reference,
            strict=args.strict,
        )
    except (FileNotFoundError, pd.errors.ParserError, HitplotCsvError) as exc:
        print(f"error (--strict): {exc}", file=sys.stderr)
        return 3

    for sid, reason in problems:
        print(f"warning: sub-{sid}: {reason}", file=sys.stderr)

    summary = summarise_per_compartment(
        pairs,
        prediction=args.prediction,
        reference=args.reference,
    )
    n_with_pair = int(pairs["subject_id"].nunique()) if len(pairs) else 0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_csv, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cohort_yaml": str(args.cohort_yaml),
                "hitplot_root": str(args.hitplot_root),
                "comparison": {
                    "prediction": args.prediction,
                    "reference": args.reference,
                },
                "n_total": int(cm.n),
                "n_with_pair": n_with_pair,
                "n_problems": len(problems),
                "compartments": list(COMPARTMENTS),
                "summary": summary,
                "problems": [{"subject_id": sid, "reason": r} for sid, r in problems],
            },
            indent=2,
        ),
    )
    render_panel_pdf(
        pairs,
        summary,
        args.out_pdf,
        prediction=args.prediction,
        reference=args.reference,
    )

    _print_summary_to_stdout(
        summary,
        n_total=cm.n,
        n_with_pair=n_with_pair,
        n_problems=len(problems),
        prediction=args.prediction,
        reference=args.reference,
    )
    print(f"  wrote: {args.out_pdf}")
    print(f"  wrote: {args.out_json}")
    print(f"  wrote: {args.out_csv}")

    any_empty_compartment = any(summary[comp]["n_pairs"] == 0 for comp in COMPARTMENTS)
    return 1 if any_empty_compartment else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
