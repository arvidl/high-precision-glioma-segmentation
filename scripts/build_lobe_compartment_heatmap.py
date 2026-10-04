"""PR-7l: build the cohort lobe x compartment glioma-distribution heatmap.

Cohort-level successor to the n=1 ``UCSF-PDGM-0085`` Hit-Plot sunburst
that ships as Figure 15 of the original manuscript. Walks every
subject in the locked cohort YAML
(``configs/cohort_ucsfpdgm_n50.yaml``), reads each subject's
PR-7f deterministic Hit-Plot CSV
(``data/derivatives_cohort50/sub-XXXX/hitplot/sub-XXXX_hitplot_<source>.csv``)
and the matching parcellation LUT
(``..._wmparc_lut.json``), bins the Desikan-Killiany aparc / wmparc
labels into the 11 anatomical lobes defined in
:mod:`hpgs.parcellate.lobes`, and writes three artefacts that replace
the single-subject sunburst with a cohort-level summary the reviewer
can interrogate:

* ``outputs/figures/fig_lobe_compartment_heatmap.pdf`` --- a
  ``len(LOBES) x len(COMPARTMENTS)`` heatmap whose cell value is the
  cohort-median *share of compartment volume* falling in that lobe
  (NB: this normalisation makes each compartment's column sum to ~100 %
  across lobes, so the figure answers "where in the brain does the
  cohort's WT / TC / ET tumor live?"). Each cell is annotated with
  ``"<median> %  (n=K/N)"`` where ``K/N`` is the lobe-prevalence: the
  number of subjects with any compartment voxel intersecting that lobe.
  Rows are ordered rostro-caudal then deep -> infratentorial -> other,
  matching the :data:`hpgs.parcellate.lobes.LOBES` tuple.
* ``outputs/figures/fig_lobe_compartment_heatmap.json`` --- the per-cell
  summary the figure renders. Schema::

      {
        "schema_version": "1.0",
        "cohort_yaml": ".../cohort_ucsfpdgm_n50.yaml",
        "hitplot_root": "data/derivatives_cohort50",
        "source": "gt",
        "n_total": 50, "n_with_csv": 50, "n_problems": 0,
        "lobes": ["frontal", "cingulate", ...],
        "compartments": ["WT", "TC", "ET"],
        "cells": [
          {
            "lobe": "frontal", "compartment": "WT",
            "n_with_overlap": 22, "n_total": 50,
            "prevalence": 0.44,
            "share_of_compartment_pct_median": 18.7,
            "share_of_compartment_pct_p25": 4.1,
            "share_of_compartment_pct_p75": 35.6,
            "share_of_lobe_pct_median_when_present": 1.83,
            "share_of_lobe_pct_p25_when_present": 0.72,
            "share_of_lobe_pct_p75_when_present": 4.55,
            "total_overlap_volume_mm3_sum": 412345.0
          },
          ...
        ],
        "problems": [{"subject_id": "0042", "reason": "..."}]
      }
* ``outputs/figures/fig_lobe_compartment_heatmap.csv`` --- long-format
  per-(subject, lobe, compartment) audit trail. One row per
  (subject, lobe, compartment) cell with non-zero compartment voxels
  *anywhere* (zero-overlap rows are written too so the cohort denominator
  is reconstructible). Columns:
  ``subject_id, lobe, compartment, lobe_volume_mm3, overlap_volume_mm3,
  share_of_lobe_pct, share_of_compartment_pct``. Reviewers can
  recompute every cell from this CSV alone.

Aggregation rules:

* Each subject's hitplot CSV is *long-format* with one row per
  ``(label, compartment)``. We bin each row into a lobe via
  :func:`hpgs.parcellate.assign_lobe_from_label` (LUT lookup) and
  collapse to one row per ``(subject, lobe, compartment)`` by summing
  ``overlap_volume_mm3``. Lobe volume per subject is the sum of
  ``parcel_voxels * voxel_volume_mm3`` over labels in that lobe; voxel
  volume is reconstructed from the per-row identity
  ``overlap_volume_mm3 / overlap_voxels`` (constant per subject for a
  given parcellation).
* ``share_of_compartment_pct`` per (subject, lobe, compartment) =
  100 * overlap_volume_mm3(lobe) / total_overlap_volume_mm3(compartment),
  where the denominator is the sum of overlap across *all* lobes for
  that subject + compartment. Subjects with zero compartment voxels
  contribute zeros to all lobes -> share_of_compartment_pct is set to
  NaN in the audit CSV (and dropped from the cohort summary so the
  median is over subjects with at least one compartment voxel).
* ``share_of_lobe_pct`` per (subject, lobe, compartment) =
  100 * overlap_volume_mm3 / lobe_volume_mm3. Subjects with zero lobe
  volume (rare; only happens if SynthSeg failed to label any voxel as
  that lobe) get NaN.
* Cohort summary: for each (lobe, compartment) cell,
  ``share_of_compartment_pct_*`` quantiles are computed across all
  subjects whose compartment was non-empty (zeros included --
  so the median answers the "where does the cohort's compartment live"
  question even for absent lobes). ``share_of_lobe_pct_*_when_present``
  is conditional on ``overlap_volume_mm3 > 0`` (median magnitude when
  the lobe is touched). ``prevalence`` is the unconditional share of
  the cohort with any overlap in that cell.

Figure 15 (UCSF-PDGM-0085 Anatomical-Lobe Profile sunburst) with a
cohort-level distribution summary. It does *not* introduce a new metric
on top of the locked Q2-C bundle (Dice / HD95 / |VE| / sens / spec) ---
it is a *descriptive* figure of the n=50 cohort's tumor footprint in
the same Desikan-Killiany / wmparc reference frame the Hit-Plot family
already uses, so reviewers can interrogate the cohort's anatomical
representativeness without relying on a single-subject anecdote.

Usage::

    # Default --- whole cohort, gt source:
    uv run python scripts/build_lobe_compartment_heatmap.py

    # Equivalent via the Make wrapper:
    make lobe-heatmap

    # Use the DL prediction instead of the GT reference (sensitivity check):
    uv run python scripts/build_lobe_compartment_heatmap.py --source dl \\
        --out-pdf outputs/figures/fig_lobe_compartment_heatmap_dl.pdf \\
        --out-json outputs/figures/fig_lobe_compartment_heatmap_dl.json \\
        --out-csv outputs/figures/fig_lobe_compartment_heatmap_dl.csv

Exit codes:
    0   heatmap built and written; every (lobe, compartment) cell has
        a finite cohort denominator
    1   heatmap written but at least one compartment had zero subjects
        with non-empty overlap (cohort denominator empty for that column)
    2   could not load the cohort YAML / hitplot root unreachable
    3   --strict and at least one CSV / LUT missing or malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpgs.io import load_cohort_metadata, normalize_subject_id
from hpgs.parcellate.lobes import LOBES, assign_lobe_from_label

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_HITPLOT_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_PDF = Path("outputs/figures/fig_lobe_compartment_heatmap.pdf")
DEFAULT_OUT_JSON = Path("outputs/figures/fig_lobe_compartment_heatmap.json")
DEFAULT_OUT_CSV = Path("outputs/figures/fig_lobe_compartment_heatmap.csv")

# Compartments locked by Q2-C in docs/scope_pr7.md § 3 (BraTS-3 framing).
COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")

# Hitplot source CSV suffixes that PR-7f writes per subject.
ACCEPTED_SOURCES: frozenset[str] = frozenset(
    {"dl", "gt", "raidionics", "segmentglioma"},
)

REQUIRED_HITPLOT_COLUMNS: tuple[str, ...] = (
    "label",
    "compartment",
    "parcel_voxels",
    "overlap_voxels",
    "overlap_volume_mm3",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-yaml", type=Path, default=DEFAULT_COHORT_YAML)
    p.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--hitplot-root", type=Path, default=DEFAULT_HITPLOT_ROOT)
    p.add_argument(
        "--source",
        default="gt",
        choices=sorted(ACCEPTED_SOURCES),
        help="Hit-Plot source CSV suffix to bin (default: %(default)s).",
    )
    p.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on the first missing / malformed CSV or LUT.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers (CSV + LUT path resolution)
# ---------------------------------------------------------------------------


class HitplotCsvError(ValueError):
    """Raised when a Hit-Plot CSV is missing required structure."""


class LutError(ValueError):
    """Raised when a wmparc LUT JSON is missing or malformed."""


def _hitplot_csv_path(hitplot_root: Path, subject_id: str, source: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return hitplot_root / f"sub-{sid}" / "hitplot" / f"sub-{sid}_hitplot_{source}.csv"


def _lut_json_path(hitplot_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return hitplot_root / f"sub-{sid}" / "parcellation" / f"sub-{sid}_wmparc_lut.json"


def _validate_hitplot_csv(df: pd.DataFrame, *, source: Path | str) -> None:
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


def _load_lut(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise LutError(f"{path}: wmparc LUT JSON not found")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LutError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, Mapping):
        raise LutError(f"{path}: LUT root is not an object")
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError) as exc:
            raise LutError(f"{path}: bad label key {k!r} ({exc})") from exc
    return out


# ---------------------------------------------------------------------------
# Per-subject lobe x compartment aggregation
# ---------------------------------------------------------------------------


def aggregate_subject(
    subject_id: str,
    hitplot_df: pd.DataFrame,
    lut: Mapping[int, str],
) -> pd.DataFrame:
    """Collapse a per-(label, compartment) Hit-Plot CSV to per-(lobe, compartment).

    Returns a long-format frame with one row per
    ``(subject_id, lobe, compartment)`` cell -- including zero-overlap
    cells -- carrying the columns:

    ``subject_id, lobe, compartment, lobe_volume_mm3,
    overlap_volume_mm3, share_of_lobe_pct, share_of_compartment_pct``.

    Per-subject voxel volume is reconstructed from the constant ratio
    ``overlap_volume_mm3 / overlap_voxels`` across rows where
    ``overlap_voxels > 0`` (the parcellation grid is constant per
    subject, so this ratio is unique). Falls back to 1.0 mm^3 if no
    such row exists, with the share_of_lobe_pct column then becoming a
    voxel-fraction rather than a volume-fraction (still monotone, but
    flagged via NaN if the cohort denominator is too sparse).
    """
    sid = normalize_subject_id(subject_id)
    df = hitplot_df.copy()

    # Reconstruct voxel volume (mm^3) from any row with overlap > 0.
    nonzero = df[(df["overlap_voxels"].astype(float) > 0)]
    if len(nonzero):
        ratios = nonzero["overlap_volume_mm3"].astype(float) / nonzero["overlap_voxels"].astype(
            float
        )
        voxel_volume_mm3 = float(np.nanmedian(ratios.to_numpy()))
        if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
            voxel_volume_mm3 = 1.0
    else:
        voxel_volume_mm3 = 1.0

    # Bin every row into a lobe via the LUT lookup.
    df["lobe"] = (
        df["label"]
        .astype(int)
        .map(
            lambda lid: assign_lobe_from_label(lid, lut),
        )
    )
    df["overlap_volume_mm3"] = df["overlap_volume_mm3"].astype(float).fillna(0.0)

    # Per (label, compartment), parcel_voxels is constant across compartments
    # of the same label, so to compute lobe volume we should not double-count.
    # Drop duplicates on (label) before summing parcel_voxels per lobe.
    label_volumes = (
        df.drop_duplicates(subset=["label"])[["label", "parcel_voxels", "lobe"]]
        .groupby("lobe", as_index=False)["parcel_voxels"]
        .sum()
        .rename(columns={"parcel_voxels": "lobe_voxels"})
    )
    label_volumes["lobe_volume_mm3"] = label_volumes["lobe_voxels"].astype(float) * voxel_volume_mm3

    # Sum overlap per (lobe, compartment).
    overlap_per_cell = df.groupby(["lobe", "compartment"], as_index=False)[
        "overlap_volume_mm3"
    ].sum()

    # Cartesian (lobe, compartment) so missing cells become explicit zeros.
    grid = pd.MultiIndex.from_product(
        [LOBES, COMPARTMENTS],
        names=["lobe", "compartment"],
    ).to_frame(index=False)
    cells = grid.merge(overlap_per_cell, on=["lobe", "compartment"], how="left")
    cells["overlap_volume_mm3"] = cells["overlap_volume_mm3"].fillna(0.0)

    # Per-compartment total volume (denominator for share_of_compartment_pct).
    comp_total = (
        cells.groupby("compartment", as_index=False)["overlap_volume_mm3"]
        .sum()
        .rename(columns={"overlap_volume_mm3": "compartment_total_mm3"})
    )
    cells = cells.merge(comp_total, on="compartment", how="left")
    cells = cells.merge(
        label_volumes[["lobe", "lobe_volume_mm3"]],
        on="lobe",
        how="left",
    )
    cells["lobe_volume_mm3"] = cells["lobe_volume_mm3"].fillna(0.0)

    # Compute the two shares with NaN where the denominator is zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        cells["share_of_lobe_pct"] = np.where(
            cells["lobe_volume_mm3"].to_numpy() > 0,
            100.0
            * cells["overlap_volume_mm3"].to_numpy()
            / np.maximum(cells["lobe_volume_mm3"].to_numpy(), 1e-12),
            np.nan,
        )
        cells["share_of_compartment_pct"] = np.where(
            cells["compartment_total_mm3"].to_numpy() > 0,
            100.0
            * cells["overlap_volume_mm3"].to_numpy()
            / np.maximum(cells["compartment_total_mm3"].to_numpy(), 1e-12),
            np.nan,
        )

    cells["subject_id"] = sid
    return cells[
        [
            "subject_id",
            "lobe",
            "compartment",
            "lobe_volume_mm3",
            "overlap_volume_mm3",
            "share_of_lobe_pct",
            "share_of_compartment_pct",
        ]
    ]


def load_cohort_records(
    hitplot_root: Path,
    subject_ids: Iterable[str],
    *,
    source: str,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Walk the cohort, return the long-format audit frame + problem list."""
    rows: list[pd.DataFrame] = []
    problems: list[tuple[str, str]] = []
    for raw_sid in subject_ids:
        sid = normalize_subject_id(raw_sid)
        csv_path = _hitplot_csv_path(hitplot_root, sid, source)
        lut_path = _lut_json_path(hitplot_root, sid)
        if not csv_path.is_file():
            reason = f"hitplot CSV not found: {csv_path}"
            problems.append((sid, reason))
            if strict:
                raise FileNotFoundError(reason)
            continue
        try:
            df = pd.read_csv(csv_path)
            _validate_hitplot_csv(df, source=csv_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, HitplotCsvError) as exc:
            problems.append((sid, str(exc)))
            if strict:
                raise
            continue
        try:
            lut = _load_lut(lut_path)
        except LutError as exc:
            problems.append((sid, str(exc)))
            if strict:
                raise
            continue
        try:
            cells = aggregate_subject(sid, df, lut)
        except (KeyError, ValueError) as exc:
            problems.append((sid, f"aggregation failed: {exc}"))
            if strict:
                raise
            continue
        rows.append(cells)

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "subject_id",
                "lobe",
                "compartment",
                "lobe_volume_mm3",
                "overlap_volume_mm3",
                "share_of_lobe_pct",
                "share_of_compartment_pct",
            ],
        )
        return empty, problems

    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined, problems


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------


def summarise_cells(
    audit: pd.DataFrame,
    *,
    n_total: int,
) -> list[dict[str, Any]]:
    """Per-(lobe, compartment) cohort summary list.

    Order matches the cartesian product of :data:`LOBES` x
    :data:`COMPARTMENTS` so JSON consumers (and the heatmap renderer)
    have a stable iteration order. ``share_of_compartment_pct_*`` is
    computed across subjects whose compartment was non-empty (any
    overlap > 0 anywhere); ``share_of_lobe_pct_*_when_present`` is
    conditional on ``overlap_volume_mm3 > 0`` for *this* cell.
    """
    out: list[dict[str, Any]] = []
    if audit.empty:
        for lobe in LOBES:
            for comp in COMPARTMENTS:
                out.append(
                    {
                        "lobe": lobe,
                        "compartment": comp,
                        "n_with_overlap": 0,
                        "n_total": int(n_total),
                        "prevalence": 0.0,
                        "share_of_compartment_pct_median": None,
                        "share_of_compartment_pct_p25": None,
                        "share_of_compartment_pct_p75": None,
                        "share_of_lobe_pct_median_when_present": None,
                        "share_of_lobe_pct_p25_when_present": None,
                        "share_of_lobe_pct_p75_when_present": None,
                        "total_overlap_volume_mm3_sum": 0.0,
                    },
                )
        return out

    for lobe in LOBES:
        for comp in COMPARTMENTS:
            sub = audit[(audit["lobe"] == lobe) & (audit["compartment"] == comp)]
            present = sub[sub["overlap_volume_mm3"].astype(float) > 0]
            n_with = int(len(present))
            soc = sub["share_of_compartment_pct"].to_numpy(dtype=float)
            soc_finite = soc[np.isfinite(soc)]
            sol_present = present["share_of_lobe_pct"].to_numpy(dtype=float)
            sol_present = sol_present[np.isfinite(sol_present)]

            cell: dict[str, Any] = {
                "lobe": lobe,
                "compartment": comp,
                "n_with_overlap": n_with,
                "n_total": int(n_total),
                "prevalence": float(n_with) / float(max(n_total, 1)),
                "total_overlap_volume_mm3_sum": float(
                    sub["overlap_volume_mm3"].astype(float).sum(),
                ),
            }
            if soc_finite.size:
                cell["share_of_compartment_pct_median"] = float(np.median(soc_finite))
                cell["share_of_compartment_pct_p25"] = float(np.percentile(soc_finite, 25))
                cell["share_of_compartment_pct_p75"] = float(np.percentile(soc_finite, 75))
            else:
                cell["share_of_compartment_pct_median"] = None
                cell["share_of_compartment_pct_p25"] = None
                cell["share_of_compartment_pct_p75"] = None
            if sol_present.size:
                cell["share_of_lobe_pct_median_when_present"] = float(
                    np.median(sol_present),
                )
                cell["share_of_lobe_pct_p25_when_present"] = float(
                    np.percentile(sol_present, 25),
                )
                cell["share_of_lobe_pct_p75_when_present"] = float(
                    np.percentile(sol_present, 75),
                )
            else:
                cell["share_of_lobe_pct_median_when_present"] = None
                cell["share_of_lobe_pct_p25_when_present"] = None
                cell["share_of_lobe_pct_p75_when_present"] = None
            out.append(cell)
    return out


# ---------------------------------------------------------------------------
# Heatmap PDF rendering
# ---------------------------------------------------------------------------


def render_heatmap_pdf(
    summary: list[dict[str, Any]],
    out_pdf: Path,
    *,
    source: str,
    n_total: int,
) -> None:
    """Render the lobe x compartment heatmap to PDF.

    Cell color = ``share_of_compartment_pct_median`` (cohort-median
    share of the compartment volume falling in that lobe). Cell
    annotation = ``"<median> %  n=K/N"``. Empty cells render as a
    white tile with a dim "n=0/N" annotation so the figure layout is
    stable.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Pull the median field into an [n_lobes, n_compartments] matrix.
    n_lobes = len(LOBES)
    n_comps = len(COMPARTMENTS)
    medians = np.full((n_lobes, n_comps), np.nan, dtype=float)
    annotations = np.full((n_lobes, n_comps), "", dtype=object)
    by_lobe_comp = {(c["lobe"], c["compartment"]): c for c in summary}
    for i, lobe in enumerate(LOBES):
        for j, comp in enumerate(COMPARTMENTS):
            cell = by_lobe_comp.get((lobe, comp))
            if cell is None:
                continue
            med = cell.get("share_of_compartment_pct_median")
            n_with = cell.get("n_with_overlap", 0)
            medians[i, j] = float(med) if med is not None else np.nan
            if med is None:
                annotations[i, j] = f"--\nn={n_with}/{n_total}"
            else:
                annotations[i, j] = f"{med:.1f}%\nn={n_with}/{n_total}"

    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#f5f5f5")
    masked = np.ma.masked_invalid(medians)
    vmax = float(np.nanmax(medians)) if np.any(np.isfinite(medians)) else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_comps))
    ax.set_xticklabels(COMPARTMENTS, fontsize=11)
    ax.set_yticks(np.arange(n_lobes))
    ax.set_yticklabels(LOBES, fontsize=10)
    ax.set_xlabel("BraTS-3 compartment", fontsize=11)
    ax.set_ylabel("Anatomical lobe (Desikan-Killiany / wmparc)", fontsize=11)
    ax.set_title(
        f"Cohort lobe x compartment glioma distribution (n={n_total}, source={source})",
        fontsize=11,
    )

    # Annotate every cell.
    for i in range(n_lobes):
        for j in range(n_comps):
            text = annotations[i, j]
            if not text:
                continue
            value = medians[i, j]
            text_color = "white" if (np.isfinite(value) and value > 0.5 * vmax) else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "cohort-median share of compartment volume in lobe (%)",
        fontsize=10,
    )
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _print_summary_to_stdout(
    summary: list[dict[str, Any]],
    *,
    n_total: int,
    n_with_csv: int,
    n_problems: int,
    source: str,
) -> None:
    print()
    print("=" * 88)
    print(
        f"PR-7l  cohort lobe x compartment heatmap  source={source}  "
        f"cohort_n={n_total}  with_csv={n_with_csv}  problems={n_problems}",
    )
    print("=" * 88)
    print(
        f"  {'lobe':<22}{'compartment':<14}{'n_with':>8}{'prev':>8}"
        f"{'share_med':>14}{'share_iqr':>22}",
    )
    print("-" * 88)
    by_lobe_comp = {(c["lobe"], c["compartment"]): c for c in summary}
    for lobe in LOBES:
        for comp in COMPARTMENTS:
            c = by_lobe_comp[(lobe, comp)]
            med = c["share_of_compartment_pct_median"]
            p25 = c["share_of_compartment_pct_p25"]
            p75 = c["share_of_compartment_pct_p75"]
            if med is None:
                print(
                    f"  {lobe:<22}{comp:<14}{c['n_with_overlap']:>8}"
                    f"{c['prevalence']:>8.2f}{'--':>14}{'[--, --]':>22}",
                )
            else:
                iqr = f"[{p25:.2f}, {p75:.2f}]"
                print(
                    f"  {lobe:<22}{comp:<14}{c['n_with_overlap']:>8}"
                    f"{c['prevalence']:>8.2f}{med:>14.2f}{iqr:>22}",
                )
    print("=" * 88)


def main() -> int:
    args = _parse_args()
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
        audit, problems = load_cohort_records(
            args.hitplot_root,
            subject_ids,
            source=args.source,
            strict=args.strict,
        )
    except (FileNotFoundError, pd.errors.ParserError, HitplotCsvError, LutError) as exc:
        print(f"error (--strict): {exc}", file=sys.stderr)
        return 3

    for sid, reason in problems:
        print(f"warning: sub-{sid}: {reason}", file=sys.stderr)

    n_with_csv = int(audit["subject_id"].nunique()) if len(audit) else 0
    summary = summarise_cells(audit, n_total=cm.n)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_csv, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cohort_yaml": str(args.cohort_yaml),
                "hitplot_root": str(args.hitplot_root),
                "source": args.source,
                "n_total": int(cm.n),
                "n_with_csv": n_with_csv,
                "n_problems": len(problems),
                "lobes": list(LOBES),
                "compartments": list(COMPARTMENTS),
                "cells": summary,
                "problems": [{"subject_id": sid, "reason": r} for sid, r in problems],
            },
            indent=2,
        ),
    )
    render_heatmap_pdf(summary, args.out_pdf, source=args.source, n_total=cm.n)

    _print_summary_to_stdout(
        summary,
        n_total=cm.n,
        n_with_csv=n_with_csv,
        n_problems=len(problems),
        source=args.source,
    )
    print(f"  wrote: {args.out_pdf}")
    print(f"  wrote: {args.out_json}")
    print(f"  wrote: {args.out_csv}")

    # Exit 1 if any compartment column has zero subjects with data.
    by_lobe_comp = {(c["lobe"], c["compartment"]): c for c in summary}
    any_empty_compartment = any(
        all(by_lobe_comp[(lobe, comp)]["n_with_overlap"] == 0 for lobe in LOBES)
        for comp in COMPARTMENTS
    )
    return 1 if any_empty_compartment else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
