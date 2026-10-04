"""Build a descriptive functional-anatomy Hit-Plot cohort summary.

This producer is intentionally conservative: it reads the existing PR-7f
per-subject Hit-Plot CSVs and the matching FreeSurfer ``wmparc`` LUT JSONs,
then aggregates tumor burden into a small set of transparent anatomical-label
groups. It does not run tractography, atlas warping, fMRI, OAR proximity, or
any treatment-planning/outcome model.

Outputs:

* ``outputs/figures/fig_functional_anatomy_hitplot_<source>.pdf`` --- a
  four-panel heatmap (all subjects + OS tertiles) by default. Cell color is
  the median share of compartment volume in the group; text annotation is
  ``median %`` plus prevalence ``n=K/N``.
* ``...json`` --- machine-readable summary, group definitions, and problems.
* ``...csv`` --- per-subject audit trail. Reviewers can recompute every cell
  from this CSV alone.

The functional groups are anatomical-label adjacency summaries derived from
``wmparc``; they are not tractography, individualized eloquence mapping, fMRI,
OAR proximity, treatment-planning validation, or outcome prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from hpgs.io import load_cohort_metadata, normalize_subject_id
from hpgs.parcellate.lobes import LOBES, assign_lobe_from_label, strip_hemisphere_prefix

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_HITPLOT_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_PDF = Path("outputs/figures/fig_functional_anatomy_hitplot_gt.pdf")
DEFAULT_OUT_JSON = Path("outputs/figures/fig_functional_anatomy_hitplot_gt.json")
DEFAULT_OUT_CSV = Path("outputs/figures/fig_functional_anatomy_hitplot_gt.csv")

COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")
ACCEPTED_SOURCES: frozenset[str] = frozenset(
    {"dl", "gt", "raidionics", "segmentglioma", "tumorsynth"},
)
OS_TERTILES: tuple[str, ...] = ("short", "mid", "long")

REQUIRED_HITPLOT_COLUMNS: tuple[str, ...] = (
    "label",
    "compartment",
    "parcel_voxels",
    "overlap_voxels",
    "overlap_volume_mm3",
)

MOTOR_ADJACENT_STEMS: frozenset[str] = frozenset(
    {
        "precentral",
        "postcentral",
        "paracentral",
        "supramarginal",
    },
)
LANGUAGE_ADJACENT_LEFT_STEMS: frozenset[str] = frozenset(
    {
        "parsopercularis",
        "parstriangularis",
        "superiortemporal",
        "middletemporal",
        "bankssts",
        "supramarginal",
    },
)
DEEP_MIDLINE_STEMS: frozenset[str] = frozenset(
    {
        "Thalamus",
        "Caudate",
        "Putamen",
        "Pallidum",
        "Accumbens-area",
        "VentralDC",
        "CC_Anterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Mid_Posterior",
        "CC_Posterior",
    },
)


@dataclass(frozen=True)
class GroupDef:
    group_id: str
    label: str
    group_type: str
    description: str


LOBE_GROUPS: tuple[GroupDef, ...] = tuple(
    GroupDef(
        group_id=f"lobe:{lobe}",
        label=f"lobe: {lobe}",
        group_type="lobar",
        description=(
            "Existing canonical wmparc-to-lobe grouping from hpgs.parcellate.lobes.assign_lobe."
        ),
    )
    for lobe in LOBES
)
FUNCTIONAL_GROUPS: tuple[GroupDef, ...] = (
    GroupDef(
        group_id="functional:motor_adjacent",
        label="motor-adjacent",
        group_type="functional-adjacency",
        description=(
            "Bilateral wmparc cortical/juxtacortical labels whose stems are "
            "precentral, postcentral, paracentral, or supramarginal."
        ),
    ),
    GroupDef(
        group_id="functional:left_language_adjacent",
        label="left language-adjacent",
        group_type="functional-adjacency",
        description=(
            "Left-hemisphere wmparc cortical/juxtacortical labels whose stems "
            "are parsopercularis, parstriangularis, superiortemporal, "
            "middletemporal, bankssts, or supramarginal."
        ),
    ),
    GroupDef(
        group_id="functional:deep_midline_commissural",
        label="deep-midline / commissural",
        group_type="functional-adjacency",
        description=(
            "Hemisphere-stripped thalamus, basal-ganglia/ventral-DC labels, "
            "and corpus-callosum wmparc labels."
        ),
    ),
)
GROUP_DEFS: tuple[GroupDef, ...] = LOBE_GROUPS + FUNCTIONAL_GROUPS
GROUP_BY_ID: dict[str, GroupDef] = {g.group_id: g for g in GROUP_DEFS}
GROUP_IDS: tuple[str, ...] = tuple(g.group_id for g in GROUP_DEFS)


class HitplotCsvError(ValueError):
    """Raised when a Hit-Plot CSV is missing required structure."""


class LutError(ValueError):
    """Raised when a wmparc LUT JSON is missing or malformed."""


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
        help="Hit-Plot source CSV suffix to aggregate (default: %(default)s).",
    )
    p.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument(
        "--no-os-tertile",
        action="store_true",
        help="Render only the all-subject panel; the CSV/JSON still keep OS_tertile.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on the first missing / malformed CSV or LUT.",
    )
    return p.parse_args()


def _hitplot_csv_path(hitplot_root: Path, subject_id: str, source: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return hitplot_root / f"sub-{sid}" / "hitplot" / f"sub-{sid}_hitplot_{source}.csv"


def _lut_json_path(hitplot_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return hitplot_root / f"sub-{sid}" / "parcellation" / f"sub-{sid}_wmparc_lut.json"


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


def _functional_group_ids(name: str) -> tuple[str, ...]:
    """Return the non-lobar functional-adjacency groups for one wmparc name."""
    stem, hemi = strip_hemisphere_prefix(name)
    groups: list[str] = []
    if stem in MOTOR_ADJACENT_STEMS:
        groups.append("functional:motor_adjacent")
    if hemi == "L" and stem in LANGUAGE_ADJACENT_LEFT_STEMS:
        groups.append("functional:left_language_adjacent")
    if stem in DEEP_MIDLINE_STEMS:
        groups.append("functional:deep_midline_commissural")
    return tuple(groups)


def group_ids_for_label(label_id: int, lut: Mapping[int | str, str]) -> tuple[str, ...]:
    """Return all analysis groups a wmparc label contributes to.

    Lobar groups form a partition. Functional-adjacency groups are additional
    overlapping summaries; therefore rows should not be interpreted as summing
    to 100% once functional rows are included.
    """
    name = lut.get(int(label_id)) or lut.get(str(int(label_id))) or "Unknown"
    lobe = assign_lobe_from_label(int(label_id), lut)
    return (f"lobe:{lobe}", *_functional_group_ids(name))


def _infer_voxel_volume_mm3(df: pd.DataFrame) -> float:
    nonzero = df[df["overlap_voxels"].astype(float) > 0]
    if len(nonzero):
        ratios = nonzero["overlap_volume_mm3"].astype(float) / nonzero["overlap_voxels"].astype(
            float
        )
        voxel_volume_mm3 = float(np.nanmedian(ratios.to_numpy()))
        if np.isfinite(voxel_volume_mm3) and voxel_volume_mm3 > 0:
            return voxel_volume_mm3
    return 1.0


def aggregate_subject(
    subject_id: str,
    hitplot_df: pd.DataFrame,
    lut: Mapping[int, str],
    *,
    os_tertile: str,
) -> pd.DataFrame:
    """Collapse one subject's per-label Hit-Plot CSV to analysis groups."""
    sid = normalize_subject_id(subject_id)
    df = hitplot_df.copy()
    df["label"] = df["label"].astype(int)
    df["overlap_volume_mm3"] = df["overlap_volume_mm3"].astype(float).fillna(0.0)
    df["parcel_voxels"] = df["parcel_voxels"].astype(float).fillna(0.0)
    voxel_volume_mm3 = _infer_voxel_volume_mm3(df)

    # Denominator for share_of_compartment_pct must be the original
    # compartment volume, not a grouped sum, because functional groups overlap.
    comp_totals = (
        df.groupby("compartment", as_index=False)["overlap_volume_mm3"]
        .sum()
        .rename(columns={"overlap_volume_mm3": "compartment_total_mm3"})
    )

    overlap_rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        for group_id in group_ids_for_label(int(row.label), lut):
            overlap_rows.append(
                {
                    "group_id": group_id,
                    "compartment": str(row.compartment),
                    "overlap_volume_mm3": float(row.overlap_volume_mm3),
                },
            )
    overlap_per_cell = (
        pd.DataFrame(overlap_rows)
        .groupby(["group_id", "compartment"], as_index=False)["overlap_volume_mm3"]
        .sum()
    )

    volume_rows: list[dict[str, Any]] = []
    unique_labels = df.drop_duplicates(subset=["label"])[["label", "parcel_voxels"]]
    for row in unique_labels.itertuples(index=False):
        for group_id in group_ids_for_label(int(row.label), lut):
            volume_rows.append(
                {
                    "group_id": group_id,
                    "group_volume_mm3": float(row.parcel_voxels) * voxel_volume_mm3,
                },
            )
    group_volumes = (
        pd.DataFrame(volume_rows).groupby("group_id", as_index=False)["group_volume_mm3"].sum()
    )

    grid = pd.MultiIndex.from_product(
        [GROUP_IDS, COMPARTMENTS],
        names=["group_id", "compartment"],
    ).to_frame(index=False)
    cells = grid.merge(overlap_per_cell, on=["group_id", "compartment"], how="left")
    cells["overlap_volume_mm3"] = cells["overlap_volume_mm3"].fillna(0.0)
    cells = cells.merge(group_volumes, on="group_id", how="left")
    cells["group_volume_mm3"] = cells["group_volume_mm3"].fillna(0.0)
    cells = cells.merge(comp_totals, on="compartment", how="left")
    cells["compartment_total_mm3"] = cells["compartment_total_mm3"].fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cells["share_of_group_pct"] = np.where(
            cells["group_volume_mm3"].to_numpy() > 0,
            100.0
            * cells["overlap_volume_mm3"].to_numpy()
            / np.maximum(cells["group_volume_mm3"].to_numpy(), 1e-12),
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
    cells["OS_tertile"] = os_tertile
    cells["group_label"] = cells["group_id"].map(lambda g: GROUP_BY_ID[g].label)
    cells["group_type"] = cells["group_id"].map(lambda g: GROUP_BY_ID[g].group_type)
    return cells[
        [
            "subject_id",
            "OS_tertile",
            "group_id",
            "group_label",
            "group_type",
            "compartment",
            "group_volume_mm3",
            "overlap_volume_mm3",
            "share_of_group_pct",
            "share_of_compartment_pct",
        ]
    ]


def load_cohort_records(
    hitplot_root: Path,
    subject_ids: Iterable[str],
    os_tertile_by_subject: Mapping[str, str],
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
            cells = aggregate_subject(
                sid,
                df,
                lut,
                os_tertile=os_tertile_by_subject.get(sid, "unknown"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            problems.append((sid, f"aggregation failed: {exc}"))
            if strict:
                raise
            continue
        rows.append(cells)

    if not rows:
        return pd.DataFrame(columns=_audit_columns()), problems
    return pd.concat(rows, ignore_index=True, sort=False), problems


def _audit_columns() -> list[str]:
    return [
        "subject_id",
        "OS_tertile",
        "group_id",
        "group_label",
        "group_type",
        "compartment",
        "group_volume_mm3",
        "overlap_volume_mm3",
        "share_of_group_pct",
        "share_of_compartment_pct",
    ]


def _empty_cell(
    context: str, group_id: str, comp: str, n_total: int, n_with_csv: int
) -> dict[str, Any]:
    g = GROUP_BY_ID[group_id]
    return {
        "context": context,
        "group_id": group_id,
        "group_label": g.label,
        "group_type": g.group_type,
        "compartment": comp,
        "n_with_overlap": 0,
        "n_total": int(n_total),
        "n_with_csv": int(n_with_csv),
        "prevalence": 0.0,
        "share_of_compartment_pct_median": None,
        "share_of_compartment_pct_p25": None,
        "share_of_compartment_pct_p75": None,
        "share_of_group_pct_median_when_present": None,
        "share_of_group_pct_p25_when_present": None,
        "share_of_group_pct_p75_when_present": None,
        "total_overlap_volume_mm3_sum": 0.0,
    }


def summarise_groups(
    audit: pd.DataFrame,
    *,
    n_total_by_context: Mapping[str, int],
    contexts: Iterable[str],
) -> list[dict[str, Any]]:
    """Return per-(context, group, compartment) cohort summaries."""
    out: list[dict[str, Any]] = []
    for context in contexts:
        if context == "all":
            sub_audit = audit
        else:
            sub_audit = audit[audit["OS_tertile"] == context] if not audit.empty else audit
        n_total = int(n_total_by_context.get(context, 0))
        n_with_csv = int(sub_audit["subject_id"].nunique()) if len(sub_audit) else 0
        if sub_audit.empty:
            for group_id in GROUP_IDS:
                for comp in COMPARTMENTS:
                    out.append(_empty_cell(context, group_id, comp, n_total, n_with_csv))
            continue

        for group_id in GROUP_IDS:
            for comp in COMPARTMENTS:
                cell_rows = sub_audit[
                    (sub_audit["group_id"] == group_id) & (sub_audit["compartment"] == comp)
                ]
                present = cell_rows[cell_rows["overlap_volume_mm3"].astype(float) > 0]
                n_with = len(present)
                soc = cell_rows["share_of_compartment_pct"].to_numpy(dtype=float)
                soc_finite = soc[np.isfinite(soc)]
                sog_present = present["share_of_group_pct"].to_numpy(dtype=float)
                sog_present = sog_present[np.isfinite(sog_present)]

                c = _empty_cell(context, group_id, comp, n_total, n_with_csv)
                c["n_with_overlap"] = n_with
                c["prevalence"] = float(n_with) / float(max(n_total, 1))
                c["total_overlap_volume_mm3_sum"] = float(
                    cell_rows["overlap_volume_mm3"].astype(float).sum()
                )
                if soc_finite.size:
                    c["share_of_compartment_pct_median"] = float(np.median(soc_finite))
                    c["share_of_compartment_pct_p25"] = float(np.percentile(soc_finite, 25))
                    c["share_of_compartment_pct_p75"] = float(np.percentile(soc_finite, 75))
                if sog_present.size:
                    c["share_of_group_pct_median_when_present"] = float(np.median(sog_present))
                    c["share_of_group_pct_p25_when_present"] = float(np.percentile(sog_present, 25))
                    c["share_of_group_pct_p75_when_present"] = float(np.percentile(sog_present, 75))
                out.append(c)
    return out


def _set_group_axis_labels(ax: plt.Axes, *, show: bool) -> None:
    ax.set_yticks(np.arange(len(GROUP_IDS)))
    if show:
        ax.set_yticklabels([GROUP_BY_ID[g].label for g in GROUP_IDS], fontsize=9)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def render_functional_heatmap_pdf(
    summary: list[dict[str, Any]],
    out_pdf: Path,
    *,
    source: str,
    contexts: Iterable[str],
) -> None:
    """Render all-subject and OS-tertile functional-anatomy heatmaps."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    contexts = tuple(contexts)
    n_contexts = len(contexts)
    ncols = 2 if n_contexts > 1 else 1
    nrows = int(np.ceil(n_contexts / ncols))
    fig = plt.figure(
        figsize=(6.9 * ncols + 0.45, 0.46 * len(GROUP_IDS) * nrows + 2.5),
    )
    gs = GridSpec(
        nrows,
        ncols + 1,
        figure=fig,
        width_ratios=[1.0] * ncols + [0.055],
        wspace=0.28,
        hspace=0.26,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    cax = fig.add_subplot(gs[:, -1])

    by_key = {(c["context"], c["group_id"], c["compartment"]): c for c in summary}
    all_values = [
        c["share_of_compartment_pct_median"]
        for c in summary
        if c["share_of_compartment_pct_median"] is not None
    ]
    vmax = max(float(np.nanmax(all_values)) if all_values else 1.0, 1.0)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f5f5f5")

    for ax_idx, context in enumerate(contexts):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        mat = np.full((len(GROUP_IDS), len(COMPARTMENTS)), np.nan, dtype=float)
        annotations = np.full(mat.shape, "", dtype=object)
        for i, group_id in enumerate(GROUP_IDS):
            for j, comp in enumerate(COMPARTMENTS):
                cell = by_key.get((context, group_id, comp))
                if cell is None:
                    continue
                med = cell["share_of_compartment_pct_median"]
                if med is not None:
                    mat[i, j] = float(med)
                    annotations[i, j] = (
                        f"{float(med):.1f}%\nn={cell['n_with_overlap']}/{cell['n_total']}"
                    )
                else:
                    annotations[i, j] = f"--\nn=0/{cell['n_total']}"

        im = ax.imshow(
            np.ma.masked_invalid(mat),
            cmap=cmap,
            aspect="auto",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(COMPARTMENTS)))
        ax.set_xticklabels(COMPARTMENTS, fontsize=10)
        _set_group_axis_labels(ax, show=ax_idx % ncols == 0)
        ax.set_title(f"{context} (source={source})", fontsize=11, pad=10)
        for i in range(len(GROUP_IDS)):
            for j in range(len(COMPARTMENTS)):
                value = mat[i, j]
                text_color = "white" if (np.isfinite(value) and value > 0.55 * vmax) else "black"
                ax.text(
                    j,
                    i,
                    annotations[i, j],
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=text_color,
                )

    for ax_idx in range(n_contexts, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].axis("off")

    fig.suptitle(
        "Functional-anatomy Hit-Plot summary: anatomical-label adjacency only",
        fontsize=14,
        y=0.985,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("median share of compartment volume in group (%)", fontsize=10, labelpad=12)
    cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(
        left=0.13,
        right=0.95,
        top=0.93,
        bottom=0.06,
    )
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _print_summary_to_stdout(
    summary: list[dict[str, Any]],
    *,
    n_total_by_context: Mapping[str, int],
    n_with_csv: int,
    n_problems: int,
    source: str,
) -> None:
    print()
    print("=" * 104)
    print(
        f"functional-anatomy Hit-Plot  source={source}  "
        f"cohort_n={n_total_by_context.get('all', 0)}  "
        f"with_csv={n_with_csv}  problems={n_problems}",
    )
    print("=" * 104)
    print(
        f"  {'context':<8}{'group':<32}{'compartment':<12}{'n_with':>8}"
        f"{'prev':>8}{'share_med':>14}{'share_iqr':>22}",
    )
    print("-" * 104)
    for c in summary:
        if c["context"] != "all":
            continue
        med = c["share_of_compartment_pct_median"]
        p25 = c["share_of_compartment_pct_p25"]
        p75 = c["share_of_compartment_pct_p75"]
        if med is None:
            print(
                f"  {c['context']:<8}{c['group_label']:<32}{c['compartment']:<12}"
                f"{c['n_with_overlap']:>8}{c['prevalence']:>8.2f}{'--':>14}{'[--, --]':>22}",
            )
        else:
            iqr = f"[{p25:.2f}, {p75:.2f}]"
            print(
                f"  {c['context']:<8}{c['group_label']:<32}{c['compartment']:<12}"
                f"{c['n_with_overlap']:>8}{c['prevalence']:>8.2f}{med:>14.2f}{iqr:>22}",
            )
    print("=" * 104)


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
    os_by_subject = {str(idx): str(row["OS_tertile"]) for idx, row in cm.df.iterrows()}
    try:
        audit, problems = load_cohort_records(
            args.hitplot_root,
            subject_ids,
            os_by_subject,
            source=args.source,
            strict=args.strict,
        )
    except (FileNotFoundError, pd.errors.ParserError, HitplotCsvError, LutError) as exc:
        print(f"error (--strict): {exc}", file=sys.stderr)
        return 3

    for sid, reason in problems:
        print(f"warning: sub-{sid}: {reason}", file=sys.stderr)

    contexts = ("all",) if args.no_os_tertile else ("all", *OS_TERTILES)
    n_total_by_context = {"all": int(cm.n)}
    n_total_by_context.update(
        {tert: int((cm.df["OS_tertile"] == tert).sum()) for tert in OS_TERTILES}
    )
    summary = summarise_groups(audit, n_total_by_context=n_total_by_context, contexts=contexts)
    n_with_csv = int(audit["subject_id"].nunique()) if len(audit) else 0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_csv, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cohort_yaml": str(args.cohort_yaml),
                "metadata_csv": str(args.metadata_csv),
                "hitplot_root": str(args.hitplot_root),
                "source": args.source,
                "safety_statement": (
                    "These functional-anatomy groups are anatomical-label adjacency "
                    "summaries derived from wmparc; they are not tractography, "
                    "individualized eloquence mapping, fMRI, OAR proximity, "
                    "treatment-planning validation, or outcome prediction."
                ),
                "n_total_by_context": dict(n_total_by_context),
                "n_with_csv": n_with_csv,
                "n_problems": len(problems),
                "contexts": list(contexts),
                "compartments": list(COMPARTMENTS),
                "groups": [asdict(g) for g in GROUP_DEFS],
                "cells": summary,
                "problems": [{"subject_id": sid, "reason": r} for sid, r in problems],
            },
            indent=2,
        ),
    )
    render_functional_heatmap_pdf(summary, args.out_pdf, source=args.source, contexts=contexts)

    _print_summary_to_stdout(
        summary,
        n_total_by_context=n_total_by_context,
        n_with_csv=n_with_csv,
        n_problems=len(problems),
        source=args.source,
    )
    print(f"  wrote: {args.out_pdf}")
    print(f"  wrote: {args.out_json}")
    print(f"  wrote: {args.out_csv}")

    # Exit 1 if the figure was emitted but every compartment has no finite
    # cohort-level data. This mirrors the tolerant producer style elsewhere.
    all_cells = [c for c in summary if c["context"] == "all"]
    any_finite = any(c["share_of_compartment_pct_median"] is not None for c in all_cells)
    return 0 if any_finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
