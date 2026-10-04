"""QC helpers for cohort segmentation outputs.

Used by :doc:`/notebooks/qc_segmentation_outliers` (stage 1 of the QC plan)
and, when promoted, by ``scripts/qc_segmentation_outliers.py`` (stage 2).

The strict-band thresholds come from the MONAI BraTS Bundle's published
BraTS-2018 held-out validation operating point: WT 0.903 / TC 0.856 /
ET 0.791 Dice. We flag below the floor of the bundle's "healthy ranges"
heuristic already printed by ``scripts/segment_all_cohort50.py``
(WT >= 0.85, TC >= 0.75, ET >= 0.70). The HD95 cap is the same 20 mm
threshold used by that heuristic for tumor-core / enhancing-tumor outliers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from hpgs.io import load_nifti_canonical, normalize_subject_id
from hpgs.segment import derive_subregions

__all__ = [
    "BRATS2018_OP_POINT",
    "BRATS_COMPARTMENTS",
    "HD95_MAX_MM_DEFAULT",
    "OutlierRow",
    "brain_bbox_ij",
    "classify_outliers",
    "load_metrics_table",
    "render_outlier_grid",
    "render_qc_panel",
    "render_single_subject_panel",
    "tumor_centroid",
]

BRATS2018_OP_POINT: Mapping[str, float] = {
    "WT": 0.85,
    "TC": 0.75,
    "ET": 0.70,
}
HD95_MAX_MM_DEFAULT: float = 20.0
BRATS_COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")


@dataclass(frozen=True)
class OutlierRow:
    """One row of the outlier table — per-subject metrics + flag reasons."""

    subject_id: str
    dice: dict[str, float]
    hd95_mm: dict[str, float]
    flagged: bool
    reasons: tuple[str, ...]


def _read_metrics_json(path: Path) -> dict[str, Any]:
    """Load and lightly validate a PR-7d ``metrics_dl_vs_gt.json`` sidecar."""
    payload = json.loads(path.read_text())
    if payload.get("comparison") != "dl_vs_gt":
        raise ValueError(
            f"{path}: expected comparison='dl_vs_gt', got {payload.get('comparison')!r}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing or non-dict 'metrics' block")
    for compartment in BRATS_COMPARTMENTS:
        if compartment not in metrics:
            raise ValueError(f"{path}: missing compartment {compartment!r} in metrics")
    return payload


def _coerce_float(value: Any) -> float:
    """Coerce a JSON metric value to float; map ``"inf"`` / ``None`` / ``NaN`` to NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        if value.lower() == "inf":
            return float("inf")
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def load_metrics_table(metrics_root: Path | str) -> list[OutlierRow]:
    """Walk ``metrics_root/sub-XXXX/metrics/sub-XXXX_metrics_dl_vs_gt.json``.

    Returns one :class:`OutlierRow` per subject **without** flagging — the
    ``flagged`` field is left ``False`` and ``reasons`` empty. Apply
    :func:`classify_outliers` to populate them.

    Raises :class:`FileNotFoundError` if ``metrics_root`` is not a directory,
    or :class:`ValueError` if any subject's sidecar is malformed.
    """
    root = Path(metrics_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    rows: list[OutlierRow] = []
    for subject_dir in sorted(root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        sidecar = subject_dir / "metrics" / f"{subject_dir.name}_metrics_dl_vs_gt.json"
        if not sidecar.is_file():
            continue
        payload = _read_metrics_json(sidecar)
        sid = normalize_subject_id(subject_dir.name)
        metrics = payload["metrics"]
        rows.append(
            OutlierRow(
                subject_id=sid,
                dice={c: _coerce_float(metrics[c].get("dice")) for c in BRATS_COMPARTMENTS},
                hd95_mm={c: _coerce_float(metrics[c].get("hd95_mm")) for c in BRATS_COMPARTMENTS},
                flagged=False,
                reasons=(),
            )
        )
    return rows


def classify_outliers(
    metrics_root: Path | str,
    *,
    op_point: Mapping[str, float] = BRATS2018_OP_POINT,
    hd95_max_mm: float = HD95_MAX_MM_DEFAULT,
) -> list[OutlierRow]:
    """Flag subjects outside the BraTS-2018 op-point band.

    A subject is flagged if either:

    * any compartment's Dice falls strictly below the op-point floor
      (default WT 0.85 / TC 0.75 / ET 0.70), **or**
    * any compartment's HD95 strictly exceeds ``hd95_max_mm`` (default 20 mm),
      **or**
    * any compartment's Dice or HD95 is NaN (unrecoverable; counted as a
      structurally-missing reference).

    NaN and ``+inf`` HD95 values count as flag reasons; they are not silently
    treated as "ok".
    """
    rows = load_metrics_table(metrics_root)
    flagged: list[OutlierRow] = []
    for row in rows:
        reasons: list[str] = []
        for compartment in BRATS_COMPARTMENTS:
            dice = row.dice[compartment]
            hd95 = row.hd95_mm[compartment]
            floor = float(op_point[compartment])
            if np.isnan(dice):
                reasons.append(f"{compartment}_dice_nan")
            elif dice < floor:
                reasons.append(f"{compartment}_dice_lt_{floor:.2f}({dice:.3f})")
            if np.isnan(hd95):
                reasons.append(f"{compartment}_hd95_nan")
            elif np.isinf(hd95):
                reasons.append(f"{compartment}_hd95_inf")
            elif hd95 > hd95_max_mm:
                reasons.append(f"{compartment}_hd95_gt_{hd95_max_mm:.0f}mm({hd95:.1f})")
        flagged.append(
            OutlierRow(
                subject_id=row.subject_id,
                dice=row.dice,
                hd95_mm=row.hd95_mm,
                flagged=bool(reasons),
                reasons=tuple(reasons),
            )
        )
    return flagged


def tumor_centroid(seg: np.ndarray, *, labels: Iterable[int] | None = None) -> tuple[int, int, int]:
    """Return the integer voxel centroid of a tumor mask.

    ``seg`` is a 3-D label map. By default any non-zero voxel is part of the
    tumor (whole-tumor union). Pass ``labels=[1, 4]`` to restrict to the tumor
    core, etc.

    Falls back to the volume centre when the mask is empty (no flag-the-error
    here — the caller already knows the subject is degenerate; we still want
    a renderable slice).
    """
    if seg.ndim != 3:
        raise ValueError(f"Expected 3-D segmentation, got shape {seg.shape}")
    if labels is None:
        mask = seg != 0
    else:
        mask = np.isin(seg, list(labels))
    if not mask.any():
        return tuple(int(s // 2) for s in seg.shape)  # type: ignore[return-value]
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)
    return tuple(round(float(c)) for c in centroid)  # type: ignore[return-value]


def _normalize_axial(volume: np.ndarray) -> np.ndarray:
    """Percentile-normalise to [0, 1] for grayscale display."""
    lo, hi = np.percentile(volume, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def brain_bbox_ij(
    background: np.ndarray,
    *,
    threshold_percentile: float = 5.0,
    pad: int = 6,
) -> tuple[slice, slice]:
    """In-plane (i, j) bounding box of non-air voxels.

    Computed across the full 3-D volume (any voxel above the
    ``threshold_percentile``-th intensity percentile counts as parenchyma)
    so every axial slice from the same subject crops to the SAME (i, j)
    window — keeping the rendered brain stable across rows. Returns two
    :class:`slice` objects suitable for fancy-indexing the (i, j) plane.

    Falls back to the full volume extents if the volume is empty / all-zero.
    """
    if background.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {background.shape}")
    threshold = float(np.percentile(background, threshold_percentile))
    mask = background > threshold
    if not mask.any():
        return slice(None), slice(None)
    coords = np.argwhere(mask)
    i_lo, j_lo = coords[:, 0].min(), coords[:, 1].min()
    i_hi, j_hi = coords[:, 0].max() + 1, coords[:, 1].max() + 1
    i_lo = max(0, int(i_lo) - pad)
    j_lo = max(0, int(j_lo) - pad)
    i_hi = min(background.shape[0], int(i_hi) + pad)
    j_hi = min(background.shape[1], int(j_hi) + pad)
    return slice(i_lo, i_hi), slice(j_lo, j_hi)


def _radiological_axial(slice_2d: np.ndarray) -> np.ndarray:
    """Radiological display transform ``fliplr(rot90(...))`` for an axial slice.

    Identical to ``scripts/render_figure2_panels._radiological`` (and pinned by
    ``tests/test_orientation.py``): on an RAS-reoriented volume this yields the
    conventional radiological view (image-left = patient-right, anterior up).
    Callers MUST pass an RAS-oriented slice; :func:`render_qc_panel` enforces
    this by loading every input through :func:`hpgs.io.load_nifti_canonical`.
    Broadcasts cleanly over trailing channel dimensions so RGBA mask overlays
    keep their colours.
    """
    return np.fliplr(np.rot90(slice_2d))


def _overlay_axial(
    ax,
    background_axial: np.ndarray,
    overlays: Mapping[str, np.ndarray],
    *,
    alpha: float = 0.45,
) -> None:
    """Render one axial slice with named binary masks on top.

    ``overlays`` keys are compartment names (``"WT"``, ``"TC"``, ``"ET"``);
    fixed colour map below keeps WT yellow / TC orange / ET red, matching the
    legacy Figure 2 convention. Unknown keys are drawn in white.
    """
    colors = {
        "WT": (1.0, 1.0, 0.2),  # yellow
        "TC": (1.0, 0.55, 0.0),  # orange
        "ET": (1.0, 0.0, 0.0),  # red
    }

    ax.imshow(_radiological_axial(background_axial), cmap="gray", interpolation="nearest")
    for name, mask_axial in overlays.items():
        if not mask_axial.any():
            continue
        rgba = np.zeros((*mask_axial.shape, 4), dtype=np.float32)
        color = colors.get(name, (1.0, 1.0, 1.0))
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = mask_axial.astype(np.float32) * alpha
        ax.imshow(_radiological_axial(rgba), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    # explicit unused-import guard for static checkers
    _ = plt


def render_qc_panel(
    t1c: np.ndarray,
    ref_label_map: np.ndarray,
    dl_label_map: np.ndarray,
    *,
    coord: tuple[int, int, int],
    axes,
    column_titles: bool = True,
    row_label: str | None = None,
    alpha: float = 0.45,
    crop_to_brain: bool = True,
    crop_pad: int = 6,
) -> None:
    """Render a 3-axis QC row at axial slice ``coord[2]``.

    Layout (left to right):

    1. T1c grayscale (no overlay).
    2. T1c + reference (UCSF-PDGM manual) overlay (WT yellow, TC orange,
       ET red).
    3. T1c + DL overlay (same colour scheme).

    ``axes`` must be a length-3 iterable of matplotlib axes. The function
    does not call ``plt.show()``; the caller controls the figure lifecycle.

    Pass ``column_titles=True`` (default) on the FIRST row of a multi-row
    grid; subsequent rows should pass ``False`` to avoid stacking column
    headers under the previous row's panels. ``row_label`` is set as the
    leftmost axis's ylabel (rotated, vertical) so per-subject annotations
    don't collide with column titles.

    All three label maps must share the native subject grid AND be in
    canonical-RAS voxel order (the caller is responsible for reorienting;
    :func:`render_outlier_grid` does this via :func:`hpgs.io.load_nifti_canonical`).
    """
    if t1c.shape != ref_label_map.shape or t1c.shape != dl_label_map.shape:
        raise ValueError(
            f"shape mismatch: t1c {t1c.shape}, ref {ref_label_map.shape}, "
            f"dl {dl_label_map.shape}"
        )
    if len(coord) != 3:
        raise ValueError(f"coord must be (i, j, k); got {coord!r}")
    k = int(coord[2])
    if not 0 <= k < t1c.shape[2]:
        raise ValueError(f"coord z={k} out of bounds for shape {t1c.shape}")

    bg_full = _normalize_axial(t1c)
    if crop_to_brain:
        i_sl, j_sl = brain_bbox_ij(bg_full, pad=crop_pad)
    else:
        i_sl, j_sl = slice(None), slice(None)
    bg = bg_full[i_sl, j_sl, k]
    ref_masks = derive_subregions(ref_label_map)
    dl_masks = derive_subregions(dl_label_map)
    ref_overlays = {
        "WT": ref_masks.wt[i_sl, j_sl, k],
        "TC": ref_masks.tc[i_sl, j_sl, k],
        "ET": ref_masks.et[i_sl, j_sl, k],
    }
    dl_overlays = {
        "WT": dl_masks.wt[i_sl, j_sl, k],
        "TC": dl_masks.tc[i_sl, j_sl, k],
        "ET": dl_masks.et[i_sl, j_sl, k],
    }

    ax_t1c, ax_ref, ax_dl = axes
    ax_t1c.imshow(_radiological_axial(bg), cmap="gray", interpolation="nearest")
    ax_t1c.set_xticks([])
    ax_t1c.set_yticks([])
    if column_titles:
        ax_t1c.set_title("T1c", fontsize=9)
    if row_label:
        ax_t1c.set_ylabel(row_label, fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
    _overlay_axial(ax_ref, bg, ref_overlays, alpha=alpha)
    if column_titles:
        ax_ref.set_title("reference (UCSF-PDGM)", fontsize=9)
    _overlay_axial(ax_dl, bg, dl_overlays, alpha=alpha)
    if column_titles:
        ax_dl.set_title("DL (MONAI Bundle BraTS3)", fontsize=9)


def _resolve_subject_paths(
    cohort_root: Path,
    deriv_root: Path,
    subject_id: str,
    *,
    background_channel: str = "T1c_bias",
    reference_name: str = "tumor_segmentation",
) -> tuple[Path, Path, Path]:
    """Return ``(t1c_path, ref_path, dl_label_path)`` for a subject."""
    sid = normalize_subject_id(subject_id)
    sub_in = cohort_root / f"sub-{sid}"
    sub_out = deriv_root / f"sub-{sid}"
    t1c = sub_in / f"sub-{sid}_{background_channel}.nii.gz"
    ref = sub_in / f"sub-{sid}_{reference_name}.nii.gz"
    dl = sub_out / "seg_dl" / f"sub-{sid}_seg_brats3_dl.nii.gz"
    for p in (t1c, ref, dl):
        if not p.is_file():
            raise FileNotFoundError(p)
    return t1c, ref, dl


def render_outlier_grid(
    outliers: list[OutlierRow],
    *,
    cohort_root: Path | str,
    deriv_root: Path | str,
    out_path: Path | str | None = None,
    background_channel: str = "T1c_bias",
    reference_name: str = "tumor_segmentation",
    figsize_per_row: tuple[float, float] = (12.0, 4.8),
    centroid_compartment: str = "wt",
    crop_to_brain: bool = True,
    crop_pad: int = 6,
):
    """Render a stacked QC grid (one row per flagged subject).

    Each row is a :func:`render_qc_panel` triple anchored at the
    ``centroid_compartment`` centroid of the **DL** segmentation (so the row
    is centred where the model thinks the lesion is — the natural framing
    for "did the DL miss?" adjudication). Falls back to the reference-mask
    centroid if the DL output is empty for that compartment, and to the
    volume centre if both are empty.

    All three NIfTIs (T1c, reference, DL) are loaded with
    :func:`hpgs.io.load_nifti_canonical` so they share canonical-RAS voxel
    order before slicing — this matches the orientation contract pinned by
    ``tests/test_orientation.py`` and used by the manuscript Figure 2.

    The row's per-subject metrics (Dice + HD95) are placed as a horizontal
    suptitle-style annotation ABOVE each row, with column titles ("T1c" /
    "reference (UCSF-PDGM)" / "DL (MONAI Bundle BraTS3)") shown only on
    row 0. Flag reasons appear as a one-line caption below each row's
    rightmost axis.

    If ``out_path`` is given, writes a PDF and returns the path; otherwise
    returns the matplotlib figure (use this in the notebook to display
    inline).
    """
    cohort_root_path = Path(cohort_root)
    deriv_root_path = Path(deriv_root)

    if not outliers:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No subjects flagged by the BraTS-2018 op-point band.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        if out_path is not None:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            return out
        return fig

    n_rows = len(outliers)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        squeeze=False,
        gridspec_kw={"hspace": 0.32, "wspace": 0.0},
    )

    label_for_centroid = {"wt": None, "tc": [1, 4], "et": [4]}[centroid_compartment]

    for row_idx, row in enumerate(outliers):
        try:
            t1c_path, ref_path, dl_path = _resolve_subject_paths(
                cohort_root_path,
                deriv_root_path,
                row.subject_id,
                background_channel=background_channel,
                reference_name=reference_name,
            )
        except FileNotFoundError as exc:
            for ax in axes[row_idx]:
                ax.axis("off")
            axes[row_idx, 0].text(
                0.02,
                0.5,
                f"sub-{row.subject_id}: missing input ({exc})",
                fontsize=8,
                transform=axes[row_idx, 0].transAxes,
            )
            continue

        t1c_ras, _, _ = load_nifti_canonical(t1c_path)
        ref_ras, _, _ = load_nifti_canonical(ref_path)
        dl_ras, _, _ = load_nifti_canonical(dl_path)
        ref_int = np.asarray(ref_ras).astype(np.int16)
        dl_int = np.asarray(dl_ras).astype(np.int16)

        coord = tumor_centroid(dl_int, labels=label_for_centroid)
        if coord == tuple(int(s // 2) for s in dl_int.shape):
            # DL was empty — fall back to the reference centroid.
            coord = tumor_centroid(ref_int, labels=label_for_centroid)

        render_qc_panel(
            np.asarray(t1c_ras).astype(np.float32),
            ref_int,
            dl_int,
            coord=coord,
            axes=axes[row_idx],
            column_titles=(row_idx == 0),
            row_label=f"sub-{row.subject_id}",
            crop_to_brain=crop_to_brain,
            crop_pad=crop_pad,
        )
        # Per-subject metric annotation above the row (does NOT collide with
        # column titles because the latter only render on row 0).
        metric_text = (
            f"sub-{row.subject_id}  ::  "
            f"Dice WT {row.dice['WT']:.2f} / TC {row.dice['TC']:.2f} / "
            f"ET {row.dice['ET']:.2f}  ::  "
            f"HD95 WT {row.hd95_mm['WT']:.0f} / TC {row.hd95_mm['TC']:.0f} / "
            f"ET {row.hd95_mm['ET']:.0f} mm  ::  z={coord[2]}"
        )
        # y in axes-coords; 1.18 keeps it clear of both column titles (row 0)
        # and the panel imshow region.
        y_offset = 1.18 if row_idx == 0 else 1.05
        axes[row_idx, 1].text(
            0.5,
            y_offset,
            metric_text,
            fontsize=9,
            ha="center",
            va="bottom",
            transform=axes[row_idx, 1].transAxes,
        )
        if row.reasons:
            axes[row_idx, 1].text(
                0.5,
                -0.08,
                "  ::  ".join(row.reasons),
                fontsize=7,
                ha="center",
                va="top",
                color="#b91c1c",
                transform=axes[row_idx, 1].transAxes,
            )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out
    return fig


def render_single_subject_panel(
    subject_id: str,
    *,
    cohort_root: Path | str,
    deriv_root: Path | str,
    out_path: Path | str | None = None,
    row: OutlierRow | None = None,
    background_channel: str = "T1c_bias",
    reference_name: str = "tumor_segmentation",
    figsize: tuple[float, float] = (12.0, 4.8),
    centroid_compartment: str = "wt",
    crop_to_brain: bool = True,
    crop_pad: int = 6,
    caption: str | None = None,
):
    """Render a single-subject 1x3 QC panel (T1c | reference | DL).

    Thin wrapper around :func:`render_qc_panel` that resolves the subject's
    NIfTI paths via :func:`_resolve_subject_paths`, picks an axial slice at
    the DL ``centroid_compartment`` centroid (falling back to the reference
    centroid if DL is empty), and writes a PDF if ``out_path`` is given.
    Used by the supplementary figure for sub-0396.

    If an :class:`OutlierRow` is passed via ``row``, the rendered panel
    carries the same metric annotation used by :func:`render_outlier_grid`
    (Dice + HD95 per compartment, plus the slice index ``z``). The optional
    ``caption`` adds a free-text line below the rightmost axis.
    """
    cohort_root_path = Path(cohort_root)
    deriv_root_path = Path(deriv_root)

    t1c_path, ref_path, dl_path = _resolve_subject_paths(
        cohort_root_path,
        deriv_root_path,
        subject_id,
        background_channel=background_channel,
        reference_name=reference_name,
    )

    t1c_ras, _, _ = load_nifti_canonical(t1c_path)
    ref_ras, _, _ = load_nifti_canonical(ref_path)
    dl_ras, _, _ = load_nifti_canonical(dl_path)
    ref_int = np.asarray(ref_ras).astype(np.int16)
    dl_int = np.asarray(dl_ras).astype(np.int16)

    label_for_centroid = {"wt": None, "tc": [1, 4], "et": [4]}[centroid_compartment]
    coord = tumor_centroid(dl_int, labels=label_for_centroid)
    if coord == tuple(int(s // 2) for s in dl_int.shape):
        coord = tumor_centroid(ref_int, labels=label_for_centroid)

    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False, gridspec_kw={"wspace": 0.0})
    sid = normalize_subject_id(subject_id)
    render_qc_panel(
        np.asarray(t1c_ras).astype(np.float32),
        ref_int,
        dl_int,
        coord=coord,
        axes=axes[0],
        column_titles=True,
        row_label=f"sub-{sid}",
        crop_to_brain=crop_to_brain,
        crop_pad=crop_pad,
    )
    if row is not None:
        metric_text = (
            f"sub-{sid}  ::  "
            f"Dice WT {row.dice['WT']:.2f} / TC {row.dice['TC']:.2f} / "
            f"ET {row.dice['ET']:.2f}  ::  "
            f"HD95 WT {row.hd95_mm['WT']:.0f} / TC {row.hd95_mm['TC']:.0f} / "
            f"ET {row.hd95_mm['ET']:.0f} mm  ::  z={coord[2]}"
        )
        axes[0, 1].text(
            0.5,
            1.18,
            metric_text,
            fontsize=9,
            ha="center",
            va="bottom",
            transform=axes[0, 1].transAxes,
        )
    if caption:
        axes[0, 1].text(
            0.5,
            -0.08,
            caption,
            fontsize=8,
            ha="center",
            va="top",
            color="#1f2937",
            transform=axes[0, 1].transAxes,
        )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out
    return fig
