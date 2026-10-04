"""Hit-Plot generation: per-parcel tumor-burden quantification and visualisation.

The Hit-Plot family committed to in the JMET revision (see
``docs/design_hitplot_alternatives.md`` and ``docs/scope_pr7.md`` § 1)
has two members:

* **Hit-Plot v1 (deterministic).** Given a hard tumor segmentation
  (e.g. NCR/ED/ET label map) and a parcellation in the *same* native
  space, compute the per-parcel coverage of each tumor compartment
  (WT, TC, ET). Implemented by :func:`parcel_tumor_burden` (single
  compartment, legacy) and :func:`compartment_region_matrix` (all
  three compartments at once, tidy CSV-friendly schema for the PR-7
  artefact contract).

* **Probabilistic Hit-Plot.** Same matrix, but propagated from the
  per-class sigmoid probabilities of the unified segmenter (see
  ``hpgs.segment.unified.predict_brats3``). Mean and std are computed
  in closed form from the Poisson-binomial of the per-voxel
  probabilities, *no* Monte Carlo sampling and *no* RNG --- this keeps
  results bitwise-deterministic for a fixed input. Implemented by
  :func:`probabilistic_hitplot`.

Both matrix functions share the same long-format schema so that the
downstream Hit-Plot agreement panel can join their outputs on
``(label, compartment)`` without bespoke pivoting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpgs.segment import derive_subregions

DEFAULT_COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")


def parcel_tumor_burden(
    parcellation: np.ndarray, tumor_mask: np.ndarray, label_lut: dict[int, str]
) -> pd.DataFrame:
    """Compute the % tumor coverage of each parcel.

    Legacy single-compartment helper retained for the existing
    notebook. New code should prefer :func:`compartment_region_matrix`
    (which reports WT/TC/ET in one tidy DataFrame) so the on-disk CSVs
    in ``data/derivatives_cohort50/sub-XXXX/hitplot/`` share a single
    schema.

    Parameters
    ----------
    parcellation
        Integer label map (SynthSeg or FreeSurfer aparc+aseg), native space.
    tumor_mask
        Binary tumor mask (e.g. WT) in the same space.
    label_lut
        Mapping of label id → anatomical name.

    Returns
    -------
    DataFrame with columns ``label, name, parcel_voxels, tumor_voxels, pct_burden``.
    """
    if parcellation.shape != tumor_mask.shape:
        raise ValueError(f"Shape mismatch: {parcellation.shape} vs {tumor_mask.shape}")
    rows = []
    tm = tumor_mask.astype(bool)
    for lab, name in label_lut.items():
        sel = parcellation == lab
        n_par = int(sel.sum())
        if n_par == 0:
            continue
        n_tum = int((sel & tm).sum())
        rows.append(
            {
                "label": lab,
                "name": name,
                "parcel_voxels": n_par,
                "tumor_voxels": n_tum,
                "pct_burden": 100.0 * n_tum / n_par,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_burden", ascending=False).reset_index(drop=True)


def topk_hit_plot(df: pd.DataFrame, k: int = 20, *, ax=None):
    """Render the top-k parcels by % burden as a horizontal bar plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.3 * k + 1))
    sub = df.head(k).iloc[::-1]
    ax.barh(sub["name"], sub["pct_burden"])
    ax.set_xlabel("Tumor coverage of parcel (%)")
    ax.set_title(f"Hit-Plot (top {k} parcels)")
    return ax


def _validate_parcellation(
    parcellation: np.ndarray,
    expected_shape: tuple[int, ...],
    label_lut: dict[int, str],
) -> None:
    if parcellation.shape != expected_shape:
        raise ValueError(
            f"compartment_region_matrix: parcellation shape {parcellation.shape} "
            f"!= reference shape {expected_shape}",
        )
    if not label_lut:
        raise ValueError("compartment_region_matrix: empty label_lut")


def compartment_region_matrix(
    label_map: np.ndarray,
    parcellation: np.ndarray,
    label_lut: dict[int, str],
    *,
    voxel_volume_mm3: float = 1.0,
    compartments: tuple[str, ...] = DEFAULT_COMPARTMENTS,
) -> pd.DataFrame:
    """Per-(region, compartment) tumor coverage from a hard segmentation.

    This is the canonical Hit-Plot v1 producer for PR-7. The returned
    DataFrame is in **long format** so that it survives a straight
    ``DataFrame.to_csv`` round-trip and is trivial to pivot to the
    (region x compartment) matrix the agreement panel consumes.

    Parameters
    ----------
    label_map:
        Integer multi-class BraTS-style label map (NCR=1, ED=2, ET=4,
        BG=0) in native space. WT/TC/ET binary masks are derived
        internally via :func:`hpgs.segment.derive_subregions`.
    parcellation:
        Integer parcellation label map in the *same* native space as
        ``label_map`` (e.g. SynthSeg / wmparc, ~100 regions).
    label_lut:
        Mapping of parcellation label id → anatomical name. Labels
        absent from the parcellation array are silently skipped (no
        rows emitted).
    voxel_volume_mm3:
        Voxel volume in mm^3, used to fill the ``overlap_volume_mm3``
        column. Defaults to 1.0 (UCSF-PDGM extracted layout is 1 mm
        isotropic). The underlying ``overlap_voxels`` count is always
        an exact integer regardless.
    compartments:
        Subset of ``("WT", "TC", "ET")`` to report. Defaults to all
        three.

    Returns
    -------
    DataFrame
        Long-format columns:

        * ``label`` (int) --- parcellation label id
        * ``name`` (str) --- anatomical name from ``label_lut``
        * ``compartment`` (str, one of ``compartments``)
        * ``parcel_voxels`` (int) --- voxels of this region
        * ``compartment_voxels`` (int) --- voxels of this compartment in
          the *whole* subject volume (constant within a compartment)
        * ``overlap_voxels`` (int) --- voxels of the region that are
          also in this compartment
        * ``overlap_volume_mm3`` (float)
        * ``pct_of_parcel`` (float) --- 100 * overlap_voxels / parcel_voxels
        * ``pct_of_compartment`` (float) --- 100 * overlap_voxels /
          compartment_voxels (NaN if the compartment is empty)

        Rows are sorted by ``(compartment, pct_of_parcel desc)`` so a
        per-compartment top-k slice is just ``head(k)`` after
        ``groupby("compartment")``.

    Raises
    ------
    ValueError
        If shapes mismatch, the LUT is empty, or an unknown compartment
        name is requested.
    """
    if label_map.shape != parcellation.shape:
        raise ValueError(
            f"compartment_region_matrix: label_map shape {label_map.shape} "
            f"!= parcellation shape {parcellation.shape}",
        )
    _validate_parcellation(parcellation, label_map.shape, label_lut)
    unknown = [c for c in compartments if c not in DEFAULT_COMPARTMENTS]
    if unknown:
        raise ValueError(
            f"compartment_region_matrix: unknown compartment(s) {unknown}; "
            f"expected subset of {DEFAULT_COMPARTMENTS}",
        )

    masks = derive_subregions(label_map)
    compartment_masks: dict[str, np.ndarray] = {
        "WT": masks.wt.astype(bool),
        "TC": masks.tc.astype(bool),
        "ET": masks.et.astype(bool),
    }
    compartment_totals: dict[str, int] = {c: int(compartment_masks[c].sum()) for c in compartments}

    rows: list[dict] = []
    for lab, name in label_lut.items():
        region_mask = parcellation == lab
        n_par = int(region_mask.sum())
        if n_par == 0:
            continue
        for comp in compartments:
            n_overlap = int(np.logical_and(region_mask, compartment_masks[comp]).sum())
            n_comp = compartment_totals[comp]
            rows.append(
                {
                    "label": int(lab),
                    "name": name,
                    "compartment": comp,
                    "parcel_voxels": n_par,
                    "compartment_voxels": n_comp,
                    "overlap_voxels": n_overlap,
                    "overlap_volume_mm3": float(n_overlap * voxel_volume_mm3),
                    "pct_of_parcel": 100.0 * n_overlap / n_par,
                    "pct_of_compartment": (
                        100.0 * n_overlap / n_comp if n_comp > 0 else float("nan")
                    ),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["compartment", "pct_of_parcel"], ascending=[True, False]).reset_index(
        drop=True
    )


def probabilistic_hitplot(
    probabilities: np.ndarray,
    parcellation: np.ndarray,
    label_lut: dict[int, str],
    *,
    voxel_volume_mm3: float = 1.0,
    compartments: tuple[str, ...] = DEFAULT_COMPARTMENTS,
    compartment_axis: int = 0,
) -> pd.DataFrame:
    """Probabilistic Hit-Plot from per-class sigmoid probabilities.

    Treats every voxel as an independent Bernoulli with success
    probability ``p_c(v)`` (the per-class sigmoid output of the
    unified segmenter, ``hpgs.segment.unified.predict_brats3``). For
    region :math:`r` and compartment :math:`c`, the *number of
    overlap voxels* is then a Poisson binomial whose mean and variance
    are available in closed form:

    .. math::

        \\mathbb{E}[N_{c,r}] = \\sum_{v \\in r} p_c(v)
        \\qquad
        \\mathrm{Var}[N_{c,r}] = \\sum_{v \\in r} p_c(v)\\,(1 - p_c(v))

    The std is therefore :math:`\\sqrt{\\mathrm{Var}[N_{c,r}]}`. No Monte
    Carlo sampling and no RNG: results are bitwise-deterministic for a
    fixed input, which the agreement panel relies on for
    reproducibility.

    Parameters
    ----------
    probabilities:
        Per-class sigmoid probability array. Default layout is
        ``(C, X, Y, Z)`` with ``C == len(compartments)`` --- this is
        the layout produced by
        :class:`hpgs.segment.unified.SegmentationResult.probabilities`.
        Pass ``compartment_axis=-1`` (or any int) to consume the
        NIfTI-friendly ``(X, Y, Z, C)`` layout instead.
    parcellation:
        Integer parcellation label map in the *same* spatial frame as
        the per-channel probabilities.
    label_lut:
        Mapping of parcellation label id → anatomical name. Labels
        absent from the parcellation are silently skipped.
    voxel_volume_mm3:
        Voxel volume in mm^3, used to fill ``overlap_volume_mm3_mean``
        and ``overlap_volume_mm3_std``.
    compartments:
        Names assigned to the per-class channels in
        ``probabilities``, in the order they appear along
        ``compartment_axis``. Defaults to ``("WT", "TC", "ET")``,
        matching the unified segmenter.
    compartment_axis:
        Axis of ``probabilities`` along which the compartments are
        stacked. Default 0 (matching ``predict_brats3``).

    Returns
    -------
    DataFrame
        Long-format columns mirroring :func:`compartment_region_matrix`:

        * ``label``, ``name``, ``compartment``, ``parcel_voxels``
        * ``overlap_voxels_mean``, ``overlap_voxels_std``
        * ``overlap_volume_mm3_mean``, ``overlap_volume_mm3_std``
        * ``pct_of_parcel_mean``, ``pct_of_parcel_std``

        ``compartment_voxels`` is intentionally omitted because it is
        not well-defined for a probabilistic input (the expected total
        ``sum_v p_c(v)`` is reportable but adds little signal beyond
        the per-region totals).

        Rows are sorted by ``(compartment, pct_of_parcel_mean desc)``.

    Raises
    ------
    ValueError
        If the probability array shape does not match
        ``parcellation.shape`` after removing the compartment axis,
        if the number of channels does not match
        ``len(compartments)``, if any probability is outside
        ``[0, 1]``, or if an unknown compartment name is requested.
    """
    unknown = [c for c in compartments if c not in DEFAULT_COMPARTMENTS]
    if unknown:
        raise ValueError(
            f"probabilistic_hitplot: unknown compartment(s) {unknown}; "
            f"expected subset of {DEFAULT_COMPARTMENTS}",
        )

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != parcellation.ndim + 1:
        raise ValueError(
            f"probabilistic_hitplot: probabilities.ndim={probs.ndim} "
            f"must equal parcellation.ndim+1={parcellation.ndim + 1}",
        )
    # Move compartment axis to position 0 to simplify per-channel slicing.
    probs = np.moveaxis(probs, compartment_axis, 0)
    if probs.shape[0] != len(compartments):
        raise ValueError(
            f"probabilistic_hitplot: probabilities has {probs.shape[0]} channels "
            f"along axis {compartment_axis} but compartments={compartments} "
            f"(len={len(compartments)})",
        )
    if probs.shape[1:] != parcellation.shape:
        raise ValueError(
            f"probabilistic_hitplot: spatial shape {probs.shape[1:]} "
            f"!= parcellation shape {parcellation.shape}",
        )
    _validate_parcellation(parcellation, parcellation.shape, label_lut)

    p_min = float(probs.min()) if probs.size else 0.0
    p_max = float(probs.max()) if probs.size else 0.0
    if p_min < 0.0 or p_max > 1.0:
        raise ValueError(
            f"probabilistic_hitplot: probabilities must lie in [0, 1]; "
            f"got [{p_min:.6g}, {p_max:.6g}]",
        )

    rows: list[dict] = []
    for lab, name in label_lut.items():
        region_mask = parcellation == lab
        n_par = int(region_mask.sum())
        if n_par == 0:
            continue
        for ci, comp in enumerate(compartments):
            p = probs[ci][region_mask]  # 1-D vector of voxel probabilities in region
            mean = float(p.sum())
            # Var of sum of independent Bernoullis = sum p (1 - p)
            var = float((p * (1.0 - p)).sum())
            std = float(np.sqrt(var))
            rows.append(
                {
                    "label": int(lab),
                    "name": name,
                    "compartment": comp,
                    "parcel_voxels": n_par,
                    "overlap_voxels_mean": mean,
                    "overlap_voxels_std": std,
                    "overlap_volume_mm3_mean": mean * voxel_volume_mm3,
                    "overlap_volume_mm3_std": std * voxel_volume_mm3,
                    "pct_of_parcel_mean": 100.0 * mean / n_par,
                    "pct_of_parcel_std": 100.0 * std / n_par,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["compartment", "pct_of_parcel_mean"], ascending=[True, False]
    ).reset_index(drop=True)
