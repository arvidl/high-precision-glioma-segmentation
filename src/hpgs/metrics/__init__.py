"""Segmentation quality metrics: Dice, Hausdorff-95, volumetric error, sens/spec.

Plus :func:`concordance_correlation_coefficient` for cross-cell summary
metrics in Tables 2/3 and the Hit-Plot agreement panel.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _to_bool(x: np.ndarray) -> np.ndarray:
    return x.astype(bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Soft Dice on binary masks; returns ``nan`` if both masks are empty."""
    p, g = _to_bool(pred), _to_bool(gt)
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * (p & g).sum() / denom)


def volumetric_error(pred: np.ndarray, gt: np.ndarray, voxel_volume_mm3: float = 1.0) -> float:
    """Signed volumetric error in mm^3, ``vol(pred) - vol(gt)``."""
    return float((pred.astype(bool).sum() - gt.astype(bool).sum()) * voxel_volume_mm3)


def absolute_volumetric_error(
    pred: np.ndarray, gt: np.ndarray, voxel_volume_mm3: float = 1.0
) -> float:
    return abs(volumetric_error(pred, gt, voxel_volume_mm3))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _to_bool(pred), _to_bool(gt)
    if g.sum() == 0:
        return float("nan")
    return float((p & g).sum() / g.sum())


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _to_bool(pred), _to_bool(gt)
    not_g = ~g
    if not_g.sum() == 0:
        return float("nan")
    return float((~p & not_g).sum() / not_g.sum())


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of surface (boundary) voxels of ``mask``.

    A voxel is on the surface iff it is foreground and at least one
    6-connected neighbour is background. Implemented as the XOR of
    ``mask`` with its 6-connected erosion; voxels touching the array
    edge are treated as surface (``border_value=0`` in the erosion).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def hausdorff95(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    percentile: float = 95.0,
) -> float:
    """95th-percentile symmetric Hausdorff distance in mm.

    The directed 95th-percentile distance from ``A`` to ``B`` is the
    95th percentile of ``{ d(a, B) : a in surface(A) }`` in millimetres.
    The symmetric value reported here is the *maximum* of the two
    directed 95th percentiles, which is the convention used by the
    BraTS challenges (e.g. Bakas et al., 2018) and most clinical
    segmentation reports; it matches what reviewers expect when they
    write "HD95".

    Distances are computed in millimetres by passing ``voxel_spacing``
    to :func:`scipy.ndimage.distance_transform_edt` as the ``sampling``
    argument. ``voxel_spacing`` is in ``(x, y, z)`` order matching the
    array axes (NIfTI / RAS convention assumed by the rest of this
    package).

    Edge cases:
        * If both masks are empty: returns ``nan`` (no surface to
          measure --- the metric is undefined; callers should drop or
          report this explicitly).
        * If exactly one mask is empty: returns ``inf`` (BraTS
          convention; the alternative of "image diagonal" hides
          catastrophic failure under a finite number).
        * If ``percentile == 100.0``: degenerates to the classical
          (non-robust) symmetric Hausdorff distance.

    Parameters
    ----------
    pred, gt:
        3-D boolean (or coercible to boolean) arrays of equal shape.
    voxel_spacing:
        Per-axis voxel size in millimetres. Defaults to isotropic
        1 mm, matching the UCSF-PDGM extracted layout.
    percentile:
        Percentile of the directed distance distribution. Defaults to
        the standard 95.

    Returns
    -------
    float
        Symmetric percentile-Hausdorff distance in millimetres.
    """
    p = _to_bool(pred)
    g = _to_bool(gt)
    if p.shape != g.shape:
        raise ValueError(
            f"hausdorff95: shape mismatch pred={p.shape} gt={g.shape}",
        )
    if p.ndim != len(voxel_spacing):
        raise ValueError(
            f"hausdorff95: voxel_spacing has {len(voxel_spacing)} entries but masks are {p.ndim}-D",
        )

    p_any = bool(p.any())
    g_any = bool(g.any())
    if not p_any and not g_any:
        return float("nan")
    if p_any != g_any:
        return float("inf")

    sampling = tuple(float(s) for s in voxel_spacing)
    p_surf = _surface_voxels(p)
    g_surf = _surface_voxels(g)

    # distance_transform_edt(x) returns, for every voxel, the distance
    # to the nearest *zero* voxel of x; we therefore feed it ~surface
    # so that distance is measured to the surface itself.
    d_to_g_surface = distance_transform_edt(~g_surf, sampling=sampling)
    d_to_p_surface = distance_transform_edt(~p_surf, sampling=sampling)

    d_pg = d_to_g_surface[p_surf]  # surface(P) -> surface(G)
    d_gp = d_to_p_surface[g_surf]  # surface(G) -> surface(P)

    if d_pg.size == 0 or d_gp.size == 0:
        # One mask is single-voxel-or-fully-internal; treat as fully overlapping
        # surface and return 0 to avoid spurious inf.
        return 0.0

    p95_pg = float(np.percentile(d_pg, percentile))
    p95_gp = float(np.percentile(d_gp, percentile))
    return max(p95_pg, p95_gp)


def concordance_correlation_coefficient(
    x: np.ndarray, y: np.ndarray, *, drop_nan: bool = True
) -> float:
    """Lin's concordance correlation coefficient (CCC) between paired samples.

    .. math::

        \\rho_c = \\frac{2 \\, s_{xy}}
                       {s_x^2 + s_y^2 + (\\bar x - \\bar y)^2}

    where :math:`s_{xy}` is the population covariance and
    :math:`s_x^2, s_y^2` are the population variances (ddof=0). CCC
    rewards both Pearson correlation *and* mean-/scale-agreement, which
    is why it is the headline summary number for Table 3 (DL vs.
    Raidionics / segment_glioma / BraTS21-manual) and for the per-
    compartment Hit-Plot agreement panel under the Q2 = C decision in
    ``docs/scope_pr7.md``.

    Convention:
        * ``ccc == 1.0`` exactly when ``y == x`` (perfect agreement).
        * ``ccc == -1.0`` exactly when ``y == 2*mean(x) - x`` (perfect
          mirror-image disagreement).
        * Pearson r and CCC differ when there is a constant offset or
          a scale change; CCC penalises both.

    Edge cases:
        * NaN pairs are dropped pairwise when ``drop_nan=True``
          (default); set to ``False`` to require the caller to do this.
        * Returns ``nan`` if fewer than 2 valid paired observations
          remain, or if both variances are zero (no signal to agree on).

    Parameters
    ----------
    x, y:
        1-D paired samples of equal length (or any shape, flattened).
    drop_nan:
        If True (default), drop pairs where either coordinate is NaN
        before computing the statistic.

    Returns
    -------
    float
        Lin's CCC in :math:`[-1, 1]`, or ``nan`` for degenerate inputs.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"concordance_correlation_coefficient: shape mismatch x={x_arr.shape} y={y_arr.shape}",
        )

    if drop_nan:
        valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]

    if x_arr.size < 2:
        return float("nan")

    mean_x = float(np.mean(x_arr))
    mean_y = float(np.mean(y_arr))
    var_x = float(np.mean((x_arr - mean_x) ** 2))  # population variance, ddof=0
    var_y = float(np.mean((y_arr - mean_y) ** 2))
    cov_xy = float(np.mean((x_arr - mean_x) * (y_arr - mean_y)))

    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0.0:
        return float("nan")
    return 2.0 * cov_xy / denom
