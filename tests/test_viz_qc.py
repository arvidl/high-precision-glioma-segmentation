"""Unit tests for ``hpgs.viz.qc``.

Stage 1 of the QC plan (`docs/scope_pr7.md`-adjacent; QC-pass on the n=50
DL outputs). These tests pin down the strict-band classifier, centroid
helper and the panel renderer against synthetic fixtures so the notebook
can be re-run deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from hpgs.viz.qc import (
    BRATS2018_OP_POINT,
    BRATS_COMPARTMENTS,
    HD95_MAX_MM_DEFAULT,
    _radiological_axial,
    brain_bbox_ij,
    classify_outliers,
    load_metrics_table,
    render_outlier_grid,
    render_qc_panel,
    render_single_subject_panel,
    tumor_centroid,
)


def _write_metrics_json(
    root: Path,
    sid: str,
    *,
    dice: dict[str, float],
    hd95: dict[str, float | str | None],
) -> Path:
    sub = root / f"sub-{sid}"
    (sub / "metrics").mkdir(parents=True, exist_ok=True)
    sidecar = sub / "metrics" / f"sub-{sid}_metrics_dl_vs_gt.json"
    payload = {
        "schema_version": "1.0",
        "subject_id": sid,
        "comparison": "dl_vs_gt",
        "voxel_volume_mm3": 1.0,
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "metrics": {
            c: {
                "dice": dice[c],
                "hd95_mm": hd95[c],
                "abs_volumetric_error_mm3": 0.0,
                "sensitivity": 1.0,
                "specificity": 1.0,
            }
            for c in BRATS_COMPARTMENTS
        },
    }
    sidecar.write_text(json.dumps(payload))
    return sidecar


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_op_point_constants_match_segment_log_heuristic() -> None:
    assert BRATS2018_OP_POINT == {"WT": 0.85, "TC": 0.75, "ET": 0.70}
    assert HD95_MAX_MM_DEFAULT == 20.0
    assert BRATS_COMPARTMENTS == ("WT", "TC", "ET")


# ---------------------------------------------------------------------------
# radiological orientation (regression guard)
# ---------------------------------------------------------------------------


def test_radiological_axial_is_fliplr_rot90() -> None:
    """Pin the radiological display transform to ``fliplr(rot90(...))``.

    Regression guard for the QC PDF orientation bug: the QC notebook loads
    every NIfTI through ``hpgs.io.load_nifti_canonical`` (canonical RAS), and
    the display transform must therefore match
    ``scripts/render_figure2_panels._radiological`` so QC slices look the same
    way the manuscript Figure 2 panels do (image-left = patient-right,
    anterior up).
    """
    rng = np.random.default_rng(0)
    sl = rng.uniform(0.0, 1.0, size=(8, 6)).astype(np.float32)
    np.testing.assert_array_equal(_radiological_axial(sl), np.fliplr(np.rot90(sl)))


def test_radiological_axial_broadcasts_over_rgba_channels() -> None:
    """The transform must keep RGBA overlays intact (no channel scrambling)."""
    rgba = np.zeros((8, 6, 4), dtype=np.float32)
    rgba[..., 0] = 1.0  # solid red
    rgba[..., 3] = 0.5
    transformed = _radiological_axial(rgba)
    assert transformed.shape == (6, 8, 4)
    np.testing.assert_array_equal(transformed[..., 0], 1.0)
    np.testing.assert_array_equal(transformed[..., 3], 0.5)


# ---------------------------------------------------------------------------
# load_metrics_table
# ---------------------------------------------------------------------------


def test_load_metrics_table_skips_subjects_without_sidecar(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0001",
        dice={"WT": 0.9, "TC": 0.85, "ET": 0.8},
        hd95={"WT": 1.0, "TC": 1.0, "ET": 1.0},
    )
    (tmp_path / "sub-0002").mkdir()
    rows = load_metrics_table(tmp_path)
    assert [r.subject_id for r in rows] == ["0001"]


def test_load_metrics_table_rejects_wrong_comparison(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0001" / "metrics"
    sub.mkdir(parents=True)
    (sub / "sub-0001_metrics_dl_vs_gt.json").write_text(
        json.dumps({"comparison": "dl_vs_raidionics", "metrics": {}})
    )
    with pytest.raises(ValueError, match="comparison='dl_vs_gt'"):
        load_metrics_table(tmp_path)


def test_load_metrics_table_rejects_missing_compartment(tmp_path: Path) -> None:
    sub = tmp_path / "sub-0001" / "metrics"
    sub.mkdir(parents=True)
    (sub / "sub-0001_metrics_dl_vs_gt.json").write_text(
        json.dumps(
            {
                "comparison": "dl_vs_gt",
                "metrics": {
                    "WT": {"dice": 0.9, "hd95_mm": 1.0},
                    "TC": {"dice": 0.9, "hd95_mm": 1.0},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="missing compartment 'ET'"):
        load_metrics_table(tmp_path)


def test_load_metrics_table_raises_for_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metrics_table(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# classify_outliers
# ---------------------------------------------------------------------------


def test_classify_outliers_within_band_is_unflagged(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0001",
        dice={"WT": 0.95, "TC": 0.90, "ET": 0.85},
        hd95={"WT": 2.0, "TC": 2.0, "ET": 2.0},
    )
    rows = classify_outliers(tmp_path)
    assert len(rows) == 1
    assert rows[0].flagged is False
    assert rows[0].reasons == ()


def test_classify_outliers_low_dice_flags_with_reason(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0002",
        dice={"WT": 0.92, "TC": 0.30, "ET": 0.85},
        hd95={"WT": 2.0, "TC": 2.0, "ET": 2.0},
    )
    rows = classify_outliers(tmp_path)
    assert rows[0].flagged is True
    assert any("TC_dice_lt" in r for r in rows[0].reasons)
    assert not any("WT_dice_lt" in r for r in rows[0].reasons)


def test_classify_outliers_high_hd95_flags(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0003",
        dice={"WT": 0.95, "TC": 0.90, "ET": 0.85},
        hd95={"WT": 50.0, "TC": 2.0, "ET": 2.0},
    )
    rows = classify_outliers(tmp_path)
    assert rows[0].flagged is True
    assert any(r.startswith("WT_hd95_gt_20mm") for r in rows[0].reasons)


def test_classify_outliers_inf_hd95_flags(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0004",
        dice={"WT": 0.95, "TC": 0.90, "ET": 0.85},
        hd95={"WT": 2.0, "TC": 2.0, "ET": "inf"},
    )
    rows = classify_outliers(tmp_path)
    assert rows[0].flagged is True
    assert "ET_hd95_inf" in rows[0].reasons


def test_classify_outliers_nan_dice_flags(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0005",
        dice={"WT": 0.95, "TC": 0.90, "ET": float("nan")},
        hd95={"WT": 2.0, "TC": 2.0, "ET": 2.0},
    )
    rows = classify_outliers(tmp_path)
    assert rows[0].flagged is True
    assert "ET_dice_nan" in rows[0].reasons


def test_classify_outliers_null_hd95_treated_as_nan(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0006",
        dice={"WT": 0.95, "TC": 0.90, "ET": 0.85},
        hd95={"WT": 2.0, "TC": None, "ET": 2.0},
    )
    rows = classify_outliers(tmp_path)
    assert rows[0].flagged is True
    assert "TC_hd95_nan" in rows[0].reasons


def test_classify_outliers_custom_thresholds_relax_flags(tmp_path: Path) -> None:
    _write_metrics_json(
        tmp_path,
        "0007",
        dice={"WT": 0.80, "TC": 0.70, "ET": 0.65},
        hd95={"WT": 25.0, "TC": 25.0, "ET": 25.0},
    )
    strict = classify_outliers(tmp_path)
    assert strict[0].flagged is True
    relaxed = classify_outliers(
        tmp_path,
        op_point={"WT": 0.5, "TC": 0.5, "ET": 0.5},
        hd95_max_mm=100.0,
    )
    assert relaxed[0].flagged is False


def test_classify_outliers_preserves_subject_order(tmp_path: Path) -> None:
    for sid in ("0010", "0003", "0007"):
        _write_metrics_json(
            tmp_path,
            sid,
            dice={"WT": 0.9, "TC": 0.9, "ET": 0.9},
            hd95={"WT": 1.0, "TC": 1.0, "ET": 1.0},
        )
    rows = classify_outliers(tmp_path)
    assert [r.subject_id for r in rows] == ["0003", "0007", "0010"]


# ---------------------------------------------------------------------------
# brain_bbox_ij
# ---------------------------------------------------------------------------


def test_brain_bbox_ij_centred_block_with_pad() -> None:
    vol = np.zeros((40, 50, 10), dtype=np.float32)
    vol[10:30, 15:35, :] = 1.0
    i_sl, j_sl = brain_bbox_ij(vol, threshold_percentile=5.0, pad=4)
    assert (i_sl.start, i_sl.stop) == (6, 34)
    assert (j_sl.start, j_sl.stop) == (11, 39)


def test_brain_bbox_ij_clamps_to_volume_bounds() -> None:
    vol = np.zeros((10, 10, 5), dtype=np.float32)
    vol[2:8, 2:8, :] = 1.0
    i_sl, j_sl = brain_bbox_ij(vol, threshold_percentile=5.0, pad=10)
    assert (i_sl.start, i_sl.stop) == (0, 10)
    assert (j_sl.start, j_sl.stop) == (0, 10)


def test_brain_bbox_ij_all_zero_falls_back_to_full() -> None:
    vol = np.zeros((8, 12, 6), dtype=np.float32)
    i_sl, j_sl = brain_bbox_ij(vol)
    assert i_sl == slice(None) and j_sl == slice(None)


def test_brain_bbox_ij_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        brain_bbox_ij(np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# tumor_centroid
# ---------------------------------------------------------------------------


def test_tumor_centroid_single_voxel() -> None:
    seg = np.zeros((10, 10, 10), dtype=np.uint8)
    seg[3, 4, 5] = 1
    assert tumor_centroid(seg) == (3, 4, 5)


def test_tumor_centroid_centre_of_block() -> None:
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    seg[5:15, 5:15, 5:15] = 1
    coord = tumor_centroid(seg)
    assert coord in ((9, 9, 9), (10, 10, 10))


def test_tumor_centroid_empty_falls_back_to_volume_centre() -> None:
    seg = np.zeros((10, 12, 14), dtype=np.uint8)
    assert tumor_centroid(seg) == (5, 6, 7)


def test_tumor_centroid_label_filter() -> None:
    seg = np.zeros((10, 10, 10), dtype=np.uint8)
    seg[2, 2, 2] = 1  # NCR
    seg[8, 8, 8] = 4  # ET
    et_only = tumor_centroid(seg, labels=[4])
    assert et_only == (8, 8, 8)
    full = tumor_centroid(seg)  # any non-zero
    assert full == (5, 5, 5)


def test_tumor_centroid_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        tumor_centroid(np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# render_qc_panel
# ---------------------------------------------------------------------------


def _toy_volume(shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=shape).astype(np.float32)


def _toy_label(shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.int16)
    seg[6:10, 6:10, 6:10] = 1  # NCR -> contributes to WT and TC
    seg[7:9, 7:9, 7:9] = 4  # ET inside the core
    seg[10:12, 6:10, 6:10] = 2  # ED -> contributes to WT only
    return seg


def test_render_qc_panel_smoke() -> None:
    t1c = _toy_volume()
    ref = _toy_label()
    dl = _toy_label()
    fig, axes = plt.subplots(1, 3)
    render_qc_panel(
        t1c,
        ref,
        dl,
        coord=(8, 8, 8),
        axes=axes,
        column_titles=True,
        row_label="sub-0001",
    )
    # All three axes have been drawn (at least the background imshow).
    for ax in axes:
        assert len(ax.get_images()) >= 1
    # Column titles are stamped on the first row.
    assert axes[0].get_title() == "T1c"
    assert "reference" in axes[1].get_title().lower()
    assert "DL" in axes[2].get_title()
    # Row label rendered as the leftmost ylabel (so per-subject metrics never
    # collide with column subtitles).
    assert axes[0].get_ylabel() == "sub-0001"
    plt.close(fig)


def test_render_qc_panel_omits_column_titles_on_subsequent_rows() -> None:
    t1c = _toy_volume()
    ref = _toy_label()
    dl = _toy_label()
    fig, axes = plt.subplots(1, 3)
    render_qc_panel(t1c, ref, dl, coord=(8, 8, 8), axes=axes, column_titles=False)
    for ax in axes:
        assert ax.get_title() == ""
    plt.close(fig)


def test_render_qc_panel_rejects_shape_mismatch() -> None:
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        render_qc_panel(
            np.zeros((4, 4, 4)),
            np.zeros((4, 4, 5), dtype=np.int16),
            np.zeros((4, 4, 4), dtype=np.int16),
            coord=(2, 2, 2),
            axes=axes,
        )
    plt.close(fig)


def test_render_qc_panel_rejects_out_of_bounds_coord() -> None:
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="out of bounds"):
        render_qc_panel(
            np.zeros((4, 4, 4)),
            np.zeros((4, 4, 4), dtype=np.int16),
            np.zeros((4, 4, 4), dtype=np.int16),
            coord=(0, 0, 99),
            axes=axes,
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_outlier_grid
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))


def _materialise_subject(
    cohort_root: Path,
    deriv_root: Path,
    sid: str,
    *,
    shape: tuple[int, int, int] = (16, 16, 16),
) -> None:
    sub_in = cohort_root / f"sub-{sid}"
    _write_nifti(sub_in / f"sub-{sid}_T1c_bias.nii.gz", _toy_volume(shape))
    _write_nifti(sub_in / f"sub-{sid}_tumor_segmentation.nii.gz", _toy_label(shape))
    sub_out = deriv_root / f"sub-{sid}"
    _write_nifti(sub_out / "seg_dl" / f"sub-{sid}_seg_brats3_dl.nii.gz", _toy_label(shape))


def test_render_outlier_grid_empty_returns_placeholder(tmp_path: Path) -> None:
    fig = render_outlier_grid(
        [],
        cohort_root=tmp_path,
        deriv_root=tmp_path,
    )
    assert fig is not None


def test_render_outlier_grid_writes_pdf(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort"
    deriv = tmp_path / "deriv"
    _materialise_subject(cohort, deriv, "0001")
    _write_metrics_json(
        deriv,
        "0001",
        dice={"WT": 0.5, "TC": 0.5, "ET": 0.4},
        hd95={"WT": 25.0, "TC": 1.0, "ET": 1.0},
    )
    rows = classify_outliers(deriv)
    out = tmp_path / "qc.pdf"
    result = render_outlier_grid(
        rows,
        cohort_root=cohort,
        deriv_root=deriv,
        out_path=out,
    )
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_render_outlier_grid_handles_missing_inputs(tmp_path: Path) -> None:
    deriv = tmp_path / "deriv"
    _write_metrics_json(
        deriv,
        "0001",
        dice={"WT": 0.5, "TC": 0.5, "ET": 0.4},
        hd95={"WT": 25.0, "TC": 1.0, "ET": 1.0},
    )
    rows = classify_outliers(deriv)
    fig = render_outlier_grid(
        rows,
        cohort_root=tmp_path / "no-such-cohort-root",
        deriv_root=deriv,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# render_single_subject_panel
# ---------------------------------------------------------------------------


def test_render_single_subject_panel_writes_pdf(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort"
    deriv = tmp_path / "deriv"
    _materialise_subject(cohort, deriv, "0001")
    _write_metrics_json(
        deriv,
        "0001",
        dice={"WT": 0.5, "TC": 0.5, "ET": 0.4},
        hd95={"WT": 25.0, "TC": 1.0, "ET": 1.0},
    )
    rows = classify_outliers(deriv)
    out = tmp_path / "single.pdf"
    result = render_single_subject_panel(
        "0001",
        cohort_root=cohort,
        deriv_root=deriv,
        out_path=out,
        row=rows[0],
        caption="exemplar caption",
    )
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_render_single_subject_panel_returns_figure_without_row(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort"
    deriv = tmp_path / "deriv"
    _materialise_subject(cohort, deriv, "0007")
    fig = render_single_subject_panel(
        "0007",
        cohort_root=cohort,
        deriv_root=deriv,
    )
    assert fig is not None
    plt.close(fig)


def test_render_single_subject_panel_raises_on_missing_subject(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_single_subject_panel(
            "9999",
            cohort_root=tmp_path / "cohort",
            deriv_root=tmp_path / "deriv",
        )
