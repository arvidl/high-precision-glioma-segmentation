"""Unit tests for ``scripts/compute_hitplot_cohort50.py`` (PR-7f).

We import the script as a module via ``importlib`` rather than via the
package path because ``scripts/`` is intentionally not on ``sys.path``;
the script is a CLI shim. Helpers, the artefact contract, source
dispatch, and the end-to-end ``_real_process_subject_source`` runner
are all covered here using synthetic NIfTIs so the test does not
depend on torch / monai weights / a real UCSF-PDGM cohort on disk.

and is the prerequisite for PR-7i's Hit-Plot agreement panel.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from hpgs.segment.unified import PROB_CHANNEL_ORDER


def _load_runner_module():
    """Load ``scripts/compute_hitplot_cohort50.py`` as a module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "compute_hitplot_cohort50.py"
    spec = importlib.util.spec_from_file_location(
        "compute_hitplot_cohort50_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# _planned_hitplot_artefacts: Section 4 of docs/scope_pr7.md
# ---------------------------------------------------------------------------


def test_planned_hitplot_artefacts_keys_cover_section_4_contract(tmp_path: Path) -> None:
    paths = mod._planned_hitplot_artefacts(tmp_path / "deriv", "0005")
    expected = {
        # parcellation inputs
        "parc_wmparc",
        "parc_lut",
        # DL (PR-7d) inputs
        "seg_dl_label_map",
        "seg_dl_probs",
        "seg_dl_sidecar",
        # baseline (PR-7e) inputs
        "seg_raidionics",
        "seg_segmentglioma",
        "seg_tumorsynth",
        # hitplot outputs
        "hp_dl",
        "hp_dl_prob",
        "hp_gt",
        "hp_raidionics",
        "hp_segmentglioma",
        "hp_tumorsynth",
    }
    assert set(paths) == expected


def test_planned_hitplot_artefacts_layout(tmp_path: Path) -> None:
    paths = mod._planned_hitplot_artefacts(tmp_path / "deriv", "0005")
    sub = tmp_path / "deriv" / "sub-0005"
    assert paths["parc_wmparc"] == sub / "parcellation" / "sub-0005_wmparc_native.nii.gz"
    assert paths["parc_lut"] == sub / "parcellation" / "sub-0005_wmparc_lut.json"
    assert paths["hp_dl"] == sub / "hitplot" / "sub-0005_hitplot_dl.csv"
    assert paths["hp_dl_prob"] == sub / "hitplot" / "sub-0005_hitplot_dl_prob.csv"
    assert paths["hp_gt"] == sub / "hitplot" / "sub-0005_hitplot_gt.csv"
    assert paths["hp_raidionics"] == sub / "hitplot" / "sub-0005_hitplot_raidionics.csv"
    assert paths["hp_segmentglioma"] == sub / "hitplot" / "sub-0005_hitplot_segmentglioma.csv"
    assert paths["hp_tumorsynth"] == sub / "hitplot" / "sub-0005_hitplot_tumorsynth.csv"
    assert paths["seg_raidionics"] == sub / "seg_raidionics" / "sub-0005_seg_raidionics.nii.gz"
    assert (
        paths["seg_segmentglioma"]
        == sub / "seg_segmentglioma" / "sub-0005_seg_segmentglioma.nii.gz"
    )
    assert paths["seg_tumorsynth"] == sub / "seg_tumorsynth" / "sub-0005_seg_tumorsynth.nii.gz"


def test_planned_hitplot_artefacts_normalises_subject_id(tmp_path: Path) -> None:
    a = mod._planned_hitplot_artefacts(tmp_path / "deriv", "0005")
    b = mod._planned_hitplot_artefacts(tmp_path / "deriv", "sub-0005")
    assert a == b


def test_hitplot_output_path_for_each_source(tmp_path: Path) -> None:
    artefacts = mod._planned_hitplot_artefacts(tmp_path / "deriv", "0005")
    for src in mod.SOURCES:
        out = mod._hitplot_output_path(artefacts, src)
        assert out.name.startswith("sub-0005_hitplot_")
        assert out.name.endswith(".csv")
        assert src in out.name or (src == "dl" and "hitplot_dl.csv" in out.name)


# ---------------------------------------------------------------------------
# _voxel_volume_mm3
# ---------------------------------------------------------------------------


def test_voxel_volume_mm3_isotropic_1mm() -> None:
    affine = np.eye(4)
    assert mod._voxel_volume_mm3(affine) == pytest.approx(1.0)


def test_voxel_volume_mm3_anisotropic() -> None:
    affine = np.diag([2.0, 0.5, 1.5, 1.0]).astype(float)
    assert mod._voxel_volume_mm3(affine) == pytest.approx(2.0 * 0.5 * 1.5)


# ---------------------------------------------------------------------------
# _select_subjects
# ---------------------------------------------------------------------------


def test_select_subjects_default_returns_full_cohort() -> None:
    cohort = ["0005", "0026", "0068"]
    assert mod._select_subjects(cohort, None) == cohort


def test_select_subjects_normalises_sub_prefix() -> None:
    cohort = ["0005", "0026"]
    assert mod._select_subjects(cohort, ["sub-0005", "0026"]) == ["0005", "0026"]


def test_select_subjects_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="not in the locked cohort"):
        mod._select_subjects(["0005"], ["9999"])


def test_select_subjects_dedup_preserves_first_occurrence() -> None:
    cohort = ["0005", "0026"]
    assert mod._select_subjects(cohort, ["0005", "sub-0005", "0026"]) == ["0005", "0026"]


def test_select_subjects_accepts_whitespace_packed_token() -> None:
    cohort = ["0005", "0026", "0068"]
    # Mirrors the Makefile shim: SUBJECTS="0005 0068" -> single CLI arg.
    assert mod._select_subjects(cohort, ["0005 0068"]) == ["0005", "0068"]


# ---------------------------------------------------------------------------
# _select_sources
# ---------------------------------------------------------------------------


def test_select_sources_default_returns_all_sources_in_contract_order() -> None:
    assert mod._select_sources(None) == list(mod.SOURCES)


def test_select_sources_all_alias_expands_to_full_list() -> None:
    assert mod._select_sources(["all"]) == list(mod.SOURCES)


def test_select_sources_filter_preserves_request_order_and_dedup() -> None:
    assert mod._select_sources(["gt", "dl", "dl", "gt"]) == ["gt", "dl"]


def test_select_sources_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="not in"):
        mod._select_sources(["bogus"])


def test_select_sources_accepts_whitespace_packed_token() -> None:
    assert mod._select_sources(["dl gt"]) == ["dl", "gt"]


def test_select_sources_empty_request_falls_back_to_default() -> None:
    assert mod._select_sources([]) == list(mod.SOURCES)


# ---------------------------------------------------------------------------
# _read_wmparc_lut
# ---------------------------------------------------------------------------


def test_read_wmparc_lut_coerces_string_keys_to_int(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.json"
    lut_path.write_text(json.dumps({"1": "Left-Cortex", "2": "Right-Cortex"}))
    out = mod._read_wmparc_lut(lut_path)
    assert out == {1: "Left-Cortex", 2: "Right-Cortex"}


def test_read_wmparc_lut_rejects_empty(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.json"
    lut_path.write_text("{}")
    with pytest.raises(ValueError, match="empty LUT"):
        mod._read_wmparc_lut(lut_path)


def test_read_wmparc_lut_rejects_non_integer_key(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.json"
    lut_path.write_text(json.dumps({"foo": "not-a-label-id"}))
    with pytest.raises(ValueError, match="non-integer label key"):
        mod._read_wmparc_lut(lut_path)


def test_read_wmparc_lut_rejects_non_object_top_level(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.json"
    lut_path.write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="expected top-level JSON object"):
        mod._read_wmparc_lut(lut_path)


# ---------------------------------------------------------------------------
# _read_prob_channel_order
# ---------------------------------------------------------------------------


def test_read_prob_channel_order_from_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "seg.json"
    sidecar.write_text(json.dumps({"probability_channel_order": ["WT", "TC", "ET"]}))
    assert mod._read_prob_channel_order(sidecar) == ("WT", "TC", "ET")


def test_read_prob_channel_order_fallback_when_sidecar_missing(tmp_path: Path) -> None:
    assert mod._read_prob_channel_order(tmp_path / "no-such.json") == tuple(PROB_CHANNEL_ORDER)


def test_read_prob_channel_order_fallback_when_sidecar_malformed(tmp_path: Path) -> None:
    sidecar = tmp_path / "seg.json"
    sidecar.write_text("{not valid json")
    assert mod._read_prob_channel_order(sidecar) == tuple(PROB_CHANNEL_ORDER)


def test_read_prob_channel_order_fallback_when_key_missing(tmp_path: Path) -> None:
    sidecar = tmp_path / "seg.json"
    sidecar.write_text(json.dumps({"backend": "dummy"}))
    assert mod._read_prob_channel_order(sidecar) == tuple(PROB_CHANNEL_ORDER)


# ---------------------------------------------------------------------------
# Synthetic cohort fixtures
# ---------------------------------------------------------------------------


SHAPE: tuple[int, int, int] = (20, 20, 20)
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)


def _write_parcellation(deriv_root: Path, subject_id: str) -> None:
    """Write a 3-region synthetic wmparc NIfTI + LUT for one subject."""
    sub = deriv_root / f"sub-{subject_id}"
    parc_dir = sub / "parcellation"
    parc_dir.mkdir(parents=True, exist_ok=True)
    parc = np.zeros(SHAPE, dtype=np.int16)
    parc[:10, :, :] = 1
    parc[10:, :10, :] = 2
    parc[10:, 10:, :] = 3
    nib.save(
        nib.Nifti1Image(parc, AFFINE),
        str(parc_dir / f"sub-{subject_id}_wmparc_native.nii.gz"),
    )
    (parc_dir / f"sub-{subject_id}_wmparc_lut.json").write_text(
        json.dumps({"1": "Region-A", "2": "Region-B", "3": "Region-C"}),
    )


def _brats_label_map(shift: int = 0) -> np.ndarray:
    """Small NCR/ED/ET label map; ``shift`` moves the tumor along X."""
    lm = np.zeros(SHAPE, dtype=np.int16)
    lm[5 + shift : 9 + shift, 5:9, 5:9] = 1  # NCR
    lm[7 + shift : 11 + shift, 7:11, 7:11] = 4  # ET (overlaps NCR -> ET)
    lm[12 + shift : 15 + shift, 12:15, 12:15] = 2  # ED
    return lm


def _write_dl_seg(deriv_root: Path, subject_id: str, *, shift: int = 0) -> None:
    sub = deriv_root / f"sub-{subject_id}" / "seg_dl"
    sub.mkdir(parents=True, exist_ok=True)
    lm = _brats_label_map(shift=shift)
    nib.save(
        nib.Nifti1Image(lm.astype(np.uint8), AFFINE),
        str(sub / f"sub-{subject_id}_seg_brats3_dl.nii.gz"),
    )
    # Synthetic probs (X, Y, Z, C) in channel order (TC, WT, ET). Values
    # are hard 0/1 derived from the label map so the probabilistic
    # matrix should numerically agree with the deterministic one (std = 0).
    probs = np.zeros((*SHAPE, 3), dtype=np.float32)
    probs[..., 0] = np.isin(lm, [1, 4]).astype(np.float32)  # TC
    probs[..., 1] = np.isin(lm, [1, 2, 4]).astype(np.float32)  # WT
    probs[..., 2] = (lm == 4).astype(np.float32)  # ET
    nib.save(
        nib.Nifti1Image(probs, AFFINE),
        str(sub / f"sub-{subject_id}_seg_brats3_dl_probs.nii.gz"),
    )
    (sub / f"sub-{subject_id}_seg_brats3_dl.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "subject_id": subject_id,
                "backend": "dummy",
                "probability_channel_order": ["TC", "WT", "ET"],
            },
        ),
    )


def _write_baseline_seg(
    deriv_root: Path,
    subject_id: str,
    baseline: str,
    *,
    shift: int = 0,
) -> None:
    """Write ``seg_<baseline>/sub-XXXX_seg_<baseline>.nii.gz`` (PR-7e)."""
    sub = deriv_root / f"sub-{subject_id}" / f"seg_{baseline}"
    sub.mkdir(parents=True, exist_ok=True)
    lm = _brats_label_map(shift=shift)
    nib.save(
        nib.Nifti1Image(lm.astype(np.uint8), AFFINE),
        str(sub / f"sub-{subject_id}_seg_{baseline}.nii.gz"),
    )


def _write_gt_nifti(cohort_root: Path, subject_id: str, *, shift: int = 0) -> None:
    """Write the minimal GT NIfTI resolvable by ``resolve_subject_inputs``."""
    sub = cohort_root / f"sub-{subject_id}"
    sub.mkdir(parents=True, exist_ok=True)
    lm = _brats_label_map(shift=shift)
    nib.save(
        nib.Nifti1Image(lm.astype(np.int16), AFFINE),
        str(sub / f"sub-{subject_id}_tumor_segmentation.nii.gz"),
    )


def _full_synthetic_subject(
    cohort_root: Path,
    deriv_root: Path,
    subject_id: str = "0005",
) -> None:
    """Populate every input the PR-7f runner reads for one subject."""
    _write_parcellation(deriv_root, subject_id)
    _write_dl_seg(deriv_root, subject_id, shift=0)
    _write_baseline_seg(deriv_root, subject_id, "raidionics", shift=1)
    _write_baseline_seg(deriv_root, subject_id, "segmentglioma", shift=-1)
    _write_baseline_seg(deriv_root, subject_id, "tumorsynth", shift=2)
    _write_gt_nifti(cohort_root, subject_id, shift=0)


# ---------------------------------------------------------------------------
# _real_process_subject_source: happy path per source
# ---------------------------------------------------------------------------


def test_real_process_dl_writes_hitplot_dl_csv(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )

    assert rec["status"] == "ok"
    out = Path(rec["out_csv"])
    assert out.is_file()
    df = pd.read_csv(out)
    # Long-format schema from compartment_region_matrix.
    expected_cols = {
        "label",
        "name",
        "compartment",
        "parcel_voxels",
        "compartment_voxels",
        "overlap_voxels",
        "overlap_volume_mm3",
        "pct_of_parcel",
        "pct_of_compartment",
    }
    assert expected_cols.issubset(df.columns)
    # 3 regions x 3 compartments = 9 rows.
    assert len(df) == 9
    assert set(df["compartment"].unique()) == {"WT", "TC", "ET"}
    assert set(df["label"].unique()) == {1, 2, 3}


def test_real_process_dl_prob_writes_probabilistic_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "dl_prob",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )

    assert rec["status"] == "ok"
    out = Path(rec["out_csv"])
    df = pd.read_csv(out)
    expected_cols = {
        "label",
        "name",
        "compartment",
        "parcel_voxels",
        "overlap_voxels_mean",
        "overlap_voxels_std",
        "overlap_volume_mm3_mean",
        "overlap_volume_mm3_std",
        "pct_of_parcel_mean",
        "pct_of_parcel_std",
    }
    assert expected_cols.issubset(df.columns)
    # Hard 0/1 probs -> Bernoulli variance p*(1-p) = 0 everywhere.
    assert np.allclose(df["overlap_voxels_std"], 0.0)


def test_real_process_dl_prob_agrees_with_dl_hard_on_0_1_probs(tmp_path: Path) -> None:
    """The probabilistic mean must equal the deterministic count when
    the probs are already hard {0, 1}. This is the key cross-check the
    agreement panel (PR-7i) relies on."""
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    for src in ("dl", "dl_prob"):
        mod._real_process_subject_source(
            "0005",
            src,
            cohort_root=cohort_root,
            deriv_root=deriv_root,
            skip_existing=False,
            require_parcellation=True,
        )

    artefacts = mod._planned_hitplot_artefacts(deriv_root, "0005")
    df_dl = pd.read_csv(artefacts["hp_dl"])
    df_prob = pd.read_csv(artefacts["hp_dl_prob"])
    # Column names are disjoint between the two schemas, so a plain
    # inner join on (label, compartment) suffices.
    merged = df_dl.merge(df_prob, on=["label", "compartment"])
    np.testing.assert_allclose(
        merged["overlap_voxels"].to_numpy(float),
        merged["overlap_voxels_mean"].to_numpy(float),
        atol=1e-6,
    )


def test_real_process_gt_writes_hitplot_gt_csv(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "gt",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )

    assert rec["status"] == "ok"
    out = Path(rec["out_csv"])
    assert out.is_file()
    assert "hitplot_gt.csv" in out.name
    df = pd.read_csv(out)
    assert len(df) == 9  # 3 regions x 3 compartments


def test_real_process_raidionics_writes_hitplot_csv(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "ok"
    assert Path(rec["out_csv"]).is_file()


def test_real_process_segmentglioma_writes_hitplot_csv(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "segmentglioma",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "ok"
    assert Path(rec["out_csv"]).is_file()


def test_real_process_tumorsynth_writes_hitplot_csv(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "tumorsynth",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "ok"
    assert Path(rec["out_csv"]).is_file()


# ---------------------------------------------------------------------------
# Missing / malformed inputs
# ---------------------------------------------------------------------------


def test_real_process_missing_parcellation_returns_missing_input(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # DL seg present but no parcellation at all.
    _write_dl_seg(deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )

    assert rec["status"] == "missing_input"
    assert "parcellation" in rec["reason"].lower()
    assert rec["require_parcellation"] is True


def test_real_process_missing_parcellation_with_require_off_still_reports(
    tmp_path: Path,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_dl_seg(deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=False,
    )
    # Per-source record is the same shape; the caller (_run_real)
    # decides whether to demote to a warning.
    assert rec["status"] == "missing_input"
    assert rec["require_parcellation"] is False


def test_real_process_missing_dl_seg_returns_missing_input(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_parcellation(deriv_root, "0005")

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "missing_input"
    assert "source NIfTI absent" in rec["reason"]


def test_real_process_missing_dl_probs_returns_missing_input(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_parcellation(deriv_root, "0005")
    # Write DL label map but not probs.
    _write_dl_seg(deriv_root, "0005")
    artefacts = mod._planned_hitplot_artefacts(deriv_root, "0005")
    artefacts["seg_dl_probs"].unlink()

    rec = mod._real_process_subject_source(
        "0005",
        "dl_prob",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "missing_input"
    assert "probs" in rec["reason"].lower()


def test_real_process_missing_gt_returns_missing_input(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_parcellation(deriv_root, "0005")
    # No GT written.

    rec = mod._real_process_subject_source(
        "0005",
        "gt",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    # resolve_subject_inputs raises FileNotFoundError -> caught as error.
    assert rec["status"] == "error"


def test_real_process_shape_mismatch_is_error(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_parcellation(deriv_root, "0005")
    # DL seg with mismatched shape.
    sub = deriv_root / "sub-0005" / "seg_dl"
    sub.mkdir(parents=True, exist_ok=True)
    lm = np.zeros((10, 10, 10), dtype=np.uint8)
    nib.save(
        nib.Nifti1Image(lm, AFFINE),
        str(sub / "sub-0005_seg_brats3_dl.nii.gz"),
    )

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
        require_parcellation=True,
    )
    assert rec["status"] == "error"
    assert "shape" in rec["reason"].lower() or "hitplot producer failed" in rec["reason"]


# ---------------------------------------------------------------------------
# --skip-existing short-circuit
# ---------------------------------------------------------------------------


def test_skip_existing_short_circuits_without_reading_inputs(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # Pre-create the output CSV with a sentinel string.
    artefacts = mod._planned_hitplot_artefacts(deriv_root, "0005")
    artefacts["hp_dl"].parent.mkdir(parents=True, exist_ok=True)
    artefacts["hp_dl"].write_text("sentinel,do,not,overwrite\n")

    rec = mod._real_process_subject_source(
        "0005",
        "dl",
        # cohort_root does not exist; skip-existing must not touch it.
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=True,
        require_parcellation=True,
    )
    assert rec["status"] == "skipped"
    assert "skip-existing" in rec["reason"]
    # The sentinel must survive.
    assert artefacts["hp_dl"].read_text().startswith("sentinel")


# ---------------------------------------------------------------------------
# CLI arg parsing: --dry-run / --no-dry-run defaults
# ---------------------------------------------------------------------------


def test_cli_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["compute_hitplot_cohort50.py"])
    args = mod._parse_args()
    assert args.dry_run is True
    assert args.skip_existing is False
    assert args.require_parcellation is True


def test_cli_no_dry_run_flag_flips_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["compute_hitplot_cohort50.py", "--no-dry-run"])
    args = mod._parse_args()
    assert args.dry_run is False


def test_cli_no_require_parcellation_flag_flips_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_hitplot_cohort50.py", "--no-require-parcellation"],
    )
    args = mod._parse_args()
    assert args.require_parcellation is False


def test_cli_source_filter_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--source",
            "dl",
            "--source",
            "gt",
        ],
    )
    args = mod._parse_args()
    assert args.sources == ["dl", "gt"]


# ---------------------------------------------------------------------------
# SOURCES tuple contract
# ---------------------------------------------------------------------------


def test_sources_tuple_matches_section_4_contract() -> None:
    # Order and content must mirror Section 4 of docs/scope_pr7.md.
    assert mod.SOURCES == ("dl", "dl_prob", "gt", "raidionics", "segmentglioma", "tumorsynth")


def test_probabilistic_sources_is_exactly_dl_prob() -> None:
    assert mod._PROBABILISTIC_SOURCES == frozenset({"dl_prob"})


# ---------------------------------------------------------------------------
# _run_real end-to-end (smoke) via a monkey-patched cohort loader
# ---------------------------------------------------------------------------


def _fake_cohort_metadata(subject_ids: list[str]):
    """Minimal stand-in for ``CohortMetadata`` with just ``.df.index`` and ``.n``."""

    class _FakeCM:
        def __init__(self, sids: list[str]) -> None:
            self.df = pd.DataFrame(index=sids)
            self.n = len(sids)
            self.seed = 0

    return _FakeCM(subject_ids)


def test_run_real_produces_csvs_for_single_subject_all_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--no-dry-run",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
            "--subject",
            "0005",
        ],
    )
    exit_code = mod.main()
    assert exit_code == 0
    artefacts = mod._planned_hitplot_artefacts(deriv_root, "0005")
    for key in (
        "hp_dl",
        "hp_dl_prob",
        "hp_gt",
        "hp_raidionics",
        "hp_segmentglioma",
        "hp_tumorsynth",
    ):
        assert artefacts[key].is_file(), f"{key} missing"


def test_run_real_returns_1_when_a_pair_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # Provide parcellation but mismatched DL seg (shape error).
    _write_parcellation(deriv_root, "0005")
    sub = deriv_root / "sub-0005" / "seg_dl"
    sub.mkdir(parents=True, exist_ok=True)
    bad_lm = np.zeros((10, 10, 10), dtype=np.uint8)
    nib.save(
        nib.Nifti1Image(bad_lm, AFFINE),
        str(sub / "sub-0005_seg_brats3_dl.nii.gz"),
    )

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--no-dry-run",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
            "--subject",
            "0005",
            "--source",
            "dl",
        ],
    )
    exit_code = mod.main()
    assert exit_code == 1


def test_run_real_returns_2_when_cohort_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args, **_kwargs):
        raise ValueError("synthetic cohort load failure")

    monkeypatch.setattr(mod, "load_cohort_metadata", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--no-dry-run",
            "--cohort-root",
            str(tmp_path / "cohort"),
            "--deriv-root",
            str(tmp_path / "deriv"),
        ],
    )
    assert mod.main() == 2


def test_run_real_returns_0_when_only_missing_non_parcellation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subject with parcellation but without the Raidionics seg should
    report ``missing_input`` without failing the whole run --- Raidionics
    is per-subject optional (Q1 = B only)."""
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_parcellation(deriv_root, "0005")
    # No seg_raidionics written.

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--no-dry-run",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
            "--subject",
            "0005",
            "--source",
            "raidionics",
        ],
    )
    exit_code = mod.main()
    assert exit_code == 0


def test_run_dry_prints_artefact_plan_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _full_synthetic_subject(cohort_root, deriv_root, "0005")

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
        ],
    )
    exit_code = mod.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "sub-0005" in out
    # Dry-run must not have created any hitplot CSV.
    artefacts = mod._planned_hitplot_artefacts(deriv_root, "0005")
    assert not artefacts["hp_dl"].exists()


def test_run_dry_prints_parc_missing_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # No parcellation, no cohort subject.
    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_hitplot_cohort50.py",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
        ],
    )
    mod.main()
    out = capsys.readouterr().out
    assert "parc=MISSING" in out
    assert "gt=MISSING" in out
