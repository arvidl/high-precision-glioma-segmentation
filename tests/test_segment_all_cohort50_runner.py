"""Unit tests for ``scripts/segment_all_cohort50.py`` (PR-7d).

We import the script as a module via ``importlib`` rather than via the
package path because ``scripts/`` is intentionally not on ``sys.path``;
the script is a CLI shim. Helpers, JSON sidecar schemas, the artefact
contract, and the end-to-end ``_real_process_subject`` runner are all
covered here using the ``dummy`` backend so the test does not depend
on torch / monai weights / network access.

segmentation engine over n=50), R2.2 (Table 2), and R2.5 / R2.10
(table producers consume the metrics JSONs validated here).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _load_runner_module():
    """Load ``scripts/segment_all_cohort50.py`` as a module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "segment_all_cohort50.py"
    spec = importlib.util.spec_from_file_location(
        "segment_all_cohort50_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# _planned_artefacts: Section 4 of docs/scope_pr7.md
# ---------------------------------------------------------------------------


def test_planned_artefacts_keys_match_section_4_contract(tmp_path: Path) -> None:
    paths = mod._planned_artefacts(tmp_path / "deriv", "0005")
    expected = {
        # seg_dl/
        "seg_dl_label_map",
        "seg_dl_probs",
        "seg_dl_sidecar",
        # parcellation/
        "parc_wmparc",
        "parc_lut",
        # hitplot/
        "hp_dl",
        "hp_dl_prob",
        "hp_gt",
        "hp_raidionics",
        "hp_segmentglioma",
        "hp_tumorsynth",
        # metrics/
        "metr_dl_vs_gt",
        "metr_dl_vs_raidionics",
        "metr_dl_vs_segmentglioma",
        "metr_dl_vs_tumorsynth",
    }
    assert set(paths) == expected


def test_planned_artefacts_layout_uses_section_4_subdirectories(tmp_path: Path) -> None:
    paths = mod._planned_artefacts(tmp_path / "deriv", "0005")
    sub = tmp_path / "deriv" / "sub-0005"
    assert paths["seg_dl_label_map"] == sub / "seg_dl" / "sub-0005_seg_brats3_dl.nii.gz"
    assert paths["seg_dl_probs"] == sub / "seg_dl" / "sub-0005_seg_brats3_dl_probs.nii.gz"
    assert paths["seg_dl_sidecar"] == sub / "seg_dl" / "sub-0005_seg_brats3_dl.json"
    assert paths["metr_dl_vs_gt"] == sub / "metrics" / "sub-0005_metrics_dl_vs_gt.json"
    assert paths["hp_dl_prob"] == sub / "hitplot" / "sub-0005_hitplot_dl_prob.csv"
    assert paths["parc_wmparc"] == sub / "parcellation" / "sub-0005_wmparc_native.nii.gz"


def test_planned_artefacts_normalises_subject_id(tmp_path: Path) -> None:
    a = mod._planned_artefacts(tmp_path / "deriv", "0005")
    b = mod._planned_artefacts(tmp_path / "deriv", "sub-0005")
    assert a == b


# ---------------------------------------------------------------------------
# _voxel_volume_mm3 / _voxel_spacing_mm
# ---------------------------------------------------------------------------


def test_voxel_helpers_isotropic_1mm() -> None:
    affine = np.eye(4)
    assert mod._voxel_volume_mm3(affine) == pytest.approx(1.0)
    assert mod._voxel_spacing_mm(affine) == (1.0, 1.0, 1.0)


def test_voxel_helpers_anisotropic() -> None:
    affine = np.diag([2.0, 0.5, 1.5, 1.0]).astype(float)
    assert mod._voxel_volume_mm3(affine) == pytest.approx(2.0 * 0.5 * 1.5)
    spacing = mod._voxel_spacing_mm(affine)
    assert spacing == pytest.approx((2.0, 0.5, 1.5))


# ---------------------------------------------------------------------------
# _json_safe: NaN -> None, +/-Inf -> "inf"
# ---------------------------------------------------------------------------


def test_json_safe_passes_through_finite_numbers() -> None:
    assert mod._json_safe(0.5) == 0.5
    assert mod._json_safe(0.0) == 0.0
    assert mod._json_safe(-3.14) == pytest.approx(-3.14)


def test_json_safe_replaces_nan_with_none() -> None:
    assert mod._json_safe(float("nan")) is None


def test_json_safe_replaces_inf_with_string() -> None:
    assert mod._json_safe(float("inf")) == "inf"
    assert mod._json_safe(-float("inf")) == "inf"


def test_json_safe_output_is_json_serialisable() -> None:
    record = {
        "dice": mod._json_safe(0.5),
        "hd95_mm": mod._json_safe(float("inf")),
        "ve_mm3": mod._json_safe(float("nan")),
    }
    encoded = json.dumps(record)
    decoded = json.loads(encoded)
    assert decoded == {"dice": 0.5, "hd95_mm": "inf", "ve_mm3": None}


# ---------------------------------------------------------------------------
# _per_compartment_full_metrics
# ---------------------------------------------------------------------------


def _label_map_with_all_classes() -> np.ndarray:
    lm = np.zeros((20, 20, 20), dtype=np.int16)
    lm[2:8, 2:8, 2:8] = 1  # NCR -> TC, WT
    lm[6:12, 6:12, 6:12] = 4  # ET -> ET, TC, WT
    lm[12:16, 12:16, 12:16] = 2  # ED -> WT
    return lm


def test_per_compartment_metrics_identical_masks_perfect_scores() -> None:
    lm = _label_map_with_all_classes()
    out = mod._per_compartment_full_metrics(
        lm,
        lm,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        voxel_volume_mm3=1.0,
    )
    for comp in ("WT", "TC", "ET"):
        assert out[comp]["dice"] == pytest.approx(1.0)
        assert out[comp]["hd95_mm"] == pytest.approx(0.0)
        assert out[comp]["abs_volumetric_error_mm3"] == pytest.approx(0.0)
        assert out[comp]["sensitivity"] == pytest.approx(1.0)
        assert out[comp]["specificity"] == pytest.approx(1.0)


def test_per_compartment_metrics_empty_reference_returns_json_safe() -> None:
    pred = _label_map_with_all_classes()
    ref = np.zeros_like(pred)
    out = mod._per_compartment_full_metrics(
        pred,
        ref,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        voxel_volume_mm3=1.0,
    )
    for comp in ("WT", "TC", "ET"):
        # Pred non-empty, ref empty:
        # Dice = 2*0/(pred_sum + 0) = 0.0 (denom > 0, well-defined)
        assert out[comp]["dice"] == pytest.approx(0.0)
        # HD95 = inf when exactly one mask is empty (BraTS convention)
        assert out[comp]["hd95_mm"] == "inf"
        # |VE| = pred volume - 0 > 0
        assert isinstance(out[comp]["abs_volumetric_error_mm3"], float)
        assert out[comp]["abs_volumetric_error_mm3"] > 0
        # sensitivity = (pred & gt).sum() / gt.sum() -> 0/0 -> nan -> None
        assert out[comp]["sensitivity"] is None
        # specificity well-defined when there is any background voxel
        assert out[comp]["specificity"] is not None


def test_per_compartment_metrics_full_record_is_json_round_trippable() -> None:
    """The metrics dict must survive ``json.dumps`` without TypeError."""
    pred = _label_map_with_all_classes()
    ref = np.zeros_like(pred)
    out = mod._per_compartment_full_metrics(
        pred,
        ref,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        voxel_volume_mm3=1.0,
    )
    s = json.dumps(out)
    decoded = json.loads(s)
    assert set(decoded) == {"WT", "TC", "ET"}


# ---------------------------------------------------------------------------
# _select_subjects
# ---------------------------------------------------------------------------


def test_select_subjects_default_returns_full_cohort() -> None:
    cohort = ["0005", "0026", "0068"]
    assert mod._select_subjects(cohort, None) == cohort


def test_select_subjects_filter_normalises_sub_prefix() -> None:
    cohort = ["0005", "0026", "0068"]
    out = mod._select_subjects(cohort, ["0005", "sub-0026"])
    assert out == ["0005", "0026"]


def test_select_subjects_rejects_unknown_id() -> None:
    with pytest.raises(SystemExit, match="not in cohort YAML"):
        mod._select_subjects(["0005"], ["9999"])


# ---------------------------------------------------------------------------
# _real_process_subject end-to-end with the dummy backend
# ---------------------------------------------------------------------------


def _write_synthetic_cohort_subject(
    cohort_root: Path,
    subject_id: str,
    *,
    shape: tuple[int, int, int] = (40, 40, 40),
    seed: int = 20260415,
) -> None:
    """Write the four channels + tumor_segmentation NIfTIs for one subject.

    The synthetic volumes are random-positive (so the brain mask
    ``volume > 0`` is non-empty in :func:`hpgs.segment.unified._zscore_brain`)
    and the reference tumor mask is a small NCR/ED/ET label cube --- enough
    to exercise the full sub-region split in
    ``_per_compartment_full_metrics``.
    """
    sub_dir = cohort_root / f"sub-{subject_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)
    rng = np.random.default_rng(seed)
    for ch in ("T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"):
        data = (rng.random(shape, dtype=np.float32) * 100.0 + 1.0).astype(np.float32)
        nib.save(
            nib.Nifti1Image(data, affine),
            str(sub_dir / f"sub-{subject_id}_{ch}.nii.gz"),
        )
    ref = np.zeros(shape, dtype=np.int16)
    ref[10:16, 10:16, 10:16] = 1  # NCR
    ref[15:20, 15:20, 15:20] = 4  # ET (overlaps NCR -> ET wins)
    ref[22:28, 22:28, 22:28] = 2  # ED
    nib.save(
        nib.Nifti1Image(ref, affine),
        str(sub_dir / f"sub-{subject_id}_tumor_segmentation.nii.gz"),
    )


def test_real_process_subject_writes_section_4_artefacts(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    record = mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )

    # In-memory record sanity.
    assert record["subject_id"] == "0005"
    assert record["skipped"] is False
    assert record["backend"] == "dummy"
    assert set(record["metrics"]) == {"WT", "TC", "ET"}

    # On-disk artefacts: every PR-7d-written path exists.
    artefacts = mod._planned_artefacts(deriv_root, "0005")
    for key in ("seg_dl_label_map", "seg_dl_probs", "seg_dl_sidecar", "metr_dl_vs_gt"):
        assert artefacts[key].is_file(), f"missing artefact {key}: {artefacts[key]}"


def test_real_process_subject_seg_sidecar_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )
    sidecar_path = mod._planned_artefacts(deriv_root, "0005")["seg_dl_sidecar"]
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == mod.SEG_SIDECAR_SCHEMA_VERSION
    assert sidecar["subject_id"] == "0005"
    assert sidecar["backend"] == "dummy"
    assert sidecar["channel_order"] == list(mod.CHANNEL_ORDER)
    assert sidecar["probability_channel_order"] == list(mod.PROB_CHANNEL_ORDER)
    assert sidecar["voxel_volume_mm3"] == pytest.approx(1.0)
    assert sidecar["voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    # The sidecar's "outputs" must point at the actual on-disk paths.
    artefacts = mod._planned_artefacts(deriv_root, "0005")
    assert sidecar["outputs"]["label_map"] == str(artefacts["seg_dl_label_map"])
    assert sidecar["outputs"]["probabilities"] == str(artefacts["seg_dl_probs"])
    assert sidecar["outputs"]["metrics_dl_vs_gt"] == str(artefacts["metr_dl_vs_gt"])


def test_real_process_subject_metrics_sidecar_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )
    metrics_path = mod._planned_artefacts(deriv_root, "0005")["metr_dl_vs_gt"]
    doc = json.loads(metrics_path.read_text())
    assert doc["schema_version"] == mod.METRICS_SIDECAR_SCHEMA_VERSION
    assert doc["subject_id"] == "0005"
    assert doc["comparison"] == "dl_vs_gt"
    assert doc["voxel_volume_mm3"] == pytest.approx(1.0)
    assert doc["voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    assert set(doc["metrics"]) == {"WT", "TC", "ET"}
    expected_keys = {
        "dice",
        "hd95_mm",
        "abs_volumetric_error_mm3",
        "sensitivity",
        "specificity",
    }
    for comp in ("WT", "TC", "ET"):
        assert set(doc["metrics"][comp]) == expected_keys


def test_real_process_subject_label_map_nifti_is_uint8_brats(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )
    seg_path = mod._planned_artefacts(deriv_root, "0005")["seg_dl_label_map"]
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj)
    assert data.dtype == np.uint8
    # The dummy backend places a synthetic tumor at the centre with all
    # three sub-regions present, so the label map must contain at least
    # background plus one of NCR/ED/ET.
    unique = set(int(v) for v in np.unique(data))
    assert 0 in unique
    assert unique.issubset({0, 1, 2, 4})
    assert unique != {0}  # something non-background must be present


def test_real_process_subject_probs_nifti_is_float32_xyz_then_compartment(
    tmp_path: Path,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005", shape=(40, 40, 40))
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )
    probs_path = mod._planned_artefacts(deriv_root, "0005")["seg_dl_probs"]
    img = nib.load(str(probs_path))
    data = np.asarray(img.dataobj)
    # NIfTI 4-D convention is (X, Y, Z, T): compartment axis last.
    assert data.shape == (40, 40, 40, 3)
    assert data.dtype == np.float32
    assert float(data.min()) >= 0.0
    assert float(data.max()) <= 1.0


def test_real_process_subject_skip_existing_short_circuits(tmp_path: Path) -> None:
    deriv_root = tmp_path / "deriv"
    artefacts = mod._planned_artefacts(deriv_root, "0005")
    artefacts["seg_dl_label_map"].parent.mkdir(parents=True, exist_ok=True)
    artefacts["seg_dl_label_map"].write_bytes(b"placeholder")
    artefacts["seg_dl_sidecar"].write_text("{}")

    record = mod._real_process_subject(
        "0005",
        # Cohort root deliberately does not exist; --skip-existing must not
        # touch it.
        cohort_root=tmp_path / "no-such-cohort",
        deriv_root=deriv_root,
        backend="monai_bundle",
        device="cpu",
        bundle_dir=None,
        skip_existing=True,
    )
    assert record["skipped"] is True
    assert "skip-existing" in record["reason"]


def test_real_process_subject_creates_subject_root_directory(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        device="cpu",
        bundle_dir=None,
        skip_existing=False,
    )
    sub_root = deriv_root / "sub-0005"
    assert sub_root.is_dir()
    assert (sub_root / "seg_dl").is_dir()
    assert (sub_root / "metrics").is_dir()
