"""Unit tests for ``scripts/run_baseline_segmenters_cohort50.py`` (PR-7e).

Mirrors the structure of ``tests/test_segment_all_cohort50_runner.py``:
the script is loaded via ``importlib`` (because ``scripts/`` is not on
``sys.path``), and the artefact contract / JSON schemas / per-subject
runner are all exercised with the ``dummy`` backend so the tests stay
torch-, monai-, docker-, network- and GPU-free.

The tests also pin down the public surface of
:mod:`hpgs.segment.baselines` (``predict_baseline``, ``BaselineResult``,
``BASELINES``) so that the PR-7e2 docker swap-in does not silently
rename or relocate the contract the cohort runner depends on.

+ segment_glioma comparison on n=50, four-column Table 3 in the
Q1 = B locked framing) and R2.8 (Hit-Plot agreement panel reads the
seg_<baseline> masks this script writes).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hpgs.segment import LABEL_BG, LABEL_ED, LABEL_ET, LABEL_NCR
from hpgs.segment.baselines import (
    BASELINES,
    BaselineResult,
    predict_baseline,
)


def _load_runner_module():
    """Load ``scripts/run_baseline_segmenters_cohort50.py`` as a module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "run_baseline_segmenters_cohort50.py"
    spec = importlib.util.spec_from_file_location(
        "run_baseline_segmenters_cohort50_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# hpgs.segment.baselines public surface
# ---------------------------------------------------------------------------


def test_baselines_constant_includes_optional_tumorsynth_comparator() -> None:
    """The baseline harness includes TumorSynth as an optional comparator."""
    assert BASELINES == ("raidionics", "segmentglioma", "tumorsynth")


def _write_synthetic_flair(path: Path, *, shape=(40, 40, 40)) -> np.ndarray:
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)
    data = np.ones(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return affine


def test_predict_baseline_dummy_returns_uint8_brats_label_map(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair)
    result = predict_baseline(
        baseline="raidionics",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="dummy",
    )
    assert isinstance(result, BaselineResult)
    assert result.baseline == "raidionics"
    assert result.backend == "dummy"
    assert result.version == "dummy"
    assert result.label_map.dtype == np.uint8
    assert result.label_map.shape == (40, 40, 40)
    unique = set(int(v) for v in np.unique(result.label_map))
    # The dummy backend writes BG + at least one of NCR/ED/ET.
    assert unique.issubset({LABEL_BG, LABEL_NCR, LABEL_ED, LABEL_ET})
    assert unique != {LABEL_BG}


def test_predict_baseline_dummy_propagates_flair_affine(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    affine = np.diag([1.5, 0.5, 2.0, 1.0]).astype(float)
    data = np.ones((30, 30, 30), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine), str(flair))
    result = predict_baseline(
        baseline="segmentglioma",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="dummy",
    )
    np.testing.assert_array_almost_equal(result.affine, affine)


def test_predict_baseline_dummy_per_baseline_outputs_differ(tmp_path: Path) -> None:
    """Baselines on the same input must produce *distinct* label maps.

    Otherwise a confused output mapping at the runner-loop level would
    silently produce identical metric files for the two baselines.
    """
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair, shape=(40, 40, 40))
    a = predict_baseline(
        baseline="raidionics",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="dummy",
    )
    b = predict_baseline(
        baseline="segmentglioma",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="dummy",
    )
    c = predict_baseline(
        baseline="tumorsynth",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="dummy",
    )
    assert a.label_map.shape == b.label_map.shape
    assert not np.array_equal(a.label_map, b.label_map)
    assert not np.array_equal(a.label_map, c.label_map)
    assert not np.array_equal(b.label_map, c.label_map)


def test_predict_baseline_unknown_baseline_raises(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair)
    with pytest.raises(ValueError, match="Unknown baseline"):
        predict_baseline(
            baseline="not_a_real_baseline",  # type: ignore[arg-type]
            t1=flair,
            t1c=flair,
            t2=flair,
            flair=flair,
            backend="dummy",
        )


def test_predict_baseline_unknown_backend_raises(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair)
    with pytest.raises(ValueError, match="Unknown backend"):
        predict_baseline(
            baseline="raidionics",
            t1=flair,
            t1c=flair,
            t2=flair,
            flair=flair,
            backend="not_a_real_backend",  # type: ignore[arg-type]
        )


def test_predict_baseline_docker_backend_is_pr7e2_stub(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair)
    with pytest.raises(NotImplementedError, match="PR-7e2"):
        predict_baseline(
            baseline="raidionics",
            t1=flair,
            t1c=flair,
            t2=flair,
            flair=flair,
            backend="docker",
        )


def test_predict_baseline_command_backend_rejects_non_tumorsynth(tmp_path: Path) -> None:
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair)
    with pytest.raises(NotImplementedError, match="tumorsynth"):
        predict_baseline(
            baseline="raidionics",
            t1=flair,
            t1c=flair,
            t2=flair,
            flair=flair,
            backend="command",
        )


def test_predict_baseline_tumorsynth_command_remaps_two_stage_outputs(tmp_path: Path) -> None:
    """A fake mri_TumorSynth command lets us test the wrapper contract."""
    flair = tmp_path / "flair.nii.gz"
    _write_synthetic_flair(flair, shape=(20, 20, 20))
    fake_cmd = tmp_path / "fake_mri_TumorSynth.py"
    fake_cmd.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys
import nibabel as nib
import numpy as np

args = sys.argv[1:]
out = Path(args[args.index("--o") + 1])
first_input = args[args.index("--i") + 1].split(",")[0]
img = nib.load(first_input)
data = np.zeros(img.shape, dtype=np.uint8)
if "--wholetumor" in args:
    data[5:15, 5:15, 5:15] = 18
else:
    data[6:10, 6:10, 6:10] = 1
    data[10:13, 10:13, 10:13] = 2
    data[13:15, 13:15, 13:15] = 3
nib.save(nib.Nifti1Image(data, img.affine), str(out))
""",
    )
    fake_cmd.chmod(0o755)

    result = predict_baseline(
        baseline="tumorsynth",
        t1=flair,
        t1c=flair,
        t2=flair,
        flair=flair,
        backend="command",
        tumorsynth_command=fake_cmd,
        tumorsynth_nnunet_dir=tmp_path,
        tumorsynth_threads=1,
        tumorsynth_cpu=True,
        tumorsynth_preserve_raw_dir=tmp_path / "raw",
        tumorsynth_preserve_raw_stem="sub-0005_tumorsynth",
    )

    unique = set(int(v) for v in np.unique(result.label_map))
    assert unique == {LABEL_BG, LABEL_NCR, LABEL_ED, LABEL_ET}
    assert result.label_map[7, 7, 7] == LABEL_ED
    assert result.label_map[14, 14, 14] == LABEL_NCR
    assert result.label_map[11, 11, 11] == LABEL_ET
    assert result.label_map[5, 5, 5] == LABEL_ED
    assert "mri_TumorSynth" in result.version or "fake_mri_TumorSynth" in result.version
    assert result.raw_outputs["tumorsynth_whole_tumor_raw"].is_file()
    assert result.raw_outputs["tumorsynth_inner_tumor_raw"].is_file()
    raw_inner = nib.load(str(result.raw_outputs["tumorsynth_inner_tumor_raw"]))
    assert set(int(v) for v in np.unique(np.asanyarray(raw_inner.dataobj))) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# _planned_baseline_artefacts: Section 4 extension
# ---------------------------------------------------------------------------


def test_planned_baseline_artefacts_keys_match_section_4_extension(tmp_path: Path) -> None:
    paths = mod._planned_baseline_artefacts(tmp_path / "deriv", "0005", "raidionics")
    expected = {
        "seg_label_map",
        "seg_sidecar",
        "metr_dl_vs_baseline",
        "seg_dl_label_map",
    }
    assert set(paths) == expected


def test_planned_baseline_artefacts_layout_matches_section_4(tmp_path: Path) -> None:
    paths = mod._planned_baseline_artefacts(
        tmp_path / "deriv",
        "0005",
        "raidionics",
    )
    sub = tmp_path / "deriv" / "sub-0005"
    assert paths["seg_label_map"] == sub / "seg_raidionics" / "sub-0005_seg_raidionics.nii.gz"
    assert paths["seg_sidecar"] == sub / "seg_raidionics" / "sub-0005_seg_raidionics.json"
    assert paths["metr_dl_vs_baseline"] == (
        sub / "metrics" / "sub-0005_metrics_dl_vs_raidionics.json"
    )
    # PR-7d input: not written by this script, only probed.
    assert paths["seg_dl_label_map"] == sub / "seg_dl" / "sub-0005_seg_brats3_dl.nii.gz"


def test_planned_baseline_artefacts_segmentglioma_paths(tmp_path: Path) -> None:
    paths = mod._planned_baseline_artefacts(
        tmp_path / "deriv",
        "0005",
        "segmentglioma",
    )
    sub = tmp_path / "deriv" / "sub-0005"
    assert paths["seg_label_map"] == (
        sub / "seg_segmentglioma" / "sub-0005_seg_segmentglioma.nii.gz"
    )
    assert paths["metr_dl_vs_baseline"] == (
        sub / "metrics" / "sub-0005_metrics_dl_vs_segmentglioma.json"
    )


def test_planned_baseline_artefacts_tumorsynth_paths(tmp_path: Path) -> None:
    paths = mod._planned_baseline_artefacts(
        tmp_path / "deriv",
        "0005",
        "tumorsynth",
    )
    sub = tmp_path / "deriv" / "sub-0005"
    assert paths["seg_label_map"] == sub / "seg_tumorsynth" / "sub-0005_seg_tumorsynth.nii.gz"
    assert paths["metr_dl_vs_baseline"] == (
        sub / "metrics" / "sub-0005_metrics_dl_vs_tumorsynth.json"
    )
    assert paths["seg_tumorsynth_raw_whole"] == (
        sub / "seg_tumorsynth" / "sub-0005_tumorsynth_wholetumor_raw.nii.gz"
    )
    assert paths["seg_tumorsynth_raw_inner"] == (
        sub / "seg_tumorsynth" / "sub-0005_tumorsynth_innertumor_raw.nii.gz"
    )


def test_planned_baseline_artefacts_normalises_subject_id(tmp_path: Path) -> None:
    a = mod._planned_baseline_artefacts(tmp_path / "deriv", "0005", "raidionics")
    b = mod._planned_baseline_artefacts(tmp_path / "deriv", "sub-0005", "raidionics")
    assert a == b


def test_planned_baseline_artefacts_unknown_baseline_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        mod._planned_baseline_artefacts(tmp_path / "deriv", "0005", "not_a_baseline")


# ---------------------------------------------------------------------------
# Voxel + JSON helpers (mirrored from PR-7d -- pin the schema)
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


def test_json_safe_passes_finite_replaces_nan_and_inf() -> None:
    assert mod._json_safe(0.5) == 0.5
    assert mod._json_safe(float("nan")) is None
    assert mod._json_safe(float("inf")) == "inf"
    assert mod._json_safe(-float("inf")) == "inf"
    encoded = json.dumps(
        {
            "dice": mod._json_safe(0.5),
            "hd95_mm": mod._json_safe(float("inf")),
            "ve_mm3": mod._json_safe(float("nan")),
        },
    )
    assert json.loads(encoded) == {"dice": 0.5, "hd95_mm": "inf", "ve_mm3": None}


# ---------------------------------------------------------------------------
# _per_compartment_paired_metrics
# ---------------------------------------------------------------------------


def _label_map_with_all_classes() -> np.ndarray:
    lm = np.zeros((20, 20, 20), dtype=np.int16)
    lm[2:8, 2:8, 2:8] = LABEL_NCR
    lm[6:12, 6:12, 6:12] = LABEL_ET
    lm[12:16, 12:16, 12:16] = LABEL_ED
    return lm


def test_paired_metrics_identical_masks_perfect_scores() -> None:
    lm = _label_map_with_all_classes()
    out = mod._per_compartment_paired_metrics(
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


def test_paired_metrics_empty_reference_returns_json_safe_record() -> None:
    pred = _label_map_with_all_classes()
    ref = np.zeros_like(pred)
    out = mod._per_compartment_paired_metrics(
        pred,
        ref,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        voxel_volume_mm3=1.0,
    )
    for comp in ("WT", "TC", "ET"):
        assert out[comp]["dice"] == pytest.approx(0.0)
        assert out[comp]["hd95_mm"] == "inf"
        assert isinstance(out[comp]["abs_volumetric_error_mm3"], float)
        assert out[comp]["abs_volumetric_error_mm3"] > 0
        assert out[comp]["sensitivity"] is None  # 0/0 -> NaN -> None
        assert out[comp]["specificity"] is not None


def test_paired_metrics_record_is_json_round_trippable() -> None:
    pred = _label_map_with_all_classes()
    ref = np.zeros_like(pred)
    out = mod._per_compartment_paired_metrics(
        pred,
        ref,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        voxel_volume_mm3=1.0,
    )
    decoded = json.loads(json.dumps(out))
    assert set(decoded) == {"WT", "TC", "ET"}


# ---------------------------------------------------------------------------
# _select_subjects / _select_baselines
# ---------------------------------------------------------------------------


def test_select_subjects_default_returns_full_cohort() -> None:
    cohort = ["0005", "0026", "0068"]
    assert mod._select_subjects(cohort, None) == cohort


def test_select_subjects_filter_normalises_sub_prefix() -> None:
    out = mod._select_subjects(["0005", "0026", "0068"], ["0005", "sub-0026"])
    assert out == ["0005", "0026"]


def test_select_subjects_rejects_unknown_id() -> None:
    with pytest.raises(SystemExit, match="not in cohort YAML"):
        mod._select_subjects(["0005"], ["9999"])


def test_select_baselines_default_returns_all() -> None:
    assert mod._select_baselines(None) == list(BASELINES)


def test_select_baselines_explicit_all_expands() -> None:
    assert mod._select_baselines(["all"]) == list(BASELINES)


def test_select_baselines_dedups_repeated_flags() -> None:
    out = mod._select_baselines(["raidionics", "raidionics", "segmentglioma"])
    assert out == ["raidionics", "segmentglioma"]


def test_select_baselines_accepts_tumorsynth() -> None:
    assert mod._select_baselines(["tumorsynth"]) == ["tumorsynth"]


def test_select_baselines_rejects_unknown() -> None:
    with pytest.raises(SystemExit, match="not in"):
        mod._select_baselines(["not_a_baseline"])


# ---------------------------------------------------------------------------
# _real_process_subject_baseline end-to-end with the dummy backend
# ---------------------------------------------------------------------------


def _write_synthetic_cohort_subject(
    cohort_root: Path,
    subject_id: str,
    *,
    shape: tuple[int, int, int] = (40, 40, 40),
    seed: int = 20260415,
) -> None:
    """Write the four channels + tumor_segmentation NIfTIs for one subject.

    Same fixture shape as ``tests/test_segment_all_cohort50_runner.py`` so
    the two suites share an identical synthetic cohort layout.
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
    ref[10:16, 10:16, 10:16] = LABEL_NCR
    ref[15:20, 15:20, 15:20] = LABEL_ET
    ref[22:28, 22:28, 22:28] = LABEL_ED
    nib.save(
        nib.Nifti1Image(ref, affine),
        str(sub_dir / f"sub-{subject_id}_tumor_segmentation.nii.gz"),
    )


def _write_dl_output(
    deriv_root: Path,
    subject_id: str,
    *,
    shape: tuple[int, int, int] = (40, 40, 40),
) -> Path:
    """Place a synthetic PR-7d DL label map on disk so paired metrics fire."""
    seg_dl_dir = deriv_root / f"sub-{subject_id}" / "seg_dl"
    seg_dl_dir.mkdir(parents=True, exist_ok=True)
    dl_lm = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    dl_lm[cz - 4 : cz + 4, cy - 4 : cy + 4, cx - 4 : cx + 4] = LABEL_ET
    dl_lm[cz - 6 : cz - 4, cy - 6 : cy - 4, cx - 6 : cx - 4] = LABEL_NCR
    dl_lm[cz + 4 : cz + 6, cy + 4 : cy + 6, cx + 4 : cx + 6] = LABEL_ED
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)
    path = seg_dl_dir / f"sub-{subject_id}_seg_brats3_dl.nii.gz"
    nib.save(nib.Nifti1Image(dl_lm, affine), str(path))
    return path


def test_real_process_writes_seg_and_paired_metrics(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")

    record = mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )

    assert record["subject_id"] == "0005"
    assert record["baseline"] == "raidionics"
    assert record["skipped"] is False
    assert record["paired_metrics_status"] == "ok"
    assert set(record["metrics"]) == {"WT", "TC", "ET"}

    artefacts = mod._planned_baseline_artefacts(deriv_root, "0005", "raidionics")
    for key in ("seg_label_map", "seg_sidecar", "metr_dl_vs_baseline"):
        assert artefacts[key].is_file(), f"missing {key}: {artefacts[key]}"


def test_real_process_seg_sidecar_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")

    mod._real_process_subject_baseline(
        "0005",
        "segmentglioma",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    sidecar_path = mod._planned_baseline_artefacts(
        deriv_root,
        "0005",
        "segmentglioma",
    )["seg_sidecar"]
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == mod.SEG_SIDECAR_SCHEMA_VERSION
    assert sidecar["subject_id"] == "0005"
    assert sidecar["baseline"] == "segmentglioma"
    assert sidecar["backend"] == "dummy"
    assert sidecar["version"] == "dummy"
    assert sidecar["voxel_volume_mm3"] == pytest.approx(1.0)
    assert sidecar["voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    assert sidecar["paired_metrics_status"] == "ok"
    artefacts = mod._planned_baseline_artefacts(deriv_root, "0005", "segmentglioma")
    assert sidecar["outputs"]["label_map"] == str(artefacts["seg_label_map"])
    assert sidecar["outputs"]["metrics_dl_vs_baseline"] == str(
        artefacts["metr_dl_vs_baseline"],
    )


def test_real_process_metrics_sidecar_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")

    mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    metrics_path = mod._planned_baseline_artefacts(
        deriv_root,
        "0005",
        "raidionics",
    )["metr_dl_vs_baseline"]
    doc = json.loads(metrics_path.read_text())
    assert doc["schema_version"] == mod.METRICS_SIDECAR_SCHEMA_VERSION
    assert doc["subject_id"] == "0005"
    assert doc["comparison"] == "dl_vs_raidionics"
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


def test_real_process_label_map_nifti_is_uint8_brats(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")
    mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    seg_path = mod._planned_baseline_artefacts(
        deriv_root,
        "0005",
        "raidionics",
    )["seg_label_map"]
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj)
    assert data.dtype == np.uint8
    unique = set(int(v) for v in np.unique(data))
    assert 0 in unique
    assert unique.issubset({0, 1, 2, 4})
    assert unique != {0}


def test_real_process_missing_dl_raises_when_required(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    # NB: no _write_dl_output call -> DL output is missing on purpose.
    with pytest.raises(FileNotFoundError, match="DL output"):
        mod._real_process_subject_baseline(
            "0005",
            "raidionics",
            cohort_root=cohort_root,
            deriv_root=deriv_root,
            backend="dummy",
            skip_existing=False,
            require_dl=True,
        )


def test_real_process_missing_dl_writes_seg_only_when_not_required(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    record = mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=False,
    )
    assert record["paired_metrics_status"] == "dl_missing"
    assert record["metrics"] is None
    artefacts = mod._planned_baseline_artefacts(deriv_root, "0005", "raidionics")
    # Mask + sidecar written; paired-metrics JSON absent.
    assert artefacts["seg_label_map"].is_file()
    assert artefacts["seg_sidecar"].is_file()
    assert not artefacts["metr_dl_vs_baseline"].is_file()


def test_real_process_skip_existing_short_circuits(tmp_path: Path) -> None:
    deriv_root = tmp_path / "deriv"
    artefacts = mod._planned_baseline_artefacts(deriv_root, "0005", "raidionics")
    artefacts["seg_label_map"].parent.mkdir(parents=True, exist_ok=True)
    artefacts["seg_label_map"].write_bytes(b"placeholder")
    artefacts["seg_sidecar"].write_text("{}")

    record = mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        # Cohort root deliberately does not exist; --skip-existing must
        # short-circuit before any input-file resolution.
        cohort_root=tmp_path / "no-such-cohort",
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=True,
        require_dl=True,
    )
    assert record["skipped"] is True
    assert "skip-existing" in record["reason"]


def test_real_process_creates_subject_root_directory(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")
    mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    sub_root = deriv_root / "sub-0005"
    assert sub_root.is_dir()
    assert (sub_root / "seg_raidionics").is_dir()
    assert (sub_root / "metrics").is_dir()


def test_real_process_two_baselines_produce_distinct_metrics(tmp_path: Path) -> None:
    """The baselines must produce *different* paired metrics on the
    same DL reference -- a regression guard against confused output
    mapping at the runner-loop level."""
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_dl_output(deriv_root, "0005")

    rec_a = mod._real_process_subject_baseline(
        "0005",
        "raidionics",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    rec_b = mod._real_process_subject_baseline(
        "0005",
        "segmentglioma",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    rec_c = mod._real_process_subject_baseline(
        "0005",
        "tumorsynth",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        skip_existing=False,
        require_dl=True,
    )
    a_dice = rec_a["metrics"]["WT"]["dice"]
    b_dice = rec_b["metrics"]["WT"]["dice"]
    c_dice = rec_c["metrics"]["WT"]["dice"]
    # Either Dice differs, or HD95 differs -- but they cannot all be
    # exactly identical because the dummy backends are jittered.
    a_hd = rec_a["metrics"]["WT"]["hd95_mm"]
    b_hd = rec_b["metrics"]["WT"]["hd95_mm"]
    c_hd = rec_c["metrics"]["WT"]["hd95_mm"]
    assert (a_dice, a_hd) != (b_dice, b_hd)
    assert (a_dice, a_hd) != (c_dice, c_hd)
