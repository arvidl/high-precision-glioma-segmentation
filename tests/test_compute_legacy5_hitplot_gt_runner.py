"""Unit tests for ``scripts/compute_legacy5_hitplot_gt.py``.

Loaded via ``importlib`` because ``scripts/`` is intentionally not on
``sys.path``. Coverage rubric mirrors the PR-7f
(``compute_hitplot_cohort50``) tests but for the GT-only legacy-5
runner: subject loading, planned paths, end-to-end synthesis from a
mini fixture, LUT JSON schema, and the leak-safety contract that this
runner does *not* import any DL inference path.

Mapping: this script underpins the per-subject Hit-Plot bar charts in
Figs.~4, 5, 12, 13, 14 of ``paper/main.tex`` (revision Phase
C; see ``paper/revision_tracker.md``).
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


def _load_runner_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "compute_legacy5_hitplot_gt.py"
    spec = importlib.util.spec_from_file_location(
        "compute_legacy5_hitplot_gt_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


# ---------------------------------------------------------------------------
# Leak-safety contract: this script must NOT import the DL engine.
# ---------------------------------------------------------------------------


def test_runner_does_not_import_dl_engine() -> None:
    """The legacy-5 GT runner must never reach ``hpgs.segment.unified``.

    The 5 subjects are leak-unsafe for DL by construction (see
    ``configs/cohort_legacy5.yaml``). If a future refactor accidentally
    pulls in ``unified`` here, this test fires loudly.
    """
    src = Path(mod.__file__).read_text()
    assert "hpgs.segment.unified" not in src
    assert "predict_label_map" not in src


# ---------------------------------------------------------------------------
# _planned_paths: schema + filename contract
# ---------------------------------------------------------------------------


def test_planned_paths_layout(tmp_path: Path) -> None:
    paths = mod._planned_paths(
        deriv_root=tmp_path / "deriv",
        cohort_root=tmp_path / "cohort",
        subject_id="0020",
    )
    assert paths["wmparc"] == tmp_path / "deriv" / "sub-0020" / "parcellation" / "wmparc.nii.gz"
    assert (
        paths["wmparc_lut"]
        == tmp_path / "deriv" / "sub-0020" / "parcellation" / "sub-0020_wmparc_lut.json"
    )
    assert (
        paths["tumor_seg"]
        == tmp_path / "cohort" / "sub-0020" / "sub-0020_tumor_segmentation.nii.gz"
    )
    assert (
        paths["out_csv"] == tmp_path / "deriv" / "sub-0020" / "hitplot" / "sub-0020_hitplot_gt.csv"
    )


def test_planned_paths_normalises_subject_id(tmp_path: Path) -> None:
    paths = mod._planned_paths(
        deriv_root=tmp_path,
        cohort_root=tmp_path,
        subject_id="sub-0066",
    )
    assert "sub-0066" in str(paths["wmparc"])


# ---------------------------------------------------------------------------
# _load_cohort_subjects + _select_subjects
# ---------------------------------------------------------------------------


def test_load_cohort_subjects_reads_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cohort.yaml"
    yaml_path.write_text(
        "subjects:\n" "  - id: '0020'\n" "  - id: '0066'\n",
    )
    out = mod._load_cohort_subjects(yaml_path)
    assert out == ["0020", "0066"]


def test_load_cohort_subjects_rejects_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod._load_cohort_subjects(tmp_path / "missing.yaml")


def test_load_cohort_subjects_rejects_empty(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cohort.yaml"
    yaml_path.write_text("subjects: []\n")
    with pytest.raises(ValueError):
        mod._load_cohort_subjects(yaml_path)


def test_select_subjects_default_returns_full_cohort() -> None:
    out = mod._select_subjects(["0020", "0022", "0039"], None)
    assert out == ["0020", "0022", "0039"]


def test_select_subjects_filters_to_requested() -> None:
    out = mod._select_subjects(["0020", "0022", "0039"], ["0022"])
    assert out == ["0022"]


def test_select_subjects_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="not in legacy-5 cohort"):
        mod._select_subjects(["0020"], ["9999"])


# ---------------------------------------------------------------------------
# _voxel_volume_mm3
# ---------------------------------------------------------------------------


def test_voxel_volume_isotropic_1mm() -> None:
    affine = np.eye(4)
    assert mod._voxel_volume_mm3(affine) == pytest.approx(1.0)


def test_voxel_volume_anisotropic() -> None:
    affine = np.diag([0.8, 0.9, 1.1, 1.0])
    assert mod._voxel_volume_mm3(affine) == pytest.approx(0.8 * 0.9 * 1.1)


# ---------------------------------------------------------------------------
# End-to-end: synthetic wmparc + tumor on a 16^3 grid
# ---------------------------------------------------------------------------


def _write_synth_subject(
    cohort_root: Path,
    deriv_root: Path,
    subject_id: str,
) -> None:
    """Write a minimal wmparc + tumor_segmentation pair for one subject."""
    sub_cohort = cohort_root / f"sub-{subject_id}"
    sub_deriv = deriv_root / f"sub-{subject_id}" / "parcellation"
    sub_cohort.mkdir(parents=True, exist_ok=True)
    sub_deriv.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    wmparc = np.zeros((16, 16, 16), dtype=np.int32)
    # Two FreeSurfer regions present in synthseg_label_lut:
    #   2  Left-Cerebral-White-Matter (large region)
    #   17 Left-Hippocampus           (smaller region)
    wmparc[0:8, :, :] = 2
    wmparc[8:16, 0:6, :] = 17

    tumor = np.zeros_like(wmparc, dtype=np.int32)
    # NCR=1 in a small box inside region 2 (so TC = NCR + ET intersects region 2)
    tumor[2:5, 2:5, 2:5] = 1
    # ED=2 in a larger box inside region 2 (so WT = NCR + ED + ET >> TC)
    tumor[1:7, 1:7, 1:7] = 2
    tumor[2:5, 2:5, 2:5] = 1  # restore NCR voxels (ED pass overwrote them)
    # ET=4 in a small box that straddles regions 2 and 17
    tumor[7:9, 0:3, 0:3] = 4

    nib.save(nib.Nifti1Image(wmparc, affine), str(sub_deriv / "wmparc.nii.gz"))
    nib.save(
        nib.Nifti1Image(tumor, affine),
        str(sub_cohort / f"sub-{subject_id}_tumor_segmentation.nii.gz"),
    )


def test_process_subject_end_to_end(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synth_subject(cohort_root, deriv_root, "0020")

    record = mod._process_subject(
        "0020",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
    )
    assert record["status"] == "ok"
    assert record["n_rows"] >= 3  # >= 1 region * 3 compartments
    assert record["voxel_volume_mm3"] == pytest.approx(1.0)

    out_csv = Path(record["out_csv"])
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    # Schema parity with the cohort-50 hitplot CSV:
    for col in (
        "label",
        "name",
        "compartment",
        "parcel_voxels",
        "compartment_voxels",
        "overlap_voxels",
        "overlap_volume_mm3",
        "pct_of_parcel",
        "pct_of_compartment",
    ):
        assert col in df.columns

    lut_path = deriv_root / "sub-0020" / "parcellation" / "sub-0020_wmparc_lut.json"
    assert lut_path.is_file()
    lut = json.loads(lut_path.read_text())
    # Keys are stringified label ids; 2 and 17 are the synthesised labels.
    assert "2" in lut and "17" in lut
    assert lut["2"] == "Left-Cerebral-White-Matter"


def test_process_subject_skip_existing(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synth_subject(cohort_root, deriv_root, "0020")

    out_csv = deriv_root / "sub-0020" / "hitplot" / "sub-0020_hitplot_gt.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("placeholder\n")

    record = mod._process_subject(
        "0020",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=True,
    )
    assert record["status"] == "skipped"
    # Untouched: confirm we did not overwrite the placeholder.
    assert out_csv.read_text() == "placeholder\n"


def test_process_subject_missing_wmparc(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    (cohort_root / "sub-0020").mkdir(parents=True)
    (cohort_root / "sub-0020" / "sub-0020_tumor_segmentation.nii.gz").write_bytes(b"x")

    record = mod._process_subject(
        "0020",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
    )
    assert record["status"] == "missing_input"
    assert "wmparc" in record["reason"]


def test_process_subject_shape_mismatch(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    sub_cohort = cohort_root / "sub-0020"
    sub_deriv = deriv_root / "sub-0020" / "parcellation"
    sub_cohort.mkdir(parents=True, exist_ok=True)
    sub_deriv.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int32), affine),
        str(sub_deriv / "wmparc.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.int32), affine),
        str(sub_cohort / "sub-0020_tumor_segmentation.nii.gz"),
    )
    record = mod._process_subject(
        "0020",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
    )
    assert record["status"] == "error"
    assert "shape mismatch" in record["reason"]


def test_process_subject_affine_mismatch(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    sub_cohort = cohort_root / "sub-0020"
    sub_deriv = deriv_root / "sub-0020" / "parcellation"
    sub_cohort.mkdir(parents=True, exist_ok=True)
    sub_deriv.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int32), np.eye(4)),
        str(sub_deriv / "wmparc.nii.gz"),
    )
    affine_offset = np.eye(4)
    affine_offset[0, 3] = 5.0
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int32), affine_offset),
        str(sub_cohort / "sub-0020_tumor_segmentation.nii.gz"),
    )
    record = mod._process_subject(
        "0020",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        skip_existing=False,
    )
    assert record["status"] == "error"
    assert "differing affines" in record["reason"]


# ---------------------------------------------------------------------------
# _write_lut_json
# ---------------------------------------------------------------------------


def test_write_lut_json_uses_string_keys(tmp_path: Path) -> None:
    out = tmp_path / "lut.json"
    mod._write_lut_json({2: "Left-Cerebral-White-Matter", 17: "Left-Hippocampus"}, out)
    parsed = json.loads(out.read_text())
    assert set(parsed) == {"2", "17"}
    assert parsed["2"] == "Left-Cerebral-White-Matter"
