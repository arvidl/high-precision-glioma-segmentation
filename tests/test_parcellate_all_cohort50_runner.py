"""Unit tests for ``scripts/parcellate_all_cohort50.py`` and the
``hpgs.parcellate.predict_wmparc`` adapter (PR-7c).

We load the script as a module via ``importlib`` so the CLI shim does
not have to be on ``sys.path``, and we exercise the full artefact
contract + ``main()`` orchestration with the ``"dummy"`` backend ---
no FreeSurfer needed on the host, so the suite is fully hermetic on
CI and on developer laptops.

producing the per-subject ``wmparc_native`` + LUT the PR-7f runner
reads.
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

from hpgs.parcellate import (
    FS_DUMMY_LUT,
    ParcellationResult,
    predict_wmparc,
)


def _load_runner_module():
    """Load ``scripts/parcellate_all_cohort50.py`` as a module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "parcellate_all_cohort50.py"
    spec = importlib.util.spec_from_file_location(
        "parcellate_all_cohort50_runner",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_runner_module()


AFFINE = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)


def _write_synthetic_cohort_subject(
    cohort_root: Path,
    subject_id: str,
    *,
    shape: tuple[int, int, int] = (32, 32, 32),
) -> None:
    """Write every bias-corrected channel + tumor_segmentation for a subject.

    Matches the flat layout :func:`hpgs.io.resolve_subject_inputs`
    expects. We write all four channels so the runner is free to
    switch ``--input-channel`` per test without touching the fixture.
    """
    sub_dir = cohort_root / f"sub-{subject_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260417 + int(subject_id))
    for ch in ("T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"):
        data = (rng.random(shape, dtype=np.float32) * 100.0 + 1.0).astype(np.float32)
        nib.save(
            nib.Nifti1Image(data, AFFINE),
            str(sub_dir / f"sub-{subject_id}_{ch}.nii.gz"),
        )
    ref = np.zeros(shape, dtype=np.int16)
    ref[8:12, 8:12, 8:12] = 4
    nib.save(
        nib.Nifti1Image(ref, AFFINE),
        str(sub_dir / f"sub-{subject_id}_tumor_segmentation.nii.gz"),
    )


def _fake_cohort_metadata(subject_ids: list[str]):
    """Minimal stand-in for ``CohortMetadata`` with just ``.df.index`` and ``.n``."""

    class _FakeCM:
        def __init__(self, sids: list[str]) -> None:
            self.df = pd.DataFrame(index=sids, data={"Sex": ["M"] * len(sids)})
            self.n = len(sids)
            self.seed = 0

    return _FakeCM(subject_ids)


# ---------------------------------------------------------------------------
# predict_wmparc adapter (dummy backend): shape, dtype, LUT
# ---------------------------------------------------------------------------


def test_predict_wmparc_dummy_returns_int16_label_map_on_native_grid(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    data = np.random.default_rng(0).random((20, 24, 18), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, AFFINE), str(inp))

    result = predict_wmparc(
        input_nifti=inp,
        subject_id="0005",
        backend="dummy",
    )
    assert isinstance(result, ParcellationResult)
    assert result.label_map.shape == (20, 24, 18)
    assert result.label_map.dtype == np.int16
    assert result.backend == "dummy"
    assert result.version == "dummy"


def test_predict_wmparc_dummy_copies_input_affine(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    affine = np.array(
        [
            [1.5, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -120.0],
            [0.0, 0.0, 0.9, 50.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), affine), str(inp))
    result = predict_wmparc(input_nifti=inp, subject_id="0007", backend="dummy")
    np.testing.assert_allclose(result.affine, affine)


def test_predict_wmparc_dummy_lut_is_subset_of_fs_dummy_lut(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((24, 24, 24), dtype=np.float32), AFFINE), str(inp))
    result = predict_wmparc(input_nifti=inp, subject_id="0042", backend="dummy")
    assert set(result.lut).issubset(set(FS_DUMMY_LUT))
    # The 24x24x24 volume is large enough for the dummy generator to
    # lay down every canonical region; assert the full complement.
    assert set(result.lut) == set(FS_DUMMY_LUT)


def test_predict_wmparc_dummy_lut_labels_match_label_map(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((24, 24, 24), dtype=np.float32), AFFINE), str(inp))
    result = predict_wmparc(input_nifti=inp, subject_id="0026", backend="dummy")
    present = set(int(v) for v in np.unique(result.label_map))
    # Every label in the returned LUT is present in the label map.
    assert set(result.lut).issubset(present)


def test_predict_wmparc_dummy_is_deterministic_per_subject(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.float32), AFFINE), str(inp))
    a = predict_wmparc(input_nifti=inp, subject_id="0005", backend="dummy")
    b = predict_wmparc(input_nifti=inp, subject_id="0005", backend="dummy")
    np.testing.assert_array_equal(a.label_map, b.label_map)


def test_predict_wmparc_dummy_distinct_per_subject(tmp_path: Path) -> None:
    """Two different subject-ids must yield distinct synthetic wmparcs so
    the cohort runner's per-subject distinctness is exercised."""
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.float32), AFFINE), str(inp))
    a = predict_wmparc(input_nifti=inp, subject_id="0005", backend="dummy")
    b = predict_wmparc(input_nifti=inp, subject_id="0006", backend="dummy")
    assert not np.array_equal(a.label_map, b.label_map)


def test_predict_wmparc_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        predict_wmparc(
            input_nifti=tmp_path / "no-such-file.nii.gz",
            subject_id="0005",
            backend="dummy",
        )


def test_predict_wmparc_rejects_unknown_backend(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), AFFINE), str(inp))
    with pytest.raises(ValueError, match="Unknown parcellation backend"):
        predict_wmparc(
            input_nifti=inp,
            subject_id="0005",
            backend="nonsense",  # type: ignore[arg-type]
        )


def test_predict_wmparc_freesurfer_requires_work_dir(tmp_path: Path) -> None:
    inp = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), AFFINE), str(inp))
    with pytest.raises(ValueError, match="fs_work_dir"):
        predict_wmparc(
            input_nifti=inp,
            subject_id="0005",
            backend="freesurfer",
            fs_work_dir=None,
        )


# ---------------------------------------------------------------------------
# _planned_artefacts: Section 4 contract
# ---------------------------------------------------------------------------


def test_planned_artefacts_keys_match_section_4_contract(tmp_path: Path) -> None:
    paths = mod._planned_artefacts(tmp_path / "deriv", "0005")
    assert set(paths) == {"parc_wmparc", "parc_lut", "parc_sidecar"}


def test_planned_artefacts_layout_uses_section_4_subdirectory(tmp_path: Path) -> None:
    paths = mod._planned_artefacts(tmp_path / "deriv", "0005")
    parc = tmp_path / "deriv" / "sub-0005" / "parcellation"
    assert paths["parc_wmparc"] == parc / "sub-0005_wmparc_native.nii.gz"
    assert paths["parc_lut"] == parc / "sub-0005_wmparc_lut.json"
    assert paths["parc_sidecar"] == parc / "sub-0005_wmparc.json"


def test_planned_artefacts_normalises_subject_id(tmp_path: Path) -> None:
    prefixed = mod._planned_artefacts(tmp_path / "deriv", "sub-0042")
    plain = mod._planned_artefacts(tmp_path / "deriv", "0042")
    assert prefixed == plain


# ---------------------------------------------------------------------------
# _check_inputs: dry-run MISS/OK contract
# ---------------------------------------------------------------------------


def test_check_inputs_ok_when_all_present(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    ok, detail = mod._check_inputs(cohort_root, "0005", input_channel="T1_bias")
    assert ok is True
    assert detail == ""


def test_check_inputs_miss_when_channel_missing(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    (cohort_root / "sub-0005" / "sub-0005_T1_bias.nii.gz").unlink()
    ok, detail = mod._check_inputs(cohort_root, "0005", input_channel="T1_bias")
    assert ok is False
    assert "T1_bias" in detail


def test_check_inputs_miss_when_tumor_seg_missing(tmp_path: Path) -> None:
    """Coherence with PR-7d/PR-7e: a subject missing tumor_segmentation is
    MISS across the board even though parcellation only needs one channel
    --- the dry-run must flag incomplete cohort drops so no downstream
    runner is surprised later."""
    cohort_root = tmp_path / "cohort"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    (cohort_root / "sub-0005" / "sub-0005_tumor_segmentation.nii.gz").unlink()
    ok, detail = mod._check_inputs(cohort_root, "0005", input_channel="T1_bias")
    assert ok is False
    assert "tumor_segmentation" in detail


# ---------------------------------------------------------------------------
# _select_subjects
# ---------------------------------------------------------------------------


def test_select_subjects_default_returns_full_cohort() -> None:
    assert mod._select_subjects(["0005", "0013", "0026"], None) == ["0005", "0013", "0026"]


def test_select_subjects_filter_normalises_sub_prefix() -> None:
    assert mod._select_subjects(["0005", "0013"], ["sub-0005"]) == ["0005"]


def test_select_subjects_rejects_unknown_id() -> None:
    with pytest.raises(SystemExit, match="not in cohort YAML"):
        mod._select_subjects(["0005"], ["9999"])


# ---------------------------------------------------------------------------
# _write_lut_json: string-integer keys, sorted, JSON round-trippable
# ---------------------------------------------------------------------------


def test_write_lut_json_uses_string_integer_keys(tmp_path: Path) -> None:
    out = tmp_path / "lut.json"
    mod._write_lut_json({2: "Left-WM", 41: "Right-WM"}, out)
    payload = json.loads(out.read_text())
    assert set(payload) == {"2", "41"}
    assert payload["2"] == "Left-WM"
    assert payload["41"] == "Right-WM"


def test_write_lut_json_keys_are_ascending_numeric(tmp_path: Path) -> None:
    out = tmp_path / "lut.json"
    mod._write_lut_json({41: "Right-WM", 2: "Left-WM", 16: "Brain-Stem"}, out)
    raw = out.read_text()
    # Order within the JSON text is 2, 16, 41 (numeric ascending).
    assert raw.index('"2"') < raw.index('"16"') < raw.index('"41"')


def test_write_lut_json_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "lut.json"
    mod._write_lut_json({0: "Unknown"}, out)
    assert out.is_file()


# ---------------------------------------------------------------------------
# _real_process_subject: Section-4 artefact contract + sidecar schema
# ---------------------------------------------------------------------------


def test_real_process_subject_writes_section_4_artefacts(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    record = mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    assert record["subject_id"] == "0005"
    assert record["skipped"] is False
    assert record["backend"] == "dummy"

    artefacts = mod._planned_artefacts(deriv_root, "0005")
    assert artefacts["parc_wmparc"].is_file()
    assert artefacts["parc_lut"].is_file()
    assert artefacts["parc_sidecar"].is_file()


def test_real_process_subject_wmparc_is_int16_native_grid(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005", shape=(28, 28, 28))

    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    wmparc_path = mod._planned_artefacts(deriv_root, "0005")["parc_wmparc"]
    img = nib.load(str(wmparc_path))
    data = np.asarray(img.dataobj)
    assert data.shape == (28, 28, 28)
    assert data.dtype == np.int16


def test_real_process_subject_lut_json_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    lut_path = mod._planned_artefacts(deriv_root, "0005")["parc_lut"]
    payload = json.loads(lut_path.read_text())
    # All keys are string-encoded ints.
    for k in payload:
        assert isinstance(k, str)
        int(k)  # does not raise
    # All values are strings (anatomical names).
    for v in payload.values():
        assert isinstance(v, str)


def test_real_process_subject_sidecar_schema(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    sidecar_path = mod._planned_artefacts(deriv_root, "0005")["parc_sidecar"]
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == mod.PARC_SIDECAR_SCHEMA_VERSION
    assert sidecar["subject_id"] == "0005"
    assert sidecar["backend"] == "dummy"
    assert sidecar["version"] == "dummy"
    assert sidecar["input_channel"] == "T1_bias"
    assert sidecar["threads"] == 4
    assert sidecar["n_labels_present"] >= 1
    assert isinstance(sidecar["elapsed_s"], float)
    assert sidecar["outputs"]["wmparc_native"].endswith("sub-0005_wmparc_native.nii.gz")
    assert sidecar["outputs"]["wmparc_lut"].endswith("sub-0005_wmparc_lut.json")


def test_real_process_subject_honours_input_channel(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    # Remove T1_bias so if the runner ignored --input-channel and fell
    # back to T1 it would fail; FLAIR_bias remains present.
    (cohort_root / "sub-0005" / "sub-0005_T1_bias.nii.gz").unlink()

    record = mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="FLAIR_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    assert record["skipped"] is False
    sidecar = json.loads(mod._planned_artefacts(deriv_root, "0005")["parc_sidecar"].read_text())
    assert sidecar["input_channel"] == "FLAIR_bias"


def test_real_process_subject_creates_subject_root_directory(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    sub_root = deriv_root / "sub-0005"
    assert sub_root.is_dir()
    assert (sub_root / "parcellation").is_dir()


def test_real_process_subject_skip_existing_short_circuits(tmp_path: Path) -> None:
    deriv_root = tmp_path / "deriv"
    artefacts = mod._planned_artefacts(deriv_root, "0005")
    artefacts["parc_wmparc"].parent.mkdir(parents=True, exist_ok=True)
    artefacts["parc_wmparc"].write_bytes(b"placeholder")
    artefacts["parc_lut"].write_text("{}")

    record = mod._real_process_subject(
        "0005",
        cohort_root=tmp_path / "no-such-cohort",  # must NOT be touched
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=True,
    )
    assert record["skipped"] is True
    assert "skip-existing" in record["reason"]


def test_real_process_subject_raises_on_missing_input(tmp_path: Path) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # No fixture written --- resolve_subject_inputs must raise.
    with pytest.raises(FileNotFoundError):
        mod._real_process_subject(
            "0005",
            cohort_root=cohort_root,
            deriv_root=deriv_root,
            backend="dummy",
            input_channel="T1_bias",
            fs_work_dir=None,
            threads=4,
            skip_existing=False,
        )


def test_real_process_subject_lut_values_match_fs_dummy_lut_names(tmp_path: Path) -> None:
    """The on-disk LUT JSON must map to the *same* anatomical names as
    :data:`hpgs.parcellate.FS_DUMMY_LUT` (i.e. the dummy backend does
    not invent names). This is the contract ``compute_hitplot_cohort50
    ._read_wmparc_lut`` implicitly relies on."""
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    payload = json.loads(mod._planned_artefacts(deriv_root, "0005")["parc_lut"].read_text())
    for label_str, name in payload.items():
        assert FS_DUMMY_LUT[int(label_str)] == name


# ---------------------------------------------------------------------------
# CLI orchestration: main() exit codes 0/1/2
# ---------------------------------------------------------------------------


def test_main_dry_run_prints_per_subject_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cohort_root = tmp_path / "cohort"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(tmp_path / "deriv"),
        ],
    )
    rc = mod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PR-7c dry-run" in out
    assert "sub-0005" in out
    assert "inputs present" in out


def test_main_dry_run_returns_1_on_missing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"  # empty --- every subject MISS

    cm = _fake_cohort_metadata(["0005", "0013"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(tmp_path / "deriv"),
        ],
    )
    assert mod.main() == 1


def test_main_real_run_dummy_backend_returns_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_synthetic_cohort_subject(cohort_root, "0013")

    cm = _fake_cohort_metadata(["0005", "0013"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--no-dry-run",
            "--backend",
            "dummy",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
        ],
    )
    assert mod.main() == 0
    # Both subjects produced the contract.
    for sid in ("0005", "0013"):
        a = mod._planned_artefacts(deriv_root, sid)
        assert a["parc_wmparc"].is_file()
        assert a["parc_lut"].is_file()
        assert a["parc_sidecar"].is_file()


def test_main_real_run_dummy_backend_returns_1_on_per_subject_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    # Only write 0005; 0013 is missing -> FileNotFoundError per subject.
    _write_synthetic_cohort_subject(cohort_root, "0005")

    cm = _fake_cohort_metadata(["0005", "0013"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--no-dry-run",
            "--backend",
            "dummy",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
        ],
    )
    rc = mod.main()
    assert rc == 1
    # 0005 still produced the contract (failures don't abort the run).
    a = mod._planned_artefacts(deriv_root, "0005")
    assert a["parc_wmparc"].is_file()


def test_main_returns_2_on_cohort_load_failure(
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
            "parcellate_all_cohort50.py",
            "--cohort-root",
            str(tmp_path / "cohort"),
            "--deriv-root",
            str(tmp_path / "deriv"),
        ],
    )
    assert mod.main() == 2


def test_main_returns_2_when_freesurfer_backend_without_fs_work_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--no-dry-run",
            "--backend",
            "freesurfer",
            # no --fs-work-dir
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(tmp_path / "deriv"),
        ],
    )
    assert mod.main() == 2


def test_main_real_run_honours_subject_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    _write_synthetic_cohort_subject(cohort_root, "0013")

    cm = _fake_cohort_metadata(["0005", "0013"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parcellate_all_cohort50.py",
            "--no-dry-run",
            "--backend",
            "dummy",
            "--subject",
            "0005",
            "--cohort-root",
            str(cohort_root),
            "--deriv-root",
            str(deriv_root),
        ],
    )
    assert mod.main() == 0
    assert mod._planned_artefacts(deriv_root, "0005")["parc_wmparc"].is_file()
    assert not mod._planned_artefacts(deriv_root, "0013")["parc_wmparc"].exists()


def test_main_real_run_skip_existing_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")

    cm = _fake_cohort_metadata(["0005"])
    monkeypatch.setattr(mod, "load_cohort_metadata", lambda y, c: cm)
    argv = [
        "parcellate_all_cohort50.py",
        "--no-dry-run",
        "--backend",
        "dummy",
        "--cohort-root",
        str(cohort_root),
        "--deriv-root",
        str(deriv_root),
        "--skip-existing",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # First call produces the artefacts.
    assert mod.main() == 0
    wmparc_path = mod._planned_artefacts(deriv_root, "0005")["parc_wmparc"]
    mtime_first = wmparc_path.stat().st_mtime_ns

    # Second call must short-circuit; mtime unchanged.
    assert mod.main() == 0
    assert wmparc_path.stat().st_mtime_ns == mtime_first


# ---------------------------------------------------------------------------
# End-to-end integration with the PR-7f Hit-Plot LUT reader
# ---------------------------------------------------------------------------


def test_written_lut_round_trips_through_int_keyed_reader(tmp_path: Path) -> None:
    """The PR-7f cohort runner coerces on-disk string keys back to ``int``.
    Verify the file we write round-trips through that contract without
    data loss."""
    cohort_root = tmp_path / "cohort"
    deriv_root = tmp_path / "deriv"
    _write_synthetic_cohort_subject(cohort_root, "0005")
    mod._real_process_subject(
        "0005",
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        backend="dummy",
        input_channel="T1_bias",
        fs_work_dir=None,
        threads=4,
        skip_existing=False,
    )
    lut_path = mod._planned_artefacts(deriv_root, "0005")["parc_lut"]
    raw = json.loads(lut_path.read_text())
    coerced = {int(k): v for k, v in raw.items()}
    # Coerced dict is a subset of FS_DUMMY_LUT with matching names.
    assert set(coerced).issubset(set(FS_DUMMY_LUT))
    for lab, name in coerced.items():
        assert FS_DUMMY_LUT[lab] == name
