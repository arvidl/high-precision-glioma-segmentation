from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_ucsfpdgm.py"
SPEC = importlib.util.spec_from_file_location("extract_ucsfpdgm", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load script module from {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CORE_SUFFIXES = MODULE.CORE_SUFFIXES
OPTIONAL_SUFFIXES = MODULE.OPTIONAL_SUFFIXES
main = MODULE.main
parse_subjects_arg = MODULE.parse_subjects_arg


def _write_source_subject(root: Path, subject_id: str) -> None:
    src_dir = root / f"UCSF-PDGM-{subject_id}_nifti"
    src_dir.mkdir(parents=True)
    for suffix in CORE_SUFFIXES + OPTIONAL_SUFFIXES:
        (src_dir / f"UCSF-PDGM-{subject_id}_{suffix}").write_bytes(b"nifti")


def test_parse_subjects_arg_normalizes_ids() -> None:
    assert parse_subjects_arg("0020, sub-0022 ,0039") == ["0020", "0022", "0039"]


def test_extract_ucsfpdgm_supports_subject_override(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_source_subject(src, "0020")
    _write_source_subject(src, "0085")

    main(src, dst, include_optional=False, subjects=["0020"])

    assert (dst / "sub-0020" / "sub-0020_T1_bias.nii.gz").is_file()
    assert not (dst / "sub-0085").exists()
    assert not (dst / "sub-0020" / "sub-0020_T1.nii.gz").exists()
