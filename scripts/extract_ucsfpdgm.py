"""Extract the UCSF-PDGM cohort used by HPGS.

Mirrors the recommendation in REVISION_PLAN.md (Section 6b).

Usage
-----
    python scripts/extract_ucsfpdgm.py --src /path/to/UCSF-PDGM-v5 \
                                       --dst ./data/ucsf_pdgm_cohort50
    python scripts/extract_ucsfpdgm.py --src /path/to/UCSF-PDGM-v5 \
                                       --dst ./data/ucsf_pdgm_legacy5 \
                                       --subjects 0020,0022,0039,0066,0085
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Selected cohort: GBM IDH-wildtype WHO grade 4 baseline, no prior biopsy, OS known,
# excluded from BraTS21 segmentation training. Stratified Sex x OS-tertile, seed=20260415.
SUBJECTS: list[str] = [
    "0005",
    "0012",
    "0018",
    "0019",
    "0026",
    "0035",
    "0047",
    "0064",
    "0068",
    "0071",
    "0076",
    "0082",
    "0113",
    "0122",
    "0123",
    "0129",
    "0141",
    "0142",
    "0145",
    "0153",
    "0155",
    "0161",
    "0165",
    "0166",
    "0176",
    "0182",
    "0204",
    "0210",
    "0229",
    "0270",
    "0279",
    "0306",
    "0314",
    "0356",
    "0388",
    "0389",
    "0392",
    "0396",
    "0399",
    "0401",
    "0402",
    "0409",
    "0420",
    "0431",
    "0457",
    "0461",
    "0463",
    "0494",
    "0505",
    "0519",
]

CORE_SUFFIXES: list[str] = [
    "T1_bias.nii.gz",
    "T1c_bias.nii.gz",
    "T2_bias.nii.gz",
    "FLAIR_bias.nii.gz",
    "tumor_segmentation.nii.gz",
    "brain_parenchyma_segmentation.nii.gz",
    "brain_segmentation.nii.gz",
]

OPTIONAL_SUFFIXES: list[str] = [
    "T1.nii.gz",  # raw T1 as SynthSR fallback
    "ADC.nii.gz",
    "DTI_eddy_FA.nii.gz",
    "DTI_eddy_MD.nii.gz",
    "SWI_bias.nii.gz",
    "ASL.nii.gz",
]


def _normalize_subject_id(subject_id: str) -> str:
    normalized = subject_id.strip().removeprefix("sub-")
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError(f"Expected a 4-digit UCSF-PDGM id, got {subject_id!r}")
    return normalized


def parse_subjects_arg(raw: str) -> list[str]:
    subjects = [_normalize_subject_id(part) for part in raw.split(",") if part.strip()]
    if not subjects:
        raise ValueError("Expected at least one subject id in --subjects.")
    return subjects


def main(
    src: Path,
    dst: Path,
    *,
    include_optional: bool = True,
    subjects: list[str] | None = None,
) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    suffixes = CORE_SUFFIXES + (OPTIONAL_SUFFIXES if include_optional else [])
    selected_subjects = subjects if subjects is not None else SUBJECTS

    n_copied = 0
    n_missing = 0
    for sid in selected_subjects:
        src_dir = src / f"UCSF-PDGM-{sid}_nifti"
        out_dir = dst / f"sub-{sid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for suf in suffixes:
            src_file = src_dir / f"UCSF-PDGM-{sid}_{suf}"
            if not src_file.exists():
                print(f"[warn] missing {src_file}")
                n_missing += 1
                continue
            shutil.copy2(src_file, out_dir / f"sub-{sid}_{suf}")
            n_copied += 1
    print(
        f"[ok] copied {n_copied} files for {len(selected_subjects)} subjects ({n_missing} missing)."
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="UCSF-PDGM v5 root directory.")
    p.add_argument(
        "--dst", type=Path, default=Path("./data/ucsf_pdgm_cohort50"), help="Destination root."
    )
    p.add_argument(
        "--no-optional", action="store_true", help="Copy only the 7 core files per subject."
    )
    p.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated 4-digit UCSF-PDGM ids to override the default n=50 cohort.",
    )
    args = p.parse_args()
    subjects = parse_subjects_arg(args.subjects) if args.subjects else None
    main(args.src, args.dst, include_optional=not args.no_optional, subjects=subjects)


if __name__ == "__main__":
    _cli()
