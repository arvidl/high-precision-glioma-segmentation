#!/usr/bin/env python3
"""Audit raw TumorSynth inner-tumor labels against DL and reference classes.

The TumorSynth wrapper remaps raw inner labels into the project-wide BraTS
scheme. This script deliberately looks *before* that remap so the local model's
label semantics can be inferred from evidence instead of from aggregate Dice
after a chosen mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hpgs.io import load_nifti, normalize_subject_id, resolve_subject_inputs
from hpgs.metrics import dice
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR

DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_CSV = Path("outputs/tables/tumorsynth_raw_label_crosswalk.csv")
DEFAULT_OUT_JSON = Path("outputs/tables/tumorsynth_raw_label_crosswalk.json")
DEFAULT_SUBJECTS = ("0005", "0026", "0012", "0018", "0035")
INPUT_CHANNELS = ["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"]
TUMOR_SEGMENTATION_NAME = "tumor_segmentation"
RAW_LABELS = (1, 2, 3)
TARGET_CLASSES = {
    "NCR": LABEL_NCR,
    "ED": LABEL_ED,
    "ET": LABEL_ET,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-root",
        type=Path,
        default=DEFAULT_COHORT_ROOT,
        help="Extracted UCSF-PDGM cohort root (default: %(default)s).",
    )
    p.add_argument(
        "--deriv-root",
        type=Path,
        default=DEFAULT_DERIV_ROOT,
        help="Per-subject derivatives root (default: %(default)s).",
    )
    p.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        help="Subject id to audit; repeatable. Defaults to the five-subject TumorSynth subset.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Long-format CSV audit output (default: %(default)s).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Summary JSON audit output (default: %(default)s).",
    )
    return p.parse_args()


def _raw_inner_path(deriv_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return (
        deriv_root / f"sub-{sid}" / "seg_tumorsynth" / f"sub-{sid}_tumorsynth_innertumor_raw.nii.gz"
    )


def _dl_path(deriv_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return deriv_root / f"sub-{sid}" / "seg_dl" / f"sub-{sid}_seg_brats3_dl.nii.gz"


def _load_label_map(path: Path) -> np.ndarray:
    data, _ = load_nifti(path)
    return np.asarray(data).astype(np.int16)


def _shape_checked(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape[:3] != expected:
        raise ValueError(f"{name} shape mismatch: {arr.shape[:3]} vs {expected}")


def audit_subject(*, cohort_root: Path, deriv_root: Path, subject_id: str) -> list[dict[str, Any]]:
    """Return long-form crosswalk rows for one subject."""
    sid = normalize_subject_id(subject_id)
    raw_path = _raw_inner_path(deriv_root, sid)
    dl_path = _dl_path(deriv_root, sid)
    inputs = resolve_subject_inputs(
        cohort_root,
        sid,
        channel_names=INPUT_CHANNELS,
        tumor_segmentation_name=TUMOR_SEGMENTATION_NAME,
    )

    raw = _load_label_map(raw_path)
    dl = _load_label_map(dl_path)
    ref = _load_label_map(inputs.tumor_segmentation)
    _shape_checked(f"sub-{sid} DL", dl, raw.shape[:3])
    _shape_checked(f"sub-{sid} reference", ref, raw.shape[:3])

    comparators = {
        "dl": dl,
        "reference": ref,
    }
    rows: list[dict[str, Any]] = []
    for raw_label in RAW_LABELS:
        raw_mask = raw == raw_label
        raw_voxels = int(raw_mask.sum())
        for comparator_name, comparator_lm in comparators.items():
            for target_name, target_label in TARGET_CLASSES.items():
                target_mask = comparator_lm == target_label
                rows.append(
                    {
                        "subject_id": sid,
                        "raw_label": raw_label,
                        "comparator": comparator_name,
                        "target_class": target_name,
                        "dice": float(dice(raw_mask, target_mask)),
                        "intersection_voxels": int(np.logical_and(raw_mask, target_mask).sum()),
                        "raw_label_voxels": raw_voxels,
                        "target_voxels": int(target_mask.sum()),
                    },
                )
    return rows


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject_id"], int(row["raw_label"]), row["comparator"])].append(row)

    best_rows: list[dict[str, Any]] = []
    vote_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (subject_id, raw_label, comparator), candidates in sorted(grouped.items()):
        best = max(
            candidates,
            key=lambda row: (float(row["dice"]), int(row["intersection_voxels"])),
        )
        best_record = {
            "subject_id": subject_id,
            "raw_label": raw_label,
            "comparator": comparator,
            "best_target_class": best["target_class"],
            "best_dice": best["dice"],
            "intersection_voxels": best["intersection_voxels"],
            "raw_label_voxels": best["raw_label_voxels"],
            "target_voxels": best["target_voxels"],
        }
        best_rows.append(best_record)
        vote_counts[f"raw_{raw_label}|{comparator}"][best["target_class"]] += 1

    return {
        "schema_version": "1.0",
        "n_rows": len(rows),
        "subjects": sorted({row["subject_id"] for row in rows}),
        "raw_labels": list(RAW_LABELS),
        "target_classes": list(TARGET_CLASSES),
        "best_match_per_subject_label_comparator": best_rows,
        "vote_counts": {key: dict(counter) for key, counter in sorted(vote_counts.items())},
    }


def write_outputs(rows: list[dict[str, Any]], *, out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "raw_label",
        "comparator",
        "target_class",
        "dice",
        "intersection_voxels",
        "raw_label_voxels",
        "target_voxels",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(_summarise(rows), indent=2))


def main() -> int:
    args = _parse_args()
    subjects = args.subjects if args.subjects else list(DEFAULT_SUBJECTS)
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        rows.extend(
            audit_subject(
                cohort_root=args.cohort_root,
                deriv_root=args.deriv_root,
                subject_id=subject,
            ),
        )
    write_outputs(rows, out_csv=args.out_csv, out_json=args.out_json)
    print(f"[audit] wrote {args.out_csv} and {args.out_json} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
