"""Compute the per-subject GT Hit-Plot CSV for the legacy-5 detail subjects.

Five UCSF-PDGM subjects (020, 022, 039, 066, 085) are showcased in the
manuscript's per-subject anatomical-profile figures (Figs. 4, 5, 12, 13,
14 of ``paper/main.tex``). Per
``data/UCSF-PDGM-metadata_v5.csv``, four of them sit in the BraTS21
Segmentation *Training* cohort and one in *Validation*; the unified DL
engine is BraTS21-derived, so running DL inference on these subjects
would punch a hole through the leak-safety guarantee that the
``configs/cohort_ucsfpdgm_n50.yaml`` evaluation cohort is built on. This
runner is therefore intentionally GT-only (no DL involved).

Inputs (all on disk for the legacy-5 tree):

* ``data/derivatives_legacy5/sub-XXXX/parcellation/wmparc.nii.gz``
  --- FreeSurfer 8.2.0 ``recon-all-clinical`` whole-brain
  white-matter+cortical parcellation (~100 regions).
* ``data/ucsf_pdgm_legacy5/sub-XXXX/sub-XXXX_tumor_segmentation.nii.gz``
  --- the dataset-provided three-compartment tumor mask
  (NCR=1, ED=2, ET=4) on the same native voxel grid as the wmparc.
* ``$FREESURFER_HOME/FreeSurferColorLUT.txt`` --- canonical FreeSurfer
  label-name table; we filter it to labels actually present in the
  per-subject wmparc and write the result alongside the CSV.

Outputs (one set per subject, mirroring the cohort-50 contract):

* ``data/derivatives_legacy5/sub-XXXX/hitplot/sub-XXXX_hitplot_gt.csv``
  --- long-format Hit-Plot dataframe identical in schema to
  ``compute_hitplot_cohort50.py`` so downstream consumers (the bar-chart
  renderer, the agreement panel) read both trees with the same code.
* ``data/derivatives_legacy5/sub-XXXX/parcellation/sub-XXXX_wmparc_lut.json``
  --- ``{label_id (str): name (str)}`` filtered to labels actually
  present, so the legacy-5 tree carries the same LUT contract as the
  cohort-50 derivatives.

Usage::

    # Default (all five subjects, dry-run prints the plan):
    uv run python scripts/compute_legacy5_hitplot_gt.py
    uv run python scripts/compute_legacy5_hitplot_gt.py --no-dry-run

    # Subset:
    uv run python scripts/compute_legacy5_hitplot_gt.py --no-dry-run \\
        --subject 0020

Exit codes:
    0   every subject succeeded or was skipped cleanly
    1   at least one subject failed (parcellation or GT missing/malformed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hpgs.hitplot import compartment_region_matrix
from hpgs.io import load_nifti, normalize_subject_id
from hpgs.parcellate import synthseg_label_lut


def _restrict_lut_to_present_labels(
    lut: dict[int, str],
    label_map: np.ndarray,
) -> dict[int, str]:
    """Drop LUT entries whose label is absent from ``label_map``.

    A small helper inlined from ``hpgs.parcellate`` (private there) to keep
    this script's import surface to the public API only. Behaviour is
    identical: 2-line numpy filter, no allocation surprises.
    """
    present = np.unique(label_map).tolist()
    return {int(lab): lut[int(lab)] for lab in present if int(lab) in lut}


DEFAULT_COHORT_YAML = Path("configs/cohort_legacy5.yaml")
DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_legacy5")
DEFAULT_DERIV_ROOT = Path("data/derivatives_legacy5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-yaml", type=Path, default=DEFAULT_COHORT_YAML)
    p.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    p.add_argument("--deriv-root", type=Path, default=DEFAULT_DERIV_ROOT)
    p.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        metavar="ID",
        help="4-digit subject id; repeat for multiple. Defaults to all in YAML.",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Enumerate planned (subject, output) pairs and exit (default).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Execute the real Hit-Plot computation and write CSVs.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Real run: skip subjects whose hitplot_gt.csv already exists.",
    )
    return p.parse_args()


def _load_cohort_subjects(yaml_path: Path) -> list[str]:
    """Load the ordered subject id list from the legacy-5 cohort YAML."""
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    doc = yaml.safe_load(yaml_path.read_text())
    if not isinstance(doc, dict) or "subjects" not in doc:
        raise ValueError(f"{yaml_path}: missing 'subjects' top-level key")
    raw = doc["subjects"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{yaml_path}: 'subjects' must be a non-empty list")
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, dict) and "id" in entry:
            out.append(normalize_subject_id(str(entry["id"])))
        elif isinstance(entry, str):
            out.append(normalize_subject_id(entry))
        else:
            raise ValueError(f"{yaml_path}: unrecognised subject entry {entry!r}")
    return out


def _select_subjects(cohort: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return list(cohort)
    allowed = {sid for sid in cohort}
    seen: set[str] = set()
    out: list[str] = []
    for token in requested:
        for raw in str(token).split():
            sid = normalize_subject_id(raw)
            if sid in seen:
                continue
            if sid not in allowed:
                raise ValueError(
                    f"subject {raw!r} not in legacy-5 cohort (allowed: {sorted(allowed)})",
                )
            seen.add(sid)
            out.append(sid)
    return out


def _planned_paths(
    deriv_root: Path,
    cohort_root: Path,
    subject_id: str,
) -> dict[str, Path]:
    sid = normalize_subject_id(subject_id)
    sub_deriv = deriv_root / f"sub-{sid}"
    sub_cohort = cohort_root / f"sub-{sid}"
    return {
        "wmparc": sub_deriv / "parcellation" / "wmparc.nii.gz",
        "wmparc_lut": sub_deriv / "parcellation" / f"sub-{sid}_wmparc_lut.json",
        "tumor_seg": sub_cohort / f"sub-{sid}_tumor_segmentation.nii.gz",
        "out_csv": sub_deriv / "hitplot" / f"sub-{sid}_hitplot_gt.csv",
    }


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def _write_lut_json(lut: dict[int, str], path: Path) -> None:
    """Write the per-subject LUT as ``{str(int): str}`` JSON.

    String-integer keys mirror the cohort-50 convention (Section 4 of
    ``docs/scope_pr7.md``) so consumers reading either tree see the
    same shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(int(lab)): str(name) for lab, name in sorted(lut.items())}
    path.write_text(json.dumps(payload, indent=2))


def _process_subject(  # noqa: PLR0911 - early-return-per-precondition is the clearest control flow
    subject_id: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Compute one subject's GT Hit-Plot CSV and write the LUT sidecar."""
    sid = normalize_subject_id(subject_id)
    paths = _planned_paths(deriv_root, cohort_root, sid)
    record: dict[str, Any] = {"subject_id": sid, "out_csv": str(paths["out_csv"])}

    if skip_existing and paths["out_csv"].is_file():
        record["status"] = "skipped"
        record["reason"] = f"--skip-existing and {paths['out_csv'].name} present"
        return record

    if not paths["wmparc"].is_file():
        record["status"] = "missing_input"
        record["reason"] = f"wmparc absent: {paths['wmparc']}"
        return record
    if not paths["tumor_seg"].is_file():
        record["status"] = "missing_input"
        record["reason"] = f"tumor_segmentation absent: {paths['tumor_seg']}"
        return record

    t0 = time.time()
    try:
        wmparc, parc_affine = load_nifti(paths["wmparc"])
        wmparc = np.asarray(wmparc).astype(np.int32)
        tumor_seg, tumor_affine = load_nifti(paths["tumor_seg"])
        tumor_seg = np.asarray(tumor_seg).astype(np.int32)
    except (FileNotFoundError, ValueError) as exc:
        record["status"] = "error"
        record["reason"] = f"failed to load NIfTI: {exc}"
        return record

    if wmparc.shape != tumor_seg.shape:
        record["status"] = "error"
        record["reason"] = f"shape mismatch: wmparc {wmparc.shape} != tumor {tumor_seg.shape}"
        return record

    if not np.allclose(parc_affine, tumor_affine, atol=1e-3):
        # Same-shape but different-affine inputs would silently overlay
        # the wrong voxels; refuse rather than emit misleading numbers.
        record["status"] = "error"
        record["reason"] = "wmparc and tumor_segmentation have differing affines"
        return record

    voxel_volume = _voxel_volume_mm3(parc_affine)

    full_lut = synthseg_label_lut()
    label_lut = _restrict_lut_to_present_labels(full_lut, wmparc)
    if not label_lut:
        record["status"] = "error"
        record["reason"] = "wmparc has no labels present in FreeSurfer LUT"
        return record

    df = compartment_region_matrix(
        tumor_seg,
        wmparc,
        label_lut,
        voxel_volume_mm3=voxel_volume,
    )

    paths["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["out_csv"], index=False)
    _write_lut_json(label_lut, paths["wmparc_lut"])

    record["status"] = "ok"
    record["n_rows"] = len(df)
    record["n_labels"] = len(label_lut)
    record["voxel_volume_mm3"] = voxel_volume
    record["elapsed_s"] = time.time() - t0
    return record


def _print_dry_run(deriv_root: Path, cohort_root: Path, subjects: list[str]) -> None:
    print("=============================================================================")
    print(
        f"PR-PHASEC  legacy-5 GT Hit-Plot dry-run  n_subjects={len(subjects)}",
    )
    print("=============================================================================")
    for sid in subjects:
        paths = _planned_paths(deriv_root, cohort_root, sid)
        wmparc_ok = "OK" if paths["wmparc"].is_file() else "MISS"
        tumor_ok = "OK" if paths["tumor_seg"].is_file() else "MISS"
        out_state = "EXISTS" if paths["out_csv"].is_file() else "TODO"
        print(
            f"  sub-{sid}  wmparc={wmparc_ok}  tumor={tumor_ok}  "
            f"out={out_state}  {paths['out_csv']}"
        )
    print("=============================================================================")
    print("(re-run with --no-dry-run to execute)")


def _print_summary(records: list[dict[str, Any]]) -> int:
    print("=============================================================================")
    print(
        f"PR-PHASEC  legacy-5 GT Hit-Plot summary  n={len(records)}",
    )
    print("=============================================================================")
    n_err = 0
    for r in records:
        sid = r["subject_id"]
        status = r.get("status", "?")
        if status == "ok":
            print(
                f"  sub-{sid} ok  n_rows={r['n_rows']}  n_labels={r['n_labels']}  "
                f"vox={r['voxel_volume_mm3']:.3f}mm^3  t={r['elapsed_s']:.2f}s"
            )
        elif status == "skipped":
            print(f"  sub-{sid} skipped  ({r.get('reason', '')})")
        elif status == "missing_input":
            print(f"  sub-{sid} missing_input  {r.get('reason', '')}")
            n_err += 1
        else:
            print(f"  sub-{sid} {status}  {r.get('reason', '')}")
            n_err += 1
    print("=============================================================================")
    return 1 if n_err else 0


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        # Argv override path: parse explicit list (used in tests). The
        # default path uses sys.argv via _parse_args() above.
        sys.argv = ["compute_legacy5_hitplot_gt.py", *argv]
    args = _parse_args()
    try:
        cohort = _load_cohort_subjects(args.cohort_yaml)
        subjects = _select_subjects(cohort, args.subjects)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_dry_run(args.deriv_root, args.cohort_root, subjects)
        return 0

    records = [
        _process_subject(
            sid,
            cohort_root=args.cohort_root,
            deriv_root=args.deriv_root,
            skip_existing=args.skip_existing,
        )
        for sid in subjects
    ]
    return _print_summary(records)


if __name__ == "__main__":
    raise SystemExit(main())
