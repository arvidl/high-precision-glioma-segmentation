"""PR-7c cohort fan-out for the FreeSurfer 8.2.0 wmparc parcellation.

Two modes:

* ``--dry-run`` (default) --- enumerate the locked n=50 UCSF-PDGM cohort
  (``configs/cohort_ucsfpdgm_n50.yaml``), join with
  ``UCSF-PDGM-metadata_v5.csv``, check that the configured input
  channel NIfTI is on disk per subject, and print the per-subject
  artefact paths the real runner would write. No FreeSurfer invocation.

* ``--no-dry-run`` (PR-7c) --- real fan-out. For each cohort subject,
  calls :func:`hpgs.parcellate.predict_wmparc` (default
  ``backend="freesurfer"``) on the configured bias-corrected mpMRI
  channel (default ``T1_bias``); FreeSurfer produces
  ``wmparc.mgz`` in its conformed space, which is then resampled
  via nearest-neighbour onto the subject's native mpMRI grid.
  Writes the Section-4 parcellation contract under
  ``data/derivatives_cohort50/sub-XXXX/parcellation/``:

  * ``sub-XXXX_wmparc_native.nii.gz`` --- ``int16`` label map on
    the native grid (matches the FLAIR / T1 / T1c / T2 / tumor_seg
    shape exactly, so the PR-7f Hit-Plot cohort runner can load
    parcellation + tumor segmentation without extra resampling);
  * ``sub-XXXX_wmparc_lut.json`` --- ``{label_id (str): name (str)}``
    restricted to labels actually present in the wmparc (the same
    JSON shape the PR-7f `_load_wmparc_lut` helper accepts);
  * ``sub-XXXX_wmparc.json`` --- provenance sidecar
    (``schema_version=1.0``, ``backend``, ``version``,
    ``freesurfer_subjects_dir``, ``input_channel``, ``elapsed_s``,
    ``output paths``) alongside the NIfTI + LUT.

The artefact contract printed by ``--dry-run`` is intentionally
identical to what ``--no-dry-run`` writes; Section 4 of
``docs/scope_pr7.md`` is the source of truth.

The ``dummy`` backend is the CI-runnable companion: it emits a
deterministic synthetic wmparc on the native grid with a 9-region
LUT subset of FreeSurferColorLUT so the PR-7f Hit-Plot cohort
runner can be exercised end-to-end without FreeSurfer on the host.
Per-subject failures are logged and the run continues; the final
exit code is non-zero if any subject failed.

Usage::

    # Dry-run on the locked cohort YAML:
    uv run python scripts/parcellate_all_cohort50.py --dry-run

    # Real run on three subjects (background; FreeSurfer takes
    # 30 min--2 h per subject depending on CPU):
    uv run python scripts/parcellate_all_cohort50.py --no-dry-run \
        --subject 0005 --subject 0026 --subject 0068 \
        --fs-work-dir /scratch/fs_subjects

    # Full-cohort run via the Make wrapper:
    make parcellate-all

    # Dummy-backend smoke test (unblocks the PR-7f Hit-Plot cohort
    # runner on CI / local laptops without FreeSurfer):
    uv run python scripts/parcellate_all_cohort50.py --no-dry-run \
        --backend dummy

    # Resume after a crash --- only parcellate subjects whose
    # wmparc_native.nii.gz is missing:
    uv run python scripts/parcellate_all_cohort50.py --no-dry-run \
        --skip-existing

Exit codes:
    0   success on every requested subject
    1   at least one subject failed (real run) or had missing inputs (dry-run)
    2   could not load the cohort YAML or the metadata CSV,
        or --backend freesurfer without --fs-work-dir

(and the cohort-wide `make hitplot-all` runner) by producing the
per-subject ``wmparc_native`` + LUT the PR-7f runner consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from hpgs.io import (
    derivatives_dir,
    load_cohort_metadata,
    normalize_subject_id,
    resolve_subject_inputs,
)
from hpgs.parcellate import ParcellationResult, predict_wmparc

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")

# Channels accepted by --input-channel. These are exactly the
# bias-corrected mpMRI suffixes PR-7d consumes; we reuse the same set
# so the parcellation and the segmentation share one on-disk input
# convention. The default is T1_bias (standard anatomical contrast
# for parcellation; tumor-bearing T1c_bias can blur GM/WM boundaries
# under contrast enhancement).
_INPUT_CHANNELS: tuple[str, ...] = ("T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias")
DEFAULT_INPUT_CHANNEL: str = "T1_bias"

# Schema version of the provenance sidecar written by this runner.
# Bumping requires coordinated updates to docs/scope_pr7.md § 4 and
# scripts/compute_hitplot_cohort50.py (the one reader of the LUT).
PARC_SIDECAR_SCHEMA_VERSION: str = "1.0"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cohort-yaml",
        type=Path,
        default=DEFAULT_COHORT_YAML,
        help="Cohort YAML (default: %(default)s).",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="UCSF-PDGM v5 metadata CSV (default: %(default)s).",
    )
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
        help="Where to write per-subject derivatives (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Enumerate subjects and print artefact paths (default).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Run the real PR-7c FreeSurfer fan-out.",
    )
    p.add_argument(
        "--show-paths",
        action="store_true",
        help="Print every artefact path per subject (verbose; dry-run only).",
    )
    p.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        metavar="ID",
        help="4-digit subject id; repeat for multiple. Defaults to all cohort subjects.",
    )
    p.add_argument(
        "--backend",
        choices=("freesurfer", "dummy"),
        default="freesurfer",
        help="Parcellation backend (default: %(default)s).",
    )
    p.add_argument(
        "--input-channel",
        choices=_INPUT_CHANNELS,
        default=DEFAULT_INPUT_CHANNEL,
        help="mpMRI channel FreeSurfer is seeded on (default: %(default)s).",
    )
    p.add_argument(
        "--fs-work-dir",
        type=Path,
        default=None,
        help="Scratch SUBJECTS_DIR for FreeSurfer (required for --backend freesurfer).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads forwarded to recon-all-clinical.sh (default: %(default)s).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Real run: skip subjects whose wmparc_native.nii.gz already exists.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Section-4 artefact contract
# ---------------------------------------------------------------------------


def _planned_artefacts(deriv_root: Path, subject_id: str) -> dict[str, Path]:
    """Per-subject parcellation artefact paths Section 4 of docs/scope_pr7.md.

    Mirrors the parcellation/ sub-tree of PR-7d's full artefact
    dictionary; we expose only the three keys PR-7c owns so a
    downstream consumer (PR-7f) can look up the two output paths
    without caring about the full-cohort contract.
    """
    sid = normalize_subject_id(subject_id)
    sub = deriv_root / f"sub-{sid}"
    parc = sub / "parcellation"
    return {
        "parc_wmparc": parc / f"sub-{sid}_wmparc_native.nii.gz",
        "parc_lut": parc / f"sub-{sid}_wmparc_lut.json",
        "parc_sidecar": parc / f"sub-{sid}_wmparc.json",
    }


def _check_inputs(
    cohort_root: Path,
    subject_id: str,
    *,
    input_channel: str,
) -> tuple[bool, str]:
    """Return ``(ok, detail)``: whether the input-channel NIfTI is present."""
    try:
        resolve_subject_inputs(
            cohort_root,
            subject_id,
            channel_names=[input_channel],
            # tumor_segmentation isn't strictly needed for parcellation,
            # but probing it here keeps the dry-run's MISS/OK contract
            # coherent with PR-7d/PR-7e (a subject with an incomplete
            # cohort drop is MISS across the board).
            tumor_segmentation_name="tumor_segmentation",
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    return True, ""


# ---------------------------------------------------------------------------
# Real runner
# ---------------------------------------------------------------------------


def _write_lut_json(lut: dict[int, str], path: Path) -> None:
    """Write the per-subject LUT as ``{str(int): str}`` JSON.

    String-integer keys are the convention documented in Section 4 of
    ``docs/scope_pr7.md`` and expected by
    ``scripts/compute_hitplot_cohort50.py::_load_wmparc_lut``. We pass
    ``sort_keys=False`` but emit keys in ascending numeric order so
    humans reading the JSON get a stable ordering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(int(lab)): str(name) for lab, name in sorted(lut.items())}
    path.write_text(json.dumps(payload, indent=2))


def _real_process_subject(
    subject_id: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    backend: str,
    input_channel: str,
    fs_work_dir: Path | None,
    threads: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Run parcellation on one subject and write the contract.

    Writes ``parcellation/sub-XXXX_wmparc_native.nii.gz`` +
    ``..._wmparc_lut.json`` + ``..._wmparc.json`` per Section 4 of
    ``docs/scope_pr7.md``. Returns the in-memory provenance record so
    the caller can build a cohort-level summary table on the fly.
    """
    sid = normalize_subject_id(subject_id)
    artefacts = _planned_artefacts(deriv_root, sid)
    wmparc_path: Path = artefacts["parc_wmparc"]
    lut_path: Path = artefacts["parc_lut"]
    sidecar_path: Path = artefacts["parc_sidecar"]

    if skip_existing and wmparc_path.exists() and lut_path.exists():
        return {
            "subject_id": sid,
            "skipped": True,
            "reason": f"--skip-existing and {wmparc_path.name} + {lut_path.name} present",
        }

    inputs = resolve_subject_inputs(
        cohort_root,
        sid,
        channel_names=[input_channel],
        tumor_segmentation_name="tumor_segmentation",
    )
    input_nifti = inputs.channels[input_channel]

    t0 = time.time()
    result: ParcellationResult = predict_wmparc(
        input_nifti=input_nifti,
        subject_id=sid,
        backend=backend,  # type: ignore[arg-type]
        fs_work_dir=fs_work_dir,
        threads=threads,
        overwrite=not skip_existing,
    )
    elapsed_s = time.time() - t0

    derivatives_dir(deriv_root, sid)  # ensure per-subject root exists
    wmparc_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(result.label_map.astype(np.int16), result.affine),
        str(wmparc_path),
    )
    _write_lut_json(result.lut, lut_path)

    sidecar = {
        "schema_version": PARC_SIDECAR_SCHEMA_VERSION,
        "subject_id": sid,
        "backend": result.backend,
        "version": result.version,
        "input_channel": input_channel,
        "freesurfer_subjects_dir": str(fs_work_dir) if fs_work_dir is not None else None,
        "threads": int(threads),
        "elapsed_s": round(float(elapsed_s), 3),
        "n_labels_present": len(result.lut),
        "outputs": {
            "wmparc_native": str(wmparc_path),
            "wmparc_lut": str(lut_path),
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return {
        "subject_id": sid,
        "skipped": False,
        "backend": result.backend,
        "version": result.version,
        "elapsed_s": round(float(elapsed_s), 3),
        "n_labels": len(result.lut),
        "shape": tuple(int(s) for s in result.label_map.shape),
        "outputs": sidecar["outputs"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_subject_block(
    subject_id: str,
    metadata_row: dict,
    artefacts: dict[str, Path],
    inputs_ok: bool,
    inputs_detail: str,
    *,
    verbose: bool,
) -> None:
    sex = metadata_row.get("Sex", "?")
    age = metadata_row.get("Age at MRI", "?")
    flag = "OK " if inputs_ok else "MISS"
    print(f"[{flag}] sub-{subject_id}  sex={sex}  age={age}")
    if not inputs_ok:
        print(f"        missing-input: {inputs_detail}")
    if verbose:
        for key, path in artefacts.items():
            print(f"        WOULD WRITE  [{key:<14}]  {path}")


def _print_dry_summary(n_total: int, n_ok: int, n_missing: int) -> None:
    print()
    print("=" * 72)
    print("PR-7c parcellation dry-run summary")
    print("=" * 72)
    print(f"  cohort size       : {n_total}")
    print(f"  inputs present    : {n_ok}")
    print(f"  inputs missing    : {n_missing}")
    print(
        "  artefact contract : Section 4 of docs/scope_pr7.md "
        "(parcellation/wmparc_native.nii.gz + wmparc_lut.json)\n"
        "  next step         : run the real fan-out with --no-dry-run "
        "(or 'make parcellate-all')",
    )
    print("=" * 72)


def _print_real_summary(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 84)
    print("PR-7c parcellation report (FreeSurfer wmparc on native mpMRI grid)")
    print("=" * 84)
    print(
        f"{'subject':<10}{'time(s)':>9}{'n_labels':>10}"
        f"{'backend':>14}{'version':>22}{'  status':>18}",
    )
    print("-" * 84)
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    for r in records:
        if r.get("error"):
            n_failed += 1
            print(
                f"{r['subject_id']:<10}{'-':>9}{'-':>10}{'-':>14}{'-':>22}{'  FAILED':>18}",
            )
            print(f"        {r['error']}")
            continue
        if r.get("skipped"):
            n_skipped += 1
            print(
                f"{r['subject_id']:<10}{'-':>9}{'-':>10}{'-':>14}{'-':>22}{'  SKIPPED':>18}",
            )
            continue
        n_ok += 1
        ver = str(r["version"])[:20]
        print(
            f"{r['subject_id']:<10}{r['elapsed_s']:>9.1f}{r['n_labels']:>10}"
            f"{r['backend']:>14}{ver:>22}{'  OK':>18}",
        )
    print("=" * 84)
    print(f"  ok={n_ok}  failed={n_failed}  skipped={n_skipped}")
    print("=" * 84)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _select_subjects(cm_index: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return list(cm_index)
    norm = [normalize_subject_id(s) for s in requested]
    cohort_set = set(cm_index)
    missing = [s for s in norm if s not in cohort_set]
    if missing:
        raise SystemExit(f"error: --subject ids not in cohort YAML: {missing}")
    return norm


def _run_dry(args: argparse.Namespace, cm) -> int:
    print(
        f"PR-7c dry-run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"              cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"              input_channel={args.input_channel}  "
        f"backend={args.backend}\n"
        f"              n={cm.n}  seed={cm.seed}",
    )
    print("-" * 72)
    n_ok = 0
    n_missing = 0
    for subject_id in cm.df.index:
        artefacts = _planned_artefacts(args.deriv_root, subject_id)
        inputs_ok, inputs_detail = _check_inputs(
            args.cohort_root,
            subject_id,
            input_channel=args.input_channel,
        )
        if inputs_ok:
            n_ok += 1
        else:
            n_missing += 1
        _print_subject_block(
            subject_id,
            cm.df.loc[subject_id].to_dict(),
            artefacts,
            inputs_ok,
            inputs_detail,
            verbose=args.show_paths,
        )
    _print_dry_summary(cm.n, n_ok, n_missing)
    return 0 if n_missing == 0 else 1


def _run_real(args: argparse.Namespace, cm) -> int:
    if args.backend == "freesurfer" and args.fs_work_dir is None:
        print(
            "error: --backend freesurfer requires --fs-work-dir "
            "(scratch SUBJECTS_DIR for recon-all-clinical).",
            file=sys.stderr,
        )
        return 2

    subjects = _select_subjects(list(cm.df.index), args.subjects)
    print(
        f"PR-7c real run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"               cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"               backend={args.backend}  "
        f"input_channel={args.input_channel}  threads={args.threads}\n"
        f"               fs_work_dir={args.fs_work_dir}  "
        f"skip_existing={args.skip_existing}\n"
        f"               n_requested={len(subjects)}  cohort_n={cm.n}  "
        f"seed={cm.seed}",
    )
    print("-" * 84)

    records: list[dict[str, Any]] = []
    for sid in subjects:
        print(f"[run] sub-{sid} ...", flush=True)
        try:
            rec = _real_process_subject(
                sid,
                cohort_root=args.cohort_root,
                deriv_root=args.deriv_root,
                backend=args.backend,
                input_channel=args.input_channel,
                fs_work_dir=args.fs_work_dir,
                threads=args.threads,
                skip_existing=args.skip_existing,
            )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            print(f"[run] sub-{sid}: FAILED: {exc}", file=sys.stderr)
            records.append({"subject_id": sid, "error": str(exc)})
            continue
        records.append(rec)

    _print_real_summary(records)
    n_failed = sum(1 for r in records if r.get("error"))
    return 0 if n_failed == 0 else 1


def main() -> int:
    args = _parse_args()

    try:
        cm = load_cohort_metadata(args.cohort_yaml, args.metadata_csv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: failed to load cohort: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _run_dry(args, cm)
    return _run_real(args, cm)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
