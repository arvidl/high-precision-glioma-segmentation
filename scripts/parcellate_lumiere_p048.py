"""Run FreeSurfer 8.2.0 ``recon-all-clinical`` on every LUMIERE Patient-048 timepoint.

This is the LUMIERE analogue of ``scripts/parcellate_all_cohort50.py``: it
fans the unified anatomical-parcellation backend (``hpgs.parcellate.predict_wmparc``)
out over the six longitudinal timepoints of LUMIERE Patient-048 and writes a
per-timepoint artefact contract that mirrors the UCSF-PDGM cohort
parcellation contract (``wmparc_native.nii.gz`` + ``wmparc_lut.json`` +
sidecar JSON), plus a single cohort-level ``parcellation_summary.{csv,json}``
that the round-3 longitudinal Hit-Plot panel (Figure 11) consumes directly.

Inputs (consumed but not modified)
----------------------------------
- ``configs/lumiere_p048_timepoints.yaml`` -- cohort manifest (single source
  of truth for tp ordering and days).
- ``data/lumiere_p048/derivatives/tp-<week>/tp-<week>_T1_to_ref.nii.gz`` --
  registered T1 channel produced by ``scripts/register_lumiere_p048.py``.
  This is the LUMIERE counterpart of the UCSF-PDGM ``T1_bias`` channel:
  bias-corrected anatomical contrast on the native registered grid; we feed
  it to ``recon-all-clinical.sh`` as the per-tp FreeSurfer subject seed.

Outputs (under the existing ``derivatives_root``)
-------------------------------------------------
::

    derivatives_root/
        parcellation_summary.csv             # one row per tp (Figure 11)
        parcellation_summary.json            # full provenance for the run
        tp-<week>/
            parcellation/
                tp-<week>_wmparc_native.nii.gz   # int16 FS labels on the registered grid
                tp-<week>_wmparc_lut.json        # {"<int>": "<region name>", ...}
                tp-<week>_wmparc.json            # provenance sidecar

The per-tp parcellation tree is schema-identical to the UCSF-PDGM contract
of ``scripts/parcellate_all_cohort50.py`` (Section 4 of ``docs/scope_pr7.md``)
so the downstream Hit-Plot computer can join wmparc_native + label LUT with
the existing ``seg_dl/`` outputs without a separate code path.

Backends
--------
``--backend freesurfer`` (default)
    Real FreeSurfer 8.2.0 ``recon-all-clinical.sh`` per timepoint. Each
    timepoint becomes a *separate* FreeSurfer subject (subject id
    ``p048_<week>``) inside ``--fs-work-dir`` so the longitudinal scans
    are not conflated. Wall-clock is roughly 30-90 minutes per timepoint
    on Apple silicon CPU; total fan-out across the 6 tp is ~3-12 h.
``--backend dummy``
    Deterministic synthetic wmparc on the registered grid via
    :func:`hpgs.parcellate.predict_wmparc`. Exercises the on-disk
    contract end-to-end without FreeSurfer; used by the integration smoke
    test in ``tests/test_parcellate_lumiere_p048.py``.

Usage
-----

    # Sanity dry-run on all six timepoints (no FreeSurfer required):
    uv run python scripts/parcellate_lumiere_p048.py

    # Smoke test the full producer chain with the dummy backend:
    uv run python scripts/parcellate_lumiere_p048.py --no-dry-run --backend dummy

    # Real FreeSurfer fan-out over all six timepoints:
    uv run python scripts/parcellate_lumiere_p048.py \\
        --no-dry-run --backend freesurfer \\
        --fs-work-dir data/lumiere_p048/derivatives/freesurfer

    # Re-run a single timepoint:
    uv run python scripts/parcellate_lumiere_p048.py \\
        --no-dry-run --backend freesurfer \\
        --fs-work-dir data/lumiere_p048/derivatives/freesurfer \\
        --timepoint week-013
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from hpgs.parcellate import ParcellationResult, predict_wmparc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_DERIVATIVES = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"

# Mirrors PARC_SIDECAR_SCHEMA_VERSION in parcellate_all_cohort50.py so the
# UCSF-PDGM and LUMIERE Patient-048 sidecars share one schema version.
PARC_SIDECAR_SCHEMA_VERSION: str = "1.0"
SUMMARY_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Manifest + path helpers.
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _t1_path(derivatives_root: Path, tp_name: str) -> Path:
    """Registered T1 channel that seeds FreeSurfer for this timepoint."""
    return derivatives_root / f"tp-{tp_name}" / f"tp-{tp_name}_T1_to_ref.nii.gz"


def _parc_dir(derivatives_root: Path, tp_name: str) -> Path:
    return derivatives_root / f"tp-{tp_name}" / "parcellation"


def _parc_paths(derivatives_root: Path, tp_name: str) -> dict[str, Path]:
    """Per-tp parcellation artefact paths (Section-4 contract analogue)."""
    parc = _parc_dir(derivatives_root, tp_name)
    base = f"tp-{tp_name}"
    return {
        "wmparc": parc / f"{base}_wmparc_native.nii.gz",
        "lut": parc / f"{base}_wmparc_lut.json",
        "sidecar": parc / f"{base}_wmparc.json",
    }


def _check_inputs(derivatives_root: Path, tp_name: str) -> tuple[bool, str]:
    """Confirm the registered T1 (FS seed) is present for one timepoint."""
    t1 = _t1_path(derivatives_root, tp_name)
    if not t1.is_file():
        return False, f"missing seed channel: {t1}"
    return True, ""


def _fs_subject_id(tp_name: str) -> str:
    """FreeSurfer ``-subjid`` for one timepoint.

    Each timepoint becomes its own FreeSurfer subject so the six longitudinal
    scans are *not* conflated. The ``p048_`` prefix keeps the LUMIERE
    Patient-048 identity in the FS scratch tree even when it lives next to
    other cohorts.
    """
    return f"p048_{tp_name}"


# ---------------------------------------------------------------------------
# Per-timepoint parcellation.
# ---------------------------------------------------------------------------


def _write_lut_json(lut: dict[int, str], path: Path) -> None:
    """Write the per-tp LUT as ``{str(int): str}`` JSON.

    Mirrors ``parcellate_all_cohort50._write_lut_json`` so the LUMIERE and
    UCSF-PDGM LUTs share one on-disk format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(int(lab)): str(name) for lab, name in sorted(lut.items())}
    path.write_text(json.dumps(payload, indent=2))


def _read_existing_sidecar(derivatives_root: Path, tp_name: str) -> dict[str, Any] | None:
    paths = _parc_paths(derivatives_root, tp_name)
    if not (paths["sidecar"].is_file() and paths["wmparc"].is_file() and paths["lut"].is_file()):
        return None
    return json.loads(paths["sidecar"].read_text())


def _parcellate_one_timepoint(
    *,
    tp_name: str,
    derivatives_root: Path,
    backend: str,
    fs_work_dir: Path | None,
    threads: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Run ``predict_wmparc`` on one timepoint and write the contract."""
    seed_path = _t1_path(derivatives_root, tp_name)
    paths = _parc_paths(derivatives_root, tp_name)
    paths["wmparc"].parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result: ParcellationResult = predict_wmparc(
        input_nifti=seed_path,
        subject_id=_fs_subject_id(tp_name),
        backend=backend,  # type: ignore[arg-type]
        fs_work_dir=fs_work_dir,
        threads=threads,
        overwrite=overwrite,
    )
    elapsed_s = time.time() - t0

    nib.save(
        nib.Nifti1Image(result.label_map.astype(np.int16), result.affine),
        str(paths["wmparc"]),
    )
    _write_lut_json(result.lut, paths["lut"])

    sidecar = {
        "schema_version": PARC_SIDECAR_SCHEMA_VERSION,
        "patient_id": "048",
        "timepoint": tp_name,
        "subject_id": _fs_subject_id(tp_name),
        "backend": result.backend,
        "version": result.version,
        "input_channel": "T1_to_ref",
        "input_nifti": str(seed_path),
        "freesurfer_subjects_dir": str(fs_work_dir) if fs_work_dir is not None else None,
        "threads": int(threads),
        "elapsed_s": round(float(elapsed_s), 3),
        "n_labels_present": len(result.lut),
        "shape": list(int(s) for s in result.label_map.shape),
        "outputs": {
            "wmparc_native": str(paths["wmparc"]),
            "wmparc_lut": str(paths["lut"]),
        },
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2))
    return sidecar


# ---------------------------------------------------------------------------
# Cohort-level aggregation.
# ---------------------------------------------------------------------------


CSV_COLUMNS: tuple[str, ...] = (
    "timepoint",
    "days_since_baseline",
    "rano",
    "backend",
    "version",
    "n_labels_present",
    "elapsed_s",
)


def _write_summary(
    *,
    manifest: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    derivatives_root: Path,
    backend: str,
    fs_work_dir: Path | None,
) -> tuple[Path, Path]:
    """Write parcellation_summary.{csv,json} at the derivatives root."""
    csv_path = derivatives_root / "parcellation_summary.csv"
    json_path = derivatives_root / "parcellation_summary.json"

    rows: list[dict[str, Any]] = []
    for tp_meta in manifest["timepoints"]:
        tp = tp_meta["name"]
        sc = sidecars.get(tp)
        if sc is None:
            continue
        rows.append(
            {
                "timepoint": tp,
                "days_since_baseline": tp_meta.get("days_since_baseline"),
                "rano": tp_meta.get("rano"),
                "backend": sc["backend"],
                "version": sc["version"],
                "n_labels_present": sc["n_labels_present"],
                "elapsed_s": sc["elapsed_s"],
            }
        )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "cohort_id": manifest.get("cohort_id"),
        "patient_id": manifest.get("patient_id"),
        "manifest_path": str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        "derivatives_root": str(derivatives_root),
        "backend": backend,
        "freesurfer_subjects_dir": str(fs_work_dir) if fs_work_dir is not None else None,
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Cohort manifest YAML. Default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Root containing the registered channels and where parcellation/ "
            f"subdirs will be written. Default: {DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--timepoint",
        action="append",
        dest="timepoints",
        default=None,
        metavar="NAME",
        help="Only process this timepoint name (e.g. 'week-013'); repeat for several.",
    )
    parser.add_argument(
        "--backend",
        choices=("freesurfer", "dummy"),
        default="freesurfer",
        help="Parcellation backend (default: %(default)s).",
    )
    parser.add_argument(
        "--fs-work-dir",
        type=Path,
        default=None,
        help=(
            "Scratch SUBJECTS_DIR for FreeSurfer (required for --backend freesurfer; "
            "ignored by the dummy backend)."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads forwarded to recon-all-clinical.sh (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Enumerate timepoints and print artefact paths (default).",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Run the real fan-out (FreeSurfer or dummy backend).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip timepoints whose wmparc_native + LUT + sidecar already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run recon-all-clinical and overwrite existing outputs.",
    )
    return parser.parse_args(argv)


def _select_timepoints(
    manifest: dict[str, Any], requested: list[str] | None
) -> list[dict[str, Any]]:
    all_tps = manifest["timepoints"]
    if not requested:
        return all_tps
    by_name = {tp["name"]: tp for tp in all_tps}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(
            f"error: --timepoint values not in manifest: {missing}; available={list(by_name)}"
        )
    return [by_name[name] for name in requested]


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _print_dry_summary(args: argparse.Namespace, timepoints: list[dict[str, Any]]) -> int:
    print(
        f"LUMIERE Patient-048 parcellation dry-run\n"
        f"  manifest          = {args.manifest}\n"
        f"  derivatives_root  = {args.derivatives_root}\n"
        f"  backend           = {args.backend}\n"
        f"  fs_work_dir       = {args.fs_work_dir}\n"
        f"  threads           = {args.threads}\n"
        f"  n_timepoints      = {len(timepoints)}"
    )
    print("-" * 80)
    n_ok = 0
    n_missing = 0
    for tp_meta in timepoints:
        tp = tp_meta["name"]
        ok, detail = _check_inputs(args.derivatives_root, tp)
        flag = "OK " if ok else "MISS"
        rano = tp_meta.get("rano", "?")
        days = tp_meta.get("days_since_baseline", "?")
        print(f"[{flag}] {tp:<14}  day={days:<4} rano={rano}")
        if not ok:
            print(f"        missing-input: {detail}")
            n_missing += 1
            continue
        n_ok += 1
        artefacts = _parc_paths(args.derivatives_root, tp)
        for key in ("wmparc", "lut", "sidecar"):
            print(f"        WOULD WRITE  [{key:<7}]  {artefacts[key]}")
    print("-" * 80)
    print(f"  inputs present : {n_ok}")
    print(f"  inputs missing : {n_missing}")
    print(
        "  artefact contract : tp-<week>/parcellation/{wmparc_native.nii.gz, "
        "wmparc_lut.json, wmparc.json}\n"
        "  next step         : run with --no-dry-run "
        "(use --backend dummy for the smoke test, --backend freesurfer for the real fan-out)"
    )
    return 0 if n_missing == 0 else 1


def _print_real_table(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 92)
    print("LUMIERE Patient-048 parcellation report (FreeSurfer wmparc on registered grid)")
    print("=" * 92)
    print(
        f"{'timepoint':<14}{'status':<10}{'time(s)':>9}{'n_labels':>10}"
        f"{'backend':>14}{'version':>30}"
    )
    print("-" * 92)
    n_ok = n_failed = n_skipped = 0
    for r in records:
        tp = r["timepoint"]
        if r.get("error"):
            n_failed += 1
            print(f"{tp:<14}{'FAILED':<10}{'-':>9}{'-':>10}{'-':>14}{'-':>30}")
            print(f"        {r['error']}")
            continue
        if r.get("status") == "skipped":
            n_skipped += 1
            print(f"{tp:<14}{'SKIPPED':<10}{'-':>9}{'-':>10}{'-':>14}{'-':>30}")
            continue
        n_ok += 1
        sc = r["sidecar"]
        ver = str(sc["version"])[:28]
        print(
            f"{tp:<14}{'OK':<10}{sc['elapsed_s']:>9.1f}{sc['n_labels_present']:>10}"
            f"{sc['backend']:>14}{ver:>30}"
        )
    print("=" * 92)
    print(f"  ok={n_ok}  failed={n_failed}  skipped={n_skipped}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _run_real_loop(
    *,
    timepoints: list[dict[str, Any]],
    derivatives_root: Path,
    backend: str,
    fs_work_dir: Path | None,
    threads: int,
    skip_existing: bool,
    overwrite: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    """Iterate the manifest's tps and run / skip / fail each parcellation.

    Returns ``(sidecars, records, n_failed)``: sidecars keyed by tp name (used
    by ``_write_summary``), per-tp records (used by ``_print_real_table``),
    and the count of hard failures (used as the script's non-zero exit code).
    """
    sidecars: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    n_failed = 0

    for tp_meta in timepoints:
        tp = tp_meta["name"]
        ok, detail = _check_inputs(derivatives_root, tp)
        if not ok:
            print(f"[run] {tp}: FAILED ({detail})", file=sys.stderr)
            records.append({"timepoint": tp, "error": detail})
            n_failed += 1
            continue

        if skip_existing and not overwrite:
            existing = _read_existing_sidecar(derivatives_root, tp)
            if existing is not None:
                print(f"[run] {tp}: SKIPPED (existing parcellation outputs)")
                sidecars[tp] = existing
                records.append({"timepoint": tp, "sidecar": existing, "status": "skipped"})
                continue

        print(f"[run] {tp}: parcellating ...", flush=True)
        try:
            sidecar = _parcellate_one_timepoint(
                tp_name=tp,
                derivatives_root=derivatives_root,
                backend=backend,
                fs_work_dir=fs_work_dir,
                threads=threads,
                overwrite=overwrite or not skip_existing,
            )
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            print(f"[run] {tp}: FAILED: {exc}", file=sys.stderr)
            records.append({"timepoint": tp, "error": str(exc)})
            n_failed += 1
            continue
        sidecars[tp] = sidecar
        records.append({"timepoint": tp, "sidecar": sidecar, "status": "ok"})

    return sidecars, records, n_failed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    derivatives_root = args.derivatives_root.expanduser().resolve()
    args.manifest = manifest_path
    args.derivatives_root = derivatives_root

    if not manifest_path.is_file():
        print(f"[err] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not derivatives_root.is_dir():
        print(
            f"[err] derivatives-root not found: {derivatives_root}\n"
            "  Run scripts/register_lumiere_p048.py first.",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(manifest_path)
    timepoints = _select_timepoints(manifest, args.timepoints)

    if args.dry_run:
        return _print_dry_summary(args, timepoints)

    if args.backend == "freesurfer" and args.fs_work_dir is None:
        print(
            "[err] --backend freesurfer requires --fs-work-dir "
            "(scratch SUBJECTS_DIR for recon-all-clinical).",
            file=sys.stderr,
        )
        return 2

    fs_work_dir = args.fs_work_dir.expanduser().resolve() if args.fs_work_dir is not None else None
    if fs_work_dir is not None:
        fs_work_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"LUMIERE Patient-048 parcellation  manifest={manifest_path}\n"
        f"                                  derivatives_root={derivatives_root}\n"
        f"                                  backend={args.backend}  threads={args.threads}  "
        f"n_timepoints={len(timepoints)}  skip_existing={args.skip_existing}  "
        f"overwrite={args.overwrite}"
    )
    if fs_work_dir is not None:
        print(f"                                  fs_work_dir={fs_work_dir}")
    print("-" * 92)

    sidecars, records, n_failed = _run_real_loop(
        timepoints=timepoints,
        derivatives_root=derivatives_root,
        backend=args.backend,
        fs_work_dir=fs_work_dir,
        threads=args.threads,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )

    csv_path, _json_path = _write_summary(
        manifest=manifest,
        sidecars=sidecars,
        derivatives_root=derivatives_root,
        backend=args.backend,
        fs_work_dir=fs_work_dir,
    )
    _print_real_table(records)
    csv_label = csv_path.relative_to(REPO_ROOT) if csv_path.is_relative_to(REPO_ROOT) else csv_path
    print(f"[ok] cohort summary: {csv_label}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
