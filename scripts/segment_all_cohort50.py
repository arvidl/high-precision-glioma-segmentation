"""PR-7 cohort fan-out for the unified BraTS-3 DL segmenter.

Two modes:

* ``--dry-run`` (default) --- enumerate the locked n=50 UCSF-PDGM cohort
  (``configs/cohort_ucsfpdgm_n50.yaml``), join with
  ``UCSF-PDGM-metadata_v5.csv``, check that every required input NIfTI
  is on disk, and print --- for each subject --- the per-subject
  artefact paths the *real* runner would write. No inference. This was
  the scoping commit for PR-7 and is the implementation of
  ``make scope-pr7``.

* ``--no-dry-run`` (PR-7d) --- real fan-out runner. For each cohort
  subject, runs :func:`hpgs.segment.unified.predict_brats3` with the
  MONAI Bundle BraTS3 SegResNet (the segmenter locked in
  ``docs/design_segmentation_role.md``), writes the Section-4
  derivatives layout under ``data/derivatives_cohort50/sub-XXXX/``
  (``seg_dl/``, ``metrics/``), and emits a per-subject sidecar with
  Dice / HD95 / |VE| / sensitivity / specificity per WT/TC/ET against
  the dataset-provided reference mask. Failures are logged and the run
  continues with the next subject; the final exit code is non-zero if
  any subject failed or had missing inputs.

The artefact contract printed by ``--dry-run`` and *written* by
``--no-dry-run`` is intentionally identical: Section 4 of
``docs/scope_pr7.md`` is the source of truth.

Usage::

    # Dry-run on the locked cohort YAML (also: 'make scope-pr7'):
    uv run python scripts/segment_all_cohort50.py --dry-run

    # Real run on three subjects on CPU (sanity check before fan-out):
    uv run python scripts/segment_all_cohort50.py --no-dry-run \
        --subject 0005 --subject 0026 --subject 0068 --device cpu

    # Full cohort run, MPS (Apple silicon), via 'make segment-all':
    make segment-all SEGMENT_DEVICE=mps

    # Resume after a crash --- only segment subjects whose seg_dl/ output
    # is missing or stale:
    uv run python scripts/segment_all_cohort50.py --no-dry-run --skip-existing

Exit codes:
    0   success on every requested subject
    1   at least one subject failed (real run) or had missing inputs (dry-run)
    2   could not load the cohort YAML or the metadata CSV
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
    load_nifti,
    normalize_subject_id,
    resolve_subject_inputs,
)
from hpgs.metrics import (
    absolute_volumetric_error,
    dice,
    hausdorff95,
    sensitivity,
    specificity,
)
from hpgs.segment import derive_subregions
from hpgs.segment.unified import CHANNEL_ORDER, PROB_CHANNEL_ORDER, predict_brats3

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")

# Mirror of scripts/smoke_predict_brats3.py so the dry-run probes the
# same on-disk inputs the real fan-out runner consumes.
_INPUT_CHANNELS = ["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"]
_INPUT_TUMOR_SEG = "tumor_segmentation"

# Schema versions of the JSON sidecars written by the real runner.
# Bumping requires coordinated updates to the Table-2 / Table-3
# producers and to docs/scope_pr7.md § 4.
SEG_SIDECAR_SCHEMA_VERSION = "1.0"
METRICS_SIDECAR_SCHEMA_VERSION = "1.0"


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
        help="Run the real PR-7d fan-out segmenter.",
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
        choices=("monai_bundle", "dummy"),
        default="monai_bundle",
        help="Segmentation backend (default: %(default)s).",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device (default: %(default)s).",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional MONAI bundle cache (default: ~/.cache/hpgs/monai_bundles).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Real run: skip subjects whose seg_dl outputs already exist.",
    )
    return p.parse_args()


def _planned_artefacts(deriv_root: Path, subject_id: str) -> dict[str, Path]:
    """Per-subject artefact paths written by the real PR-7 runner.

    Mirrors Section 4 of ``docs/scope_pr7.md``; the dry-run prints
    these paths verbatim and ``--no-dry-run`` writes a subset of them
    (the ``seg_dl_*`` and ``metr_dl_vs_gt`` keys; the rest are written
    by later PR-7 producers and are listed here only so the operator
    sees the full per-subject picture in the dry-run report).
    """
    sid = normalize_subject_id(subject_id)
    sub = deriv_root / f"sub-{sid}"
    seg_dl = sub / "seg_dl"
    parc = sub / "parcellation"
    hp = sub / "hitplot"
    metr = sub / "metrics"
    return {
        "seg_dl_label_map": seg_dl / f"sub-{sid}_seg_brats3_dl.nii.gz",
        "seg_dl_probs": seg_dl / f"sub-{sid}_seg_brats3_dl_probs.nii.gz",
        "seg_dl_sidecar": seg_dl / f"sub-{sid}_seg_brats3_dl.json",
        "parc_wmparc": parc / f"sub-{sid}_wmparc_native.nii.gz",
        "parc_lut": parc / f"sub-{sid}_wmparc_lut.json",
        "hp_dl": hp / f"sub-{sid}_hitplot_dl.csv",
        "hp_dl_prob": hp / f"sub-{sid}_hitplot_dl_prob.csv",
        "hp_gt": hp / f"sub-{sid}_hitplot_gt.csv",
        "hp_raidionics": hp / f"sub-{sid}_hitplot_raidionics.csv",
        "hp_segmentglioma": hp / f"sub-{sid}_hitplot_segmentglioma.csv",
        "hp_tumorsynth": hp / f"sub-{sid}_hitplot_tumorsynth.csv",
        "metr_dl_vs_gt": metr / f"sub-{sid}_metrics_dl_vs_gt.json",
        "metr_dl_vs_raidionics": metr / f"sub-{sid}_metrics_dl_vs_raidionics.json",
        "metr_dl_vs_segmentglioma": metr / f"sub-{sid}_metrics_dl_vs_segmentglioma.json",
        "metr_dl_vs_tumorsynth": metr / f"sub-{sid}_metrics_dl_vs_tumorsynth.json",
    }


def _check_inputs(cohort_root: Path, subject_id: str) -> tuple[bool, str]:
    """Return ``(ok, detail)``: whether every input NIfTI is present."""
    try:
        resolve_subject_inputs(
            cohort_root,
            subject_id,
            channel_names=_INPUT_CHANNELS,
            tumor_segmentation_name=_INPUT_TUMOR_SEG,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    return True, ""


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def _voxel_spacing_mm(affine: np.ndarray) -> tuple[float, float, float]:
    """Per-axis voxel spacing in mm from the NIfTI affine."""
    diag = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return float(diag[0]), float(diag[1]), float(diag[2])


def _per_compartment_full_metrics(
    pred_lm: np.ndarray,
    ref_lm: np.ndarray,
    *,
    voxel_spacing_mm: tuple[float, float, float],
    voxel_volume_mm3: float,
) -> dict[str, dict[str, float | None]]:
    """Dice / HD95 / |VE| / sensitivity / specificity per WT/TC/ET.

    Returns NaN values as ``None`` so the JSON sidecar is valid (NaN
    is not legal JSON). Inf is preserved as ``"inf"`` (string) for the
    same reason --- callers reading the JSON should treat both ``None``
    and ``"inf"`` as "metric undefined / catastrophic" and exclude
    such entries from cohort medians.
    """
    pred = derive_subregions(pred_lm)
    ref = derive_subregions(ref_lm)
    out: dict[str, dict[str, float | None]] = {}
    for name, p_mask, r_mask in (
        ("WT", pred.wt, ref.wt),
        ("TC", pred.tc, ref.tc),
        ("ET", pred.et, ref.et),
    ):
        d = dice(p_mask, r_mask)
        hd = hausdorff95(p_mask, r_mask, voxel_spacing=voxel_spacing_mm)
        ave = absolute_volumetric_error(p_mask, r_mask, voxel_volume_mm3)
        sens = sensitivity(p_mask, r_mask)
        spec = specificity(p_mask, r_mask)
        out[name] = {
            "dice": _json_safe(d),
            "hd95_mm": _json_safe(hd),
            "abs_volumetric_error_mm3": _json_safe(ave),
            "sensitivity": _json_safe(sens),
            "specificity": _json_safe(spec),
        }
    return out


def _json_safe(x: float) -> float | str | None:
    """Replace NaN with ``None`` and Inf with the string ``"inf"``.

    JSON has no native NaN / Inf; the readers in
    ``scripts/build_table2_dl_vs_gt.py`` (later in PR-7) treat ``None``
    and ``"inf"`` as "missing / catastrophic" and exclude such entries
    from medians.
    """
    if isinstance(x, float):
        if np.isnan(x):
            return None
        if np.isinf(x):
            return "inf"
    return float(x)


def _real_process_subject(
    subject_id: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    backend: str,
    device: str,
    bundle_dir: Path | None,
    skip_existing: bool,
) -> dict[str, Any]:
    """Run unified DL segmentation on one subject and write the contract.

    Writes ``seg_dl/sub-XXXX_seg_brats3_dl.{nii.gz,_probs.nii.gz,.json}``
    and ``metrics/sub-XXXX_metrics_dl_vs_gt.json`` per Section 4 of
    ``docs/scope_pr7.md``. Returns the in-memory metrics record so the
    caller can build a cohort-level summary table on the fly.
    """
    sid = normalize_subject_id(subject_id)
    artefacts = _planned_artefacts(deriv_root, sid)
    seg_dl_path: Path = artefacts["seg_dl_label_map"]
    probs_path: Path = artefacts["seg_dl_probs"]
    seg_sidecar_path: Path = artefacts["seg_dl_sidecar"]
    metrics_path: Path = artefacts["metr_dl_vs_gt"]

    if skip_existing and seg_dl_path.exists() and seg_sidecar_path.exists():
        record: dict[str, Any] = {
            "subject_id": sid,
            "skipped": True,
            "reason": f"--skip-existing and {seg_dl_path.name} present",
        }
        return record

    inputs = resolve_subject_inputs(
        cohort_root,
        sid,
        channel_names=_INPUT_CHANNELS,
        tumor_segmentation_name=_INPUT_TUMOR_SEG,
    )
    channels = {
        "T1": inputs.channels["T1_bias"],
        "T1c": inputs.channels["T1c_bias"],
        "T2": inputs.channels["T2_bias"],
        "FLAIR": inputs.channels["FLAIR_bias"],
    }

    t0 = time.time()
    result = predict_brats3(
        t1=channels["T1"],
        t1c=channels["T1c"],
        t2=channels["T2"],
        flair=channels["FLAIR"],
        backend=backend,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        bundle_dir=bundle_dir,
    )
    elapsed_s = time.time() - t0

    ref_lm, ref_affine = load_nifti(inputs.tumor_segmentation)
    if ref_lm.shape != result.label_map.shape:
        raise RuntimeError(
            f"sub-{sid}: reference mask shape {ref_lm.shape} != "
            f"predicted shape {result.label_map.shape}",
        )
    voxel_vol = _voxel_volume_mm3(ref_affine)
    voxel_spacing = _voxel_spacing_mm(ref_affine)
    metrics = _per_compartment_full_metrics(
        result.label_map,
        np.asarray(ref_lm),
        voxel_spacing_mm=voxel_spacing,
        voxel_volume_mm3=voxel_vol,
    )

    seg_dl_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    derivatives_dir(deriv_root, sid)  # ensures the per-subject root exists too

    nib.save(
        nib.Nifti1Image(result.label_map.astype(np.uint8), result.affine),
        str(seg_dl_path),
    )
    # NIfTI's 4-D convention is (X, Y, Z, T): move compartment axis last.
    probs_xyzt = np.moveaxis(result.probabilities, 0, -1).astype(np.float32)
    nib.save(nib.Nifti1Image(probs_xyzt, result.affine), str(probs_path))

    seg_sidecar = {
        "schema_version": SEG_SIDECAR_SCHEMA_VERSION,
        "subject_id": sid,
        "backend": result.backend,
        "bundle_version": result.bundle_version,
        "device": device,
        "elapsed_s": round(float(elapsed_s), 3),
        "voxel_volume_mm3": round(voxel_vol, 6),
        "voxel_spacing_mm": [round(s, 6) for s in voxel_spacing],
        "channel_order": list(CHANNEL_ORDER),
        "probability_channel_order": list(PROB_CHANNEL_ORDER),
        "outputs": {
            "label_map": str(seg_dl_path),
            "probabilities": str(probs_path),
            "metrics_dl_vs_gt": str(metrics_path),
        },
    }
    seg_sidecar_path.write_text(json.dumps(seg_sidecar, indent=2))

    metrics_doc = {
        "schema_version": METRICS_SIDECAR_SCHEMA_VERSION,
        "subject_id": sid,
        "comparison": "dl_vs_gt",
        "voxel_volume_mm3": round(voxel_vol, 6),
        "voxel_spacing_mm": [round(s, 6) for s in voxel_spacing],
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_doc, indent=2))

    return {
        "subject_id": sid,
        "skipped": False,
        "backend": result.backend,
        "bundle_version": result.bundle_version,
        "device": device,
        "elapsed_s": round(float(elapsed_s), 3),
        "voxel_volume_mm3": voxel_vol,
        "metrics": metrics,
        "outputs": seg_sidecar["outputs"],
    }


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
    os_days = metadata_row.get("OS", "?")
    os_tert = metadata_row.get("OS_tertile", "?")
    flag = "OK " if inputs_ok else "MISS"
    print(f"[{flag}] sub-{subject_id}  sex={sex}  age={age}  OS={os_days}d ({os_tert})")
    if not inputs_ok:
        print(f"        missing-input: {inputs_detail}")
    if verbose:
        for key, path in artefacts.items():
            print(f"        WOULD WRITE  [{key:<22}]  {path}")


def _print_dry_summary(n_total: int, n_ok: int, n_missing: int) -> None:
    print()
    print("=" * 72)
    print("PR-7 dry-run summary")
    print("=" * 72)
    print(f"  cohort size       : {n_total}")
    print(f"  inputs present    : {n_ok}")
    print(f"  inputs missing    : {n_missing}")
    print(
        "  artefact contract : Section 4 of docs/scope_pr7.md\n"
        "  next step         : run the real fan-out with --no-dry-run "
        "(or 'make segment-all')"
    )
    print("=" * 72)


def _fmt_metric(value: object, *, width: int = 7, precision: int = 3) -> str:
    if value is None:
        return f"{'nan':>{width}}"
    if isinstance(value, str):
        return f"{value:>{width}}"  # 'inf'
    return f"{float(value):>{width}.{precision}f}"


def _print_real_summary(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("PR-7d cohort-segmentation report (DL vs UCSF-PDGM reference)")
    print("=" * 90)
    print(
        f"{'subject':<10}{'time(s)':>8}"
        f"{'WT-Dice':>9}{'TC-Dice':>9}{'ET-Dice':>9}"
        f"{'WT-HD95':>9}{'TC-HD95':>9}{'ET-HD95':>9}"
        f"{'  status':>20}",
    )
    print("-" * 90)
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    for r in records:
        if r.get("error"):
            n_failed += 1
            print(
                f"{r['subject_id']:<10}{'-':>8}{'-':>9}{'-':>9}{'-':>9}"
                f"{'-':>9}{'-':>9}{'-':>9}{'  FAILED':>20}"
            )
            print(f"        {r['error']}")
            continue
        if r.get("skipped"):
            n_skipped += 1
            print(
                f"{r['subject_id']:<10}{'-':>8}{'-':>9}{'-':>9}{'-':>9}"
                f"{'-':>9}{'-':>9}{'-':>9}{'  SKIPPED':>20}"
            )
            continue
        n_ok += 1
        m = r["metrics"]
        print(
            f"{r['subject_id']:<10}{r['elapsed_s']:>8.1f}"
            f"{_fmt_metric(m['WT']['dice'])}{_fmt_metric(m['TC']['dice'])}"
            f"{_fmt_metric(m['ET']['dice'])}"
            f"{_fmt_metric(m['WT']['hd95_mm'], precision=2)}"
            f"{_fmt_metric(m['TC']['hd95_mm'], precision=2)}"
            f"{_fmt_metric(m['ET']['hd95_mm'], precision=2)}"
            f"{'  OK':>20}",
        )
    print("=" * 90)
    print(
        f"  ok={n_ok}  failed={n_failed}  skipped={n_skipped}\n"
        "  Healthy ranges (heuristic, BraTS bundle published op-point):\n"
        "      WT 0.85-0.95, TC 0.75-0.90, ET 0.70-0.90 Dice."
    )
    print("=" * 90)


def _select_subjects(cm_index: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return list(cm_index)
    norm = [normalize_subject_id(s) for s in requested]
    cohort_set = set(cm_index)
    missing = [s for s in norm if s not in cohort_set]
    if missing:
        raise SystemExit(
            f"error: --subject ids not in cohort YAML: {missing}",
        )
    return norm


def _run_dry(args: argparse.Namespace, cm) -> int:
    print(
        f"PR-7 dry-run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"             cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"             n={cm.n}  seed={cm.seed}",
    )
    print("-" * 72)
    n_ok = 0
    n_missing = 0
    for subject_id in cm.df.index:
        artefacts = _planned_artefacts(args.deriv_root, subject_id)
        inputs_ok, inputs_detail = _check_inputs(args.cohort_root, subject_id)
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
    subjects = _select_subjects(list(cm.df.index), args.subjects)
    print(
        f"PR-7d real run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"               cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"               backend={args.backend}  device={args.device}  "
        f"skip_existing={args.skip_existing}\n"
        f"               n_requested={len(subjects)}  cohort_n={cm.n}  "
        f"seed={cm.seed}",
    )
    print("-" * 90)

    records: list[dict[str, Any]] = []
    for sid in subjects:
        print(f"[run] sub-{sid} ...", flush=True)
        try:
            rec = _real_process_subject(
                sid,
                cohort_root=args.cohort_root,
                deriv_root=args.deriv_root,
                backend=args.backend,
                device=args.device,
                bundle_dir=args.bundle_dir,
                skip_existing=args.skip_existing,
            )
        except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
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


if __name__ == "__main__":
    sys.exit(main())
