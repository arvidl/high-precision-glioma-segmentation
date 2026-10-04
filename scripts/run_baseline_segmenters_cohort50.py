"""PR-7e cohort fan-out for external segmenter baselines.

Runs external baseline segmenters on every subject in the locked n=50
UCSF-PDGM cohort and writes the per-subject artefacts that the PR-7h
Table 3 producer and the PR-7i Hit-Plot agreement panel can consume.

Per subject and per baseline, this script writes (extending Section 4
of ``docs/scope_pr7.md`` with two new sub-trees -- ``seg_raidionics/``
and ``seg_segmentglioma/`` -- one mirror per baseline)::

    data/derivatives_cohort50/sub-XXXX/
      seg_raidionics/
        sub-XXXX_seg_raidionics.nii.gz       # uint8 BraTS labels {0,1,2,4}
        sub-XXXX_seg_raidionics.json         # backend, version, runtime, paths
      seg_segmentglioma/
        sub-XXXX_seg_segmentglioma.nii.gz
        sub-XXXX_seg_segmentglioma.json
      seg_tumorsynth/
        sub-XXXX_seg_tumorsynth.nii.gz
        sub-XXXX_seg_tumorsynth.json
      metrics/
        sub-XXXX_metrics_dl_vs_raidionics.json
        sub-XXXX_metrics_dl_vs_segmentglioma.json
        sub-XXXX_metrics_dl_vs_tumorsynth.json

The paired metrics file uses the convention **prediction = DL,
reference = baseline**, so on a four-column Table 3 a row reads "DL
under each segmenter taken as the reference". Dice / HD95 / |VE| are
symmetric so the convention only matters for sensitivity / specificity
(de-emphasised in Q2 = C, but still recorded so that the sidecar is a
strict superset of Table 2's per-compartment metric bundle and the
schema is identical).

Usage::

    # Dry-run on the locked cohort YAML: print the artefact paths that
    # WOULD be written, no inference.
    uv run python scripts/run_baseline_segmenters_cohort50.py --dry-run

    # Real run with the dummy backend (CI / smoke -- writes valid
    # artefacts that the PR-7h producer can already consume):
    uv run python scripts/run_baseline_segmenters_cohort50.py --no-dry-run \
        --backend dummy --subject 0005 --subject 0026

    # Full cohort run (PR-7e2; once the docker backend lands):
    make baselines-all BASELINE_BACKEND=docker

    # TumorSynth as an additional comparator, using the local wrapper:
    make tumorsynth-all TUMORSYNTH_NNUNET_DIR=/path/to/nnUNet

    # Resume after a crash -- only run baselines whose seg_<baseline>/
    # output is missing or stale:
    uv run python scripts/run_baseline_segmenters_cohort50.py --no-dry-run \
        --skip-existing

Exit codes::

    0   success on every requested (subject, baseline) pair
    1   at least one (subject, baseline) failed (real run) or had
        missing inputs (dry-run)
    2   could not load the cohort YAML or the metadata CSV

**Hit-Plot agreement panel** (R2.8) by producing the per-subject
baseline label maps + paired-metric JSONs that the table / panel
producers consume.
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
from hpgs.segment.baselines import (
    BASELINES,
    TUMORSYNTH_INNER_ED_LABELS,
    TUMORSYNTH_INNER_ET_LABELS,
    TUMORSYNTH_INNER_NCR_LABELS,
    TUMORSYNTH_WHOLE_TUMOR_LABEL,
    predict_baseline,
)

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")

# Mirror of scripts/segment_all_cohort50.py so the two cohort runners
# probe the same on-disk inputs and the dry-run reports are comparable.
_INPUT_CHANNELS = ["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"]
_INPUT_TUMOR_SEG = "tumor_segmentation"

# Schema versions of the JSON sidecars written by the real runner.
# Bumping requires coordinated updates to the Table 3 producer (PR-7h)
# and to docs/scope_pr7.md Section 4.
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
        help="Enumerate (subject, baseline) pairs and print artefact paths (default).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Run the real PR-7e baseline fan-out.",
    )
    p.add_argument(
        "--show-paths",
        action="store_true",
        help="Print every artefact path per (subject, baseline) (verbose; dry-run only).",
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
        "--baseline",
        action="append",
        dest="baselines",
        default=None,
        choices=(*BASELINES, "all"),
        help=(
            "Baseline to run; repeat for multiple. 'all' expands to all known "
            "baselines (default: all)."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("docker", "command", "dummy"),
        default="docker",
        help=(
            "Baseline backend (default: %(default)s). Use 'command' for mri_TumorSynth; "
            "the generic docker backend is not wired in this repository."
        ),
    )
    p.add_argument(
        "--tumorsynth-command",
        default="mri_TumorSynth",
        help="TumorSynth executable for backend='command' (default: %(default)s).",
    )
    p.add_argument(
        "--tumorsynth-nnunet-dir",
        type=Path,
        default=None,
        help=(
            "TumorSynth nnU-Net v1.7 model root. If omitted, NNUNET_ENV_DIR is used "
            "(TumorSynth only)."
        ),
    )
    p.add_argument(
        "--tumorsynth-threads",
        type=int,
        default=8,
        help="Threads passed to mri_TumorSynth (default: %(default)s; -1 = all).",
    )
    p.add_argument(
        "--tumorsynth-cpu",
        action="store_true",
        help="Pass --cpu to mri_TumorSynth.",
    )
    p.add_argument(
        "--tumorsynth-work-dir",
        type=Path,
        default=None,
        help="Optional parent directory for temporary TumorSynth ROI/intermediate files.",
    )
    p.add_argument(
        "--tumorsynth-whole-tumor-label",
        type=int,
        default=TUMORSYNTH_WHOLE_TUMOR_LABEL,
        help="Label interpreted as whole tumor in TumorSynth whole-tumor output.",
    )
    p.add_argument(
        "--tumorsynth-inner-ncr-labels",
        default=",".join(str(x) for x in TUMORSYNTH_INNER_NCR_LABELS),
        help="Comma-separated inner-tumor labels remapped to BraTS NCR (default: %(default)s).",
    )
    p.add_argument(
        "--tumorsynth-inner-ed-labels",
        default=",".join(str(x) for x in TUMORSYNTH_INNER_ED_LABELS),
        help="Comma-separated inner-tumor labels remapped to BraTS ED (default: %(default)s).",
    )
    p.add_argument(
        "--tumorsynth-inner-et-labels",
        default=",".join(str(x) for x in TUMORSYNTH_INNER_ET_LABELS),
        help="Comma-separated inner-tumor labels remapped to BraTS ET (default: %(default)s).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Real run: skip (subject, baseline) pairs whose outputs already exist.",
    )
    p.add_argument(
        "--require-dl",
        dest="require_dl",
        action="store_true",
        default=True,
        help=(
            "Real run: fail subject if seg_dl/sub-XXXX_seg_brats3_dl.nii.gz "
            "(PR-7d output) is missing (default; needed for paired metrics)."
        ),
    )
    p.add_argument(
        "--no-require-dl",
        dest="require_dl",
        action="store_false",
        help=(
            "Real run: keep going even if the DL output is missing for a "
            "subject; only the seg_<baseline> mask is written, the paired "
            "metrics JSON is skipped with a warning."
        ),
    )
    return p.parse_args()


def _parse_int_csv(raw: str, *, option_name: str) -> tuple[int, ...]:
    """Parse comma-separated integer label options for TumorSynth remapping."""
    labels: list[int] = []
    for raw_token in str(raw).split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            labels.append(int(token))
        except ValueError as exc:
            raise SystemExit(
                f"error: {option_name} must be a comma-separated integer list",
            ) from exc
    if not labels:
        raise SystemExit(f"error: {option_name} must contain at least one label")
    return tuple(labels)


# ---------------------------------------------------------------------------
# Section 4 contract: per-(subject, baseline) artefact paths.
# ---------------------------------------------------------------------------


def _planned_baseline_artefacts(
    deriv_root: Path,
    subject_id: str,
    baseline: str,
) -> dict[str, Path]:
    """Per-(subject, baseline) artefact paths written by PR-7e.

    Mirrors the section-4 contract style of
    :func:`scripts.segment_all_cohort50._planned_artefacts` and *extends*
    it with two new sub-trees (one per baseline). Keys:

    * ``seg_label_map`` -- ``seg_<baseline>/sub-XXXX_seg_<baseline>.nii.gz``
    * ``seg_sidecar``   -- ``seg_<baseline>/sub-XXXX_seg_<baseline>.json``
    * ``metr_dl_vs_baseline`` -- ``metrics/sub-XXXX_metrics_dl_vs_<baseline>.json``
    * ``seg_dl_label_map`` -- ``seg_dl/sub-XXXX_seg_brats3_dl.nii.gz`` (input
      from PR-7d; included so the dry-run can probe it).
    """
    sid = normalize_subject_id(subject_id)
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline: {baseline!r} (expected one of {list(BASELINES)})",
        )
    sub = deriv_root / f"sub-{sid}"
    seg_b = sub / f"seg_{baseline}"
    metr = sub / "metrics"
    seg_dl = sub / "seg_dl"
    return {
        "seg_label_map": seg_b / f"sub-{sid}_seg_{baseline}.nii.gz",
        "seg_sidecar": seg_b / f"sub-{sid}_seg_{baseline}.json",
        "metr_dl_vs_baseline": metr / f"sub-{sid}_metrics_dl_vs_{baseline}.json",
        "seg_dl_label_map": seg_dl / f"sub-{sid}_seg_brats3_dl.nii.gz",
        **(
            {
                "seg_tumorsynth_raw_whole": (seg_b / f"sub-{sid}_tumorsynth_wholetumor_raw.nii.gz"),
                "seg_tumorsynth_raw_inner": (seg_b / f"sub-{sid}_tumorsynth_innertumor_raw.nii.gz"),
            }
            if baseline == "tumorsynth"
            else {}
        ),
    }


def _check_inputs(cohort_root: Path, subject_id: str) -> tuple[bool, str]:
    """Return ``(ok, detail)``: whether every input NIfTI is present.

    Same probe as PR-7d so the two dry-runs report identical missing-input
    diagnostics.
    """
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


# ---------------------------------------------------------------------------
# Voxel + JSON helpers (mirrored verbatim from PR-7d so the metric files
# the two runners write share an identical schema and the Table 3 / Hit-Plot
# producers can read them with one parser).
# ---------------------------------------------------------------------------


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def _voxel_spacing_mm(affine: np.ndarray) -> tuple[float, float, float]:
    diag = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return float(diag[0]), float(diag[1]), float(diag[2])


def _json_safe(x: float) -> float | str | None:
    """Replace NaN with ``None`` and Inf with the string ``"inf"``.

    JSON has no native NaN / Inf; the readers in
    ``scripts/build_table*.py`` treat ``None`` and ``"inf"`` as
    "missing / catastrophic" and exclude such entries from medians.
    """
    if isinstance(x, float):
        if np.isnan(x):
            return None
        if np.isinf(x):
            return "inf"
    return float(x)


def _per_compartment_paired_metrics(
    pred_lm: np.ndarray,
    ref_lm: np.ndarray,
    *,
    voxel_spacing_mm: tuple[float, float, float],
    voxel_volume_mm3: float,
) -> dict[str, dict[str, float | None | str]]:
    """Dice / HD95 / |VE| / sensitivity / specificity per WT/TC/ET.

    Identical signature and semantics to
    :func:`scripts.segment_all_cohort50._per_compartment_full_metrics`,
    used here with ``pred = DL`` and ``ref = baseline`` so the resulting
    JSON can be aggregated alongside the ``metrics_dl_vs_gt.json`` files
    by a single Table 3 reader.

    NaN values are returned as ``None`` so the JSON sidecar is valid
    (NaN is not legal JSON). Inf is preserved as ``"inf"`` (string)
    for the same reason.
    """
    pred = derive_subregions(pred_lm)
    ref = derive_subregions(ref_lm)
    out: dict[str, dict[str, float | None | str]] = {}
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


# ---------------------------------------------------------------------------
# Per-(subject, baseline) processing.
# ---------------------------------------------------------------------------


def _real_process_subject_baseline(
    subject_id: str,
    baseline: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    backend: str,
    skip_existing: bool,
    require_dl: bool,
    image_digest: str | None = None,
    tumorsynth_command: str = "mri_TumorSynth",
    tumorsynth_nnunet_dir: Path | None = None,
    tumorsynth_threads: int = 8,
    tumorsynth_cpu: bool = False,
    tumorsynth_work_dir: Path | None = None,
    tumorsynth_whole_tumor_label: int = TUMORSYNTH_WHOLE_TUMOR_LABEL,
    tumorsynth_inner_ncr_labels: tuple[int, ...] = TUMORSYNTH_INNER_NCR_LABELS,
    tumorsynth_inner_ed_labels: tuple[int, ...] = TUMORSYNTH_INNER_ED_LABELS,
    tumorsynth_inner_et_labels: tuple[int, ...] = TUMORSYNTH_INNER_ET_LABELS,
) -> dict[str, Any]:
    """Run one baseline on one subject and write the section-4 artefacts.

    Writes ``seg_<baseline>/sub-XXXX_seg_<baseline>.{nii.gz,.json}`` and,
    when the PR-7d DL output for the same subject exists,
    ``metrics/sub-XXXX_metrics_dl_vs_<baseline>.json``. Returns the
    in-memory record so the caller can build a cohort-level summary
    table on the fly.
    """
    sid = normalize_subject_id(subject_id)
    artefacts = _planned_baseline_artefacts(deriv_root, sid, baseline)
    seg_path: Path = artefacts["seg_label_map"]
    sidecar_path: Path = artefacts["seg_sidecar"]
    metrics_path: Path = artefacts["metr_dl_vs_baseline"]
    dl_lm_path: Path = artefacts["seg_dl_label_map"]

    existing_required = [seg_path, sidecar_path]
    if baseline == "tumorsynth":
        existing_required.extend(
            [
                artefacts["seg_tumorsynth_raw_whole"],
                artefacts["seg_tumorsynth_raw_inner"],
            ],
        )
    if skip_existing and all(path.exists() for path in existing_required):
        return {
            "subject_id": sid,
            "baseline": baseline,
            "skipped": True,
            "reason": f"--skip-existing and {seg_path.name} present",
        }

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
    result = predict_baseline(
        baseline=baseline,  # type: ignore[arg-type]
        t1=channels["T1"],
        t1c=channels["T1c"],
        t2=channels["T2"],
        flair=channels["FLAIR"],
        backend=backend,  # type: ignore[arg-type]
        image_digest=image_digest,
        tumorsynth_command=tumorsynth_command,
        tumorsynth_nnunet_dir=tumorsynth_nnunet_dir,
        tumorsynth_threads=tumorsynth_threads,
        tumorsynth_cpu=tumorsynth_cpu,
        tumorsynth_work_dir=tumorsynth_work_dir,
        tumorsynth_whole_tumor_label=tumorsynth_whole_tumor_label,
        tumorsynth_inner_ncr_labels=tumorsynth_inner_ncr_labels,
        tumorsynth_inner_ed_labels=tumorsynth_inner_ed_labels,
        tumorsynth_inner_et_labels=tumorsynth_inner_et_labels,
        tumorsynth_preserve_raw_dir=(seg_path.parent if baseline == "tumorsynth" else None),
        tumorsynth_preserve_raw_stem=f"sub-{sid}_tumorsynth",
    )
    elapsed_s = time.time() - t0

    voxel_vol = _voxel_volume_mm3(result.affine)
    voxel_spacing = _voxel_spacing_mm(result.affine)

    seg_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    derivatives_dir(deriv_root, sid)  # ensures the per-subject root exists too

    nib.save(
        nib.Nifti1Image(result.label_map.astype(np.uint8), result.affine),
        str(seg_path),
    )

    paired_metrics: dict[str, Any] | None = None
    paired_status: str
    if dl_lm_path.is_file():
        dl_lm, _ = load_nifti(dl_lm_path)
        dl_lm = np.asarray(dl_lm)
        if dl_lm.shape != result.label_map.shape:
            paired_status = "shape_mismatch"
            print(
                f"[run] sub-{sid}: WARNING DL/{baseline} shape mismatch "
                f"({dl_lm.shape} vs {result.label_map.shape}); paired metrics skipped",
                file=sys.stderr,
            )
        else:
            paired_metrics = _per_compartment_paired_metrics(
                dl_lm,
                result.label_map,
                voxel_spacing_mm=voxel_spacing,
                voxel_volume_mm3=voxel_vol,
            )
            metrics_doc = {
                "schema_version": METRICS_SIDECAR_SCHEMA_VERSION,
                "subject_id": sid,
                "comparison": f"dl_vs_{baseline}",
                "voxel_volume_mm3": round(voxel_vol, 6),
                "voxel_spacing_mm": [round(s, 6) for s in voxel_spacing],
                "metrics": paired_metrics,
            }
            metrics_path.write_text(json.dumps(metrics_doc, indent=2))
            paired_status = "ok"
    else:
        paired_status = "dl_missing"
        if require_dl:
            raise FileNotFoundError(
                f"sub-{sid}: PR-7d DL output {dl_lm_path} missing; "
                "rerun 'make segment-all' first or pass --no-require-dl",
            )
        print(
            f"[run] sub-{sid}: WARNING DL output missing; "
            "writing baseline mask only, no paired metrics",
            file=sys.stderr,
        )

    sidecar = {
        "schema_version": SEG_SIDECAR_SCHEMA_VERSION,
        "subject_id": sid,
        "baseline": result.baseline,
        "backend": result.backend,
        "version": result.version,
        "elapsed_s": round(float(elapsed_s), 3),
        "voxel_volume_mm3": round(voxel_vol, 6),
        "voxel_spacing_mm": [round(s, 6) for s in voxel_spacing],
        "paired_metrics_status": paired_status,
        "tumorsynth": (
            {
                "command": tumorsynth_command,
                "nnunet_dir": str(tumorsynth_nnunet_dir) if tumorsynth_nnunet_dir else None,
                "threads": tumorsynth_threads,
                "cpu": tumorsynth_cpu,
                "whole_tumor_label": tumorsynth_whole_tumor_label,
                "inner_ncr_labels": list(tumorsynth_inner_ncr_labels),
                "inner_ed_labels": list(tumorsynth_inner_ed_labels),
                "inner_et_labels": list(tumorsynth_inner_et_labels),
                "note": (
                    "Only meaningful when baseline == 'tumorsynth'; TumorSynth is run "
                    "as whole-tumor segmentation followed by inner-tumor ROI segmentation."
                ),
            }
            if result.baseline == "tumorsynth"
            else None
        ),
        "outputs": {
            "label_map": str(seg_path),
            "metrics_dl_vs_baseline": str(metrics_path) if paired_metrics else None,
            **(
                {
                    "tumorsynth_whole_tumor_raw": str(
                        result.raw_outputs.get(
                            "tumorsynth_whole_tumor_raw",
                            artefacts["seg_tumorsynth_raw_whole"],
                        ),
                    ),
                    "tumorsynth_inner_tumor_raw": str(
                        result.raw_outputs.get(
                            "tumorsynth_inner_tumor_raw",
                            artefacts["seg_tumorsynth_raw_inner"],
                        ),
                    ),
                }
                if result.baseline == "tumorsynth"
                else {}
            ),
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return {
        "subject_id": sid,
        "baseline": result.baseline,
        "backend": result.backend,
        "version": result.version,
        "elapsed_s": round(float(elapsed_s), 3),
        "voxel_volume_mm3": voxel_vol,
        "metrics": paired_metrics,
        "paired_metrics_status": paired_status,
        "outputs": sidecar["outputs"],
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Selection helpers.
# ---------------------------------------------------------------------------


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


def _select_baselines(requested: list[str] | None) -> list[str]:
    """Resolve ``--baseline`` flags into a concrete, deduplicated list.

    ``None`` (default) -> all baselines, in :data:`BASELINES` order.
    Any occurrence of ``"all"`` expands to all baselines.
    """
    if not requested:
        return list(BASELINES)
    out: list[str] = []
    for tag in requested:
        if tag == "all":
            for b in BASELINES:
                if b not in out:
                    out.append(b)
        else:
            if tag not in BASELINES:
                raise SystemExit(
                    f"error: --baseline {tag!r} not in {list(BASELINES)}",
                )
            if tag not in out:
                out.append(tag)
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _print_subject_block_dry(
    subject_id: str,
    metadata_row: dict,
    baselines: list[str],
    deriv_root: Path,
    inputs_ok: bool,
    inputs_detail: str,
    dl_present: bool,
    *,
    verbose: bool,
) -> None:
    sex = metadata_row.get("Sex", "?")
    age = metadata_row.get("Age at MRI", "?")
    os_days = metadata_row.get("OS", "?")
    os_tert = metadata_row.get("OS_tertile", "?")
    flag = "OK " if inputs_ok else "MISS"
    dl_flag = "DL=on" if dl_present else "DL=off"
    print(
        f"[{flag}] sub-{subject_id}  {dl_flag}  sex={sex}  age={age}  OS={os_days}d ({os_tert})",
    )
    if not inputs_ok:
        print(f"        missing-input: {inputs_detail}")
    if verbose:
        for baseline in baselines:
            artefacts = _planned_baseline_artefacts(deriv_root, subject_id, baseline)
            for key, path in artefacts.items():
                if key == "seg_dl_label_map":
                    continue
                print(f"        WOULD WRITE  [{baseline:<14} / {key:<22}]  {path}")


def _print_dry_summary(
    n_subjects: int,
    n_baselines: int,
    n_inputs_ok: int,
    n_dl_present: int,
) -> None:
    print()
    print("=" * 78)
    print("PR-7e dry-run summary (baseline cohort fan-out)")
    print("=" * 78)
    print(f"  cohort size           : {n_subjects}")
    print(f"  baselines per subject : {n_baselines}")
    print(f"  inputs present        : {n_inputs_ok} / {n_subjects}")
    print(f"  DL output present     : {n_dl_present} / {n_subjects}")
    print(
        "  artefact contract     : Section 4 of docs/scope_pr7.md "
        "(extended with seg_<baseline>/)\n"
        "  next step             : run the real fan-out with --no-dry-run "
        "(or 'make baselines-all')"
    )
    print("=" * 78)


def _fmt_metric(value: object, *, width: int = 7, precision: int = 3) -> str:
    if value is None:
        return f"{'nan':>{width}}"
    if isinstance(value, str):
        return f"{value:>{width}}"
    return f"{float(value):>{width}.{precision}f}"


def _print_real_summary(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 96)
    print("PR-7e baseline-segmentation report (DL vs baseline; paired)")
    print("=" * 96)
    print(
        f"{'subject':<10}{'baseline':<16}{'time(s)':>9}"
        f"{'WT-Dice':>9}{'TC-Dice':>9}{'ET-Dice':>9}"
        f"{'WT-HD95':>9}{'TC-HD95':>9}{'ET-HD95':>9}"
        f"{'  status':>16}",
    )
    print("-" * 96)
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    n_no_paired = 0
    for r in records:
        if r.get("error"):
            n_failed += 1
            print(
                f"{r['subject_id']:<10}{r.get('baseline', '?'):<16}"
                f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}"
                f"{'-':>9}{'-':>9}{'-':>9}{'  FAILED':>16}"
            )
            print(f"        {r['error']}")
            continue
        if r.get("skipped"):
            n_skipped += 1
            print(
                f"{r['subject_id']:<10}{r['baseline']:<16}"
                f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}"
                f"{'-':>9}{'-':>9}{'-':>9}{'  SKIPPED':>16}"
            )
            continue
        n_ok += 1
        m = r.get("metrics") or {
            "WT": {"dice": None, "hd95_mm": None},
            "TC": {"dice": None, "hd95_mm": None},
            "ET": {"dice": None, "hd95_mm": None},
        }
        if r.get("metrics") is None:
            n_no_paired += 1
            status = "OK (no paired)"
        else:
            status = "OK"
        print(
            f"{r['subject_id']:<10}{r['baseline']:<16}"
            f"{r['elapsed_s']:>9.1f}"
            f"{_fmt_metric(m['WT']['dice'])}{_fmt_metric(m['TC']['dice'])}"
            f"{_fmt_metric(m['ET']['dice'])}"
            f"{_fmt_metric(m['WT']['hd95_mm'], precision=2)}"
            f"{_fmt_metric(m['TC']['hd95_mm'], precision=2)}"
            f"{_fmt_metric(m['ET']['hd95_mm'], precision=2)}"
            f"  {status:>14}",
        )
    print("=" * 96)
    print(
        f"  ok={n_ok}  failed={n_failed}  skipped={n_skipped}  "
        f"no-paired-metrics={n_no_paired}\n"
        "  Paired metrics use convention: prediction = DL, reference = baseline.\n"
        "  Dice / HD95 / |VE| are symmetric; sens/spec read as DL relative to baseline."
    )
    print("=" * 96)


# ---------------------------------------------------------------------------
# Top-level dry / real flows.
# ---------------------------------------------------------------------------


def _run_dry(args: argparse.Namespace, cm) -> int:
    baselines = _select_baselines(args.baselines)
    subjects = _select_subjects(list(cm.df.index), args.subjects)
    print(
        f"PR-7e dry-run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"             cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"             baselines={baselines}  n_requested={len(subjects)}  "
        f"cohort_n={cm.n}  seed={cm.seed}",
    )
    print("-" * 78)
    n_inputs_ok = 0
    n_dl_present = 0
    n_missing = 0
    for subject_id in subjects:
        inputs_ok, inputs_detail = _check_inputs(args.cohort_root, subject_id)
        if inputs_ok:
            n_inputs_ok += 1
        else:
            n_missing += 1
        # The DL output is the same for every baseline; probe once.
        dl_path = _planned_baseline_artefacts(
            args.deriv_root,
            subject_id,
            baselines[0],
        )["seg_dl_label_map"]
        dl_present = dl_path.is_file()
        if dl_present:
            n_dl_present += 1
        _print_subject_block_dry(
            subject_id,
            cm.df.loc[subject_id].to_dict(),
            baselines,
            args.deriv_root,
            inputs_ok,
            inputs_detail,
            dl_present,
            verbose=args.show_paths,
        )
    _print_dry_summary(len(subjects), len(baselines), n_inputs_ok, n_dl_present)
    return 0 if n_missing == 0 else 1


def _run_real(args: argparse.Namespace, cm) -> int:
    subjects = _select_subjects(list(cm.df.index), args.subjects)
    baselines = _select_baselines(args.baselines)
    tumorsynth_inner_ncr_labels = _parse_int_csv(
        args.tumorsynth_inner_ncr_labels,
        option_name="--tumorsynth-inner-ncr-labels",
    )
    tumorsynth_inner_ed_labels = _parse_int_csv(
        args.tumorsynth_inner_ed_labels,
        option_name="--tumorsynth-inner-ed-labels",
    )
    tumorsynth_inner_et_labels = _parse_int_csv(
        args.tumorsynth_inner_et_labels,
        option_name="--tumorsynth-inner-et-labels",
    )
    print(
        f"PR-7e real run  cohort_yaml={args.cohort_yaml}  "
        f"metadata_csv={args.metadata_csv}\n"
        f"               cohort_root={args.cohort_root}  "
        f"deriv_root={args.deriv_root}\n"
        f"               baselines={baselines}  backend={args.backend}  "
        f"skip_existing={args.skip_existing}  require_dl={args.require_dl}\n"
        f"               n_requested={len(subjects)}  cohort_n={cm.n}  "
        f"seed={cm.seed}",
    )
    print("-" * 96)

    records: list[dict[str, Any]] = []
    for sid in subjects:
        for baseline in baselines:
            print(f"[run] sub-{sid}  baseline={baseline} ...", flush=True)
            try:
                rec = _real_process_subject_baseline(
                    sid,
                    baseline,
                    cohort_root=args.cohort_root,
                    deriv_root=args.deriv_root,
                    backend=args.backend,
                    skip_existing=args.skip_existing,
                    require_dl=args.require_dl,
                    tumorsynth_command=args.tumorsynth_command,
                    tumorsynth_nnunet_dir=args.tumorsynth_nnunet_dir,
                    tumorsynth_threads=args.tumorsynth_threads,
                    tumorsynth_cpu=args.tumorsynth_cpu,
                    tumorsynth_work_dir=args.tumorsynth_work_dir,
                    tumorsynth_whole_tumor_label=args.tumorsynth_whole_tumor_label,
                    tumorsynth_inner_ncr_labels=tumorsynth_inner_ncr_labels,
                    tumorsynth_inner_ed_labels=tumorsynth_inner_ed_labels,
                    tumorsynth_inner_et_labels=tumorsynth_inner_et_labels,
                )
            except (
                NotImplementedError,
                RuntimeError,
                FileNotFoundError,
                ValueError,
                KeyError,
            ) as exc:
                print(
                    f"[run] sub-{sid}  baseline={baseline}: FAILED: {exc}",
                    file=sys.stderr,
                )
                records.append(
                    {
                        "subject_id": sid,
                        "baseline": baseline,
                        "error": str(exc),
                    },
                )
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
