"""End-to-end smoke test of the unified BraTS three-class segmenter (PR-8).

Runs :func:`hpgs.segment.unified.predict_brats3` on a small subset of the
n=50 UCSF-PDGM cohort, writes the resulting label map and per-class
sigmoid probabilities to ``data/derivatives_cohort50/sub-XXXX/`` in NIfTI
form, and prints a per-subject sanity report (per-compartment Dice and
volumetric error against the dataset-provided reference mask).

This is the *smoke* version of the full PR-8 / PR-9 pipeline:
    * Three subjects only (default), one per OS-tertile if possible.
    * No multi-channel pre-processing beyond brain-masked z-score.
    * No statistical comparison against Raidionics or segment_glioma yet.
    * No paper artefacts written.

The goal is to learn, *before* committing to a 50-subject run, whether:
    1. The MONAI bundle download + load succeeds in this environment.
    2. SegResNet inference fits in the available memory at ``roi=240^3``.
    3. The bundle's input expectations (RAS, 1mm, brain-masked z-score)
       agree with the UCSF-PDGM extracted layout (they should).
    4. The DL output broadly agrees with the dataset-provided reference
       (Dice in the 0.7--0.9 range per compartment is a healthy sign).

Usage::

    # Real run on three default subjects with the MONAI bundle:
    uv run python scripts/smoke_predict_brats3.py

    # Pick subjects explicitly:
    uv run python scripts/smoke_predict_brats3.py --subject 0005 --subject 0026

    # Adapter-only dry run (no weights, no torch GPU; useful in CI):
    uv run python scripts/smoke_predict_brats3.py --backend dummy --no-write

agreement tables and uncertainty bands).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from hpgs.io import (
    derivatives_dir,
    load_nifti,
    resolve_subject_inputs,
)
from hpgs.metrics import absolute_volumetric_error, dice
from hpgs.segment import derive_subregions
from hpgs.segment.unified import predict_brats3

DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")
DEFAULT_SUBJECTS = ("0005", "0035", "0068")  # one near each OS tertile in cohort YAML
DEFAULT_CHANNELS = ["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        metavar="ID",
        help=f"4-digit subject id; repeat for multiple. Defaults to {DEFAULT_SUBJECTS}.",
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
        "--no-write",
        action="store_true",
        help="Skip writing NIfTI outputs (useful for dry-runs in CI).",
    )
    return p.parse_args()


def _per_compartment_metrics(
    pred_lm: np.ndarray,
    ref_lm: np.ndarray,
    voxel_volume_mm3: float,
) -> dict[str, dict[str, float]]:
    """Per-compartment Dice + |VE| between the unified DL output and a reference."""
    pred = derive_subregions(pred_lm)
    ref = derive_subregions(ref_lm)
    out: dict[str, dict[str, float]] = {}
    for name, p_mask, r_mask in (
        ("WT", pred.wt, ref.wt),
        ("TC", pred.tc, ref.tc),
        ("ET", pred.et, ref.et),
    ):
        out[name] = {
            "dice": dice(p_mask, r_mask),
            "abs_volumetric_error_mm3": absolute_volumetric_error(p_mask, r_mask, voxel_volume_mm3),
        }
    return out


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume from the affine; for UCSF-PDGM this is 1mm^3."""
    return float(abs(np.linalg.det(affine[:3, :3])))


def _process_subject(
    subject_id: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    backend: str,
    device: str,
    bundle_dir: Path | None,
    write_outputs: bool,
) -> dict:
    inputs = resolve_subject_inputs(
        cohort_root,
        subject_id,
        channel_names=DEFAULT_CHANNELS,
        tumor_segmentation_name="tumor_segmentation",
    )
    # Map UCSF-PDGM channel names to the predictor's canonical names.
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
            f"sub-{subject_id}: reference mask shape {ref_lm.shape} != "
            f"predicted shape {result.label_map.shape}",
        )
    voxel_vol = _voxel_volume_mm3(ref_affine)
    metrics = _per_compartment_metrics(result.label_map, np.asarray(ref_lm), voxel_vol)

    out_dir = derivatives_dir(deriv_root, subject_id)
    sidecar: dict = {
        "subject_id": subject_id,
        "backend": result.backend,
        "bundle_version": result.bundle_version,
        "device": device,
        "elapsed_s": round(elapsed_s, 2),
        "voxel_volume_mm3": round(voxel_vol, 6),
        "metrics_vs_reference": {
            comp: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in m.items()}
            for comp, m in metrics.items()
        },
    }
    if write_outputs:
        seg_path = out_dir / f"sub-{subject_id}_seg_brats3_dl.nii.gz"
        nib.save(
            nib.Nifti1Image(result.label_map.astype(np.uint8), result.affine),
            str(seg_path),
        )
        prob_path = out_dir / f"sub-{subject_id}_seg_brats3_dl_probs.nii.gz"
        # Move channel axis to the last position for NIfTI's (X, Y, Z, T) convention.
        probs_xyzt = np.moveaxis(result.probabilities, 0, -1).astype(np.float32)
        nib.save(nib.Nifti1Image(probs_xyzt, result.affine), str(prob_path))
        sidecar_path = out_dir / f"sub-{subject_id}_seg_brats3_dl.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        sidecar["outputs"] = {
            "label_map": str(seg_path),
            "probabilities": str(prob_path),
            "sidecar": str(sidecar_path),
        }

    return sidecar


def _print_report(records: list[dict]) -> None:
    print()
    print("=" * 78)
    print("PR-8 smoke report")
    print("=" * 78)
    print(
        f"{'subject':<10}{'backend':<14}{'time(s)':>8}"
        f"{'WT-Dice':>10}{'TC-Dice':>10}{'ET-Dice':>10}"
        f"{'WT |VE|':>12}{'TC |VE|':>12}{'ET |VE|':>12}",
    )
    print("-" * 78)
    for r in records:
        m = r["metrics_vs_reference"]

        def _fmt_d(c: str, _m: dict = m) -> str:
            v = _m[c]["dice"]
            return "nan" if v is None else f"{v:.3f}"

        def _fmt_v(c: str, _m: dict = m) -> str:
            v = _m[c]["abs_volumetric_error_mm3"]
            return "nan" if v is None else f"{v:>10.0f}"

        print(
            f"{r['subject_id']:<10}{r['backend']:<14}{r['elapsed_s']:>8.1f}"
            f"{_fmt_d('WT'):>10}{_fmt_d('TC'):>10}{_fmt_d('ET'):>10}"
            f"{_fmt_v('WT'):>12}{_fmt_v('TC'):>12}{_fmt_v('ET'):>12}",
        )
    print("=" * 78)
    print(
        "Healthy ranges (heuristic): WT 0.85-0.95, TC 0.75-0.90, ET 0.70-0.90 Dice.\n"
        "Substantially lower numbers across all subjects suggest a preprocessing\n"
        "mismatch (channel order, intensity range, or orientation).",
    )


def main() -> int:
    args = _parse_args()
    subjects: tuple[str, ...] = tuple(args.subjects) if args.subjects else DEFAULT_SUBJECTS

    if not args.cohort_root.is_dir():
        print(f"error: cohort root does not exist: {args.cohort_root}", file=sys.stderr)
        return 2

    records: list[dict] = []
    for sub in subjects:
        print(f"[smoke] sub-{sub}: backend={args.backend} device={args.device} ...", flush=True)
        try:
            record = _process_subject(
                sub,
                cohort_root=args.cohort_root,
                deriv_root=args.deriv_root,
                backend=args.backend,
                device=args.device,
                bundle_dir=args.bundle_dir,
                write_outputs=not args.no_write,
            )
        except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
            # Surface the failure but keep going on the remaining subjects so the
            # report still shows the partial picture.
            print(f"[smoke] sub-{sub}: FAILED: {exc}", file=sys.stderr)
            records.append(
                {
                    "subject_id": sub,
                    "backend": args.backend,
                    "bundle_version": None,
                    "elapsed_s": 0.0,
                    "metrics_vs_reference": {
                        "WT": {"dice": None, "abs_volumetric_error_mm3": None},
                        "TC": {"dice": None, "abs_volumetric_error_mm3": None},
                        "ET": {"dice": None, "abs_volumetric_error_mm3": None},
                    },
                    "error": str(exc),
                },
            )
            continue
        records.append(record)

    _print_report(records)
    return 0 if all("error" not in r for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
