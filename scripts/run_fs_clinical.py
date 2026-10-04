"""Run FreeSurfer 8.2.0 ``recon-all-clinical`` + standalone ``mri_synthsr`` for one subject.

The script is the single-subject entry point for regenerating the FS-derived
panels of manuscript Figure 2. For a UCSF-PDGM subject it:

1. Runs ``mri_synthsr`` standalone on the raw T1 to produce the panel-c image
   (``synthSR_standalone.nii.gz``).
2. Runs ``recon-all-clinical.sh`` on the raw T1 to produce
   ``synthSR.mgz`` / ``aparc+aseg.mgz`` / ``wmparc.mgz`` / ``synthseg.mgz`` in
   the FS conformed space inside ``<subjects_dir>/<subject>/mri``.
3. Resamples the FS outputs back onto the native mpMRI grid using
   ``mri_convert -rl`` so they align voxel-for-voxel with the bias-corrected
   channels and the tumor segmentation.
4. Places the resampled NIfTI derivatives into
   ``<derivatives_root>/sub-<id>/{preprocess,parcellation}/``.

Usage::

    uv run python scripts/run_fs_clinical.py --subject 0020 \
        --config configs/legacy5.yaml --threads 6
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from hpgs.io import normalize_subject_id
from hpgs.parcellate import recon_all_clinical, resample_like, synthseg
from hpgs.preprocess import synth_sr


def _load_config(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def run(
    *,
    subject_id: str,
    data_root: Path,
    derivatives_root: Path,
    subjects_dir: Path,
    threads: int,
    overwrite: bool,
) -> dict[str, Path]:
    """Run the full FS clinical pipeline for one subject."""
    subject_id = normalize_subject_id(subject_id)
    subject_dir = data_root / f"sub-{subject_id}"
    if not subject_dir.is_dir():
        raise FileNotFoundError(subject_dir)

    t1_raw = subject_dir / f"sub-{subject_id}_T1.nii.gz"
    t1_bias = subject_dir / f"sub-{subject_id}_T1_bias.nii.gz"
    reference_grid = t1_bias if t1_bias.is_file() else t1_raw
    for p in (t1_raw, reference_grid):
        if not p.is_file():
            raise FileNotFoundError(p)

    deriv = derivatives_root / f"sub-{subject_id}"
    preprocess_dir = deriv / "preprocess"
    parcellation_dir = deriv / "parcellation"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    parcellation_dir.mkdir(parents=True, exist_ok=True)

    standalone_synthsr = preprocess_dir / "synthSR_standalone.nii.gz"
    if overwrite or not standalone_synthsr.is_file():
        synth_sr(t1_raw, standalone_synthsr, threads=threads, cpu=True)

    standalone_synthseg = parcellation_dir / "synthseg.nii.gz"
    if overwrite or not standalone_synthseg.is_file():
        synthseg(reference_grid, standalone_synthseg, robust=True, parc=True, fast=False)

    fs_subject_id = f"sub-{subject_id}"
    subject_mri = recon_all_clinical(
        t1_raw,
        fs_subject_id,
        subjects_dir=subjects_dir,
        threads=threads,
        overwrite=overwrite,
    )

    resampled = {
        "synthSR_recon": (
            subject_mri / "synthSR.mgz",
            preprocess_dir / "synthSR_recon.nii.gz",
            "cubic",
        ),
        "aparc+aseg": (
            subject_mri / "aparc+aseg.mgz",
            parcellation_dir / "aparc+aseg.nii.gz",
            "nearest",
        ),
        "wmparc": (
            subject_mri / "wmparc.mgz",
            parcellation_dir / "wmparc.nii.gz",
            "nearest",
        ),
        "synthseg_clinical": (
            subject_mri / "synthseg.mgz",
            parcellation_dir / "synthseg_clinical.nii.gz",
            "nearest",
        ),
    }

    outputs: dict[str, Path] = {
        "synthSR_standalone": standalone_synthsr,
        "synthseg": standalone_synthseg,
    }
    for name, (src, dst, interp) in resampled.items():
        if not src.is_file():
            continue
        if overwrite or not dst.is_file():
            resample_like(src, reference_grid, dst, interpolation=interp)
        outputs[name] = dst

    manifest = deriv / "fs_clinical_manifest.txt"
    manifest.write_text("\n".join(f"{k}\t{v}" for k, v in sorted(outputs.items())) + "\n")
    return outputs


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = cfg.get("paths", {})
    if not all(k in paths for k in ("data_root", "derivatives", "freesurfer_subjects")):
        raise KeyError(
            "Config must define paths.data_root, paths.derivatives, paths.freesurfer_subjects"
        )

    if not shutil.which("recon-all-clinical.sh"):
        fs_home = os.environ.get("FREESURFER_HOME", "?")
        print(
            f"[warning] recon-all-clinical.sh not on PATH (FREESURFER_HOME={fs_home}); "
            "the wrapper will still try FREESURFER_HOME/bin."
        )

    outputs = run(
        subject_id=args.subject,
        data_root=Path(paths["data_root"]).resolve(),
        derivatives_root=Path(paths["derivatives"]).resolve(),
        subjects_dir=Path(paths["freesurfer_subjects"]).resolve(),
        threads=args.threads,
        overwrite=args.overwrite,
    )
    for name, path in sorted(outputs.items()):
        print(f"{name}\t{path}")


if __name__ == "__main__":
    _cli()
