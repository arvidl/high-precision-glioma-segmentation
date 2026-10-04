"""Baseline glioma-segmentation backends.

This module provides a thin, unified API around two external open-source
brain-tumor segmenters that the revision benchmarks the unified DL
backbone against in Table 3 of ``paper/main.tex`` (R2.5 + the
Q1=B four-column framing locked in ``docs/scope_pr7.md`` Section 3):

* **Raidionics** -- atlas-space pipeline run via Docker.
* **segment_glioma** -- pretrained PICTURE-project model run via Docker.
* **mri_TumorSynth** -- FreeSurfer-ecosystem research tool run via its
  command-line wrapper.

Both real backends are pinned to upstream Docker image digests in
``configs/baselines.yaml`` (PR-7e2). The generic ``"docker"`` backend is
intentionally a stub in this commit and raises ``NotImplementedError``.
TumorSynth is wired through the ``"command"`` backend because upstream
ships a local shell wrapper (``mri_TumorSynth`` / ``mri_tumorsynth``)
that expects an installed nnU-Net v1.7 model tree. The CI-runnable
``"dummy"`` backend exercises the *adapter contract* (label-map dtype,
BraTS label set, affine propagation, sidecar schema) without docker /
network / GPU.

Public API::

    from hpgs.segment.baselines import predict_baseline, BaselineResult

    result = predict_baseline(
        baseline="raidionics",
        t1=..., t1c=..., t2=..., flair=...,
        backend="dummy",
    )
    result.label_map      # (D, H, W) uint8, BraTS labels {0, 1, 2, 4}
    result.affine         # (4, 4) float, copied from FLAIR
    result.baseline       # "raidionics" | "segmentglioma" | "tumorsynth"
    result.backend        # "docker" | "command" | "dummy"
    result.version        # provenance (docker digest or "dummy")

The output label scheme is deliberately the same as
:func:`hpgs.segment.unified.predict_brats3` so that downstream consumers
(the Table 3 producer in PR-7h, the per-cell agreement panel in PR-7i)
can call :func:`hpgs.segment.derive_subregions` on either segmenter's
output without needing per-baseline label-set translation.

Raidionics and segment_glioma on the same 50 subjects), R2.10
(structured quantitative summaries -- four-column DL / Raidionics /
segment_glioma / BraTS21-manual agreement table).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from hpgs.io import load_nifti
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR

Baseline = Literal["raidionics", "segmentglioma", "tumorsynth"]
Backend = Literal["docker", "command", "dummy"]

# All baseline names accepted by :func:`predict_baseline`. Used by the
# cohort runner in ``scripts/run_baseline_segmenters_cohort50.py`` to
# expand ``--baseline all``.
BASELINES: tuple[Baseline, ...] = ("raidionics", "segmentglioma", "tumorsynth")

# Channel signature accepted by :func:`predict_baseline`. There is no
# *single* canonical channel order across baselines (Raidionics and
# segment_glioma both have their own internal preprocessing and accept
# the four mpMRI channels by filename, not by stack position), so this
# is just the kwarg signature, not an order constraint.
CHANNEL_NAMES: tuple[str, ...] = ("T1", "T1c", "T2", "FLAIR")
TUMORSYNTH_CHANNEL_ORDER: tuple[str, ...] = ("T1c", "T1", "T2", "FLAIR")
TUMORSYNTH_WHOLE_TUMOR_LABEL = 18
TUMORSYNTH_INNER_NCR_LABELS: tuple[int, ...] = (3,)
TUMORSYNTH_INNER_ED_LABELS: tuple[int, ...] = (1,)
TUMORSYNTH_INNER_ET_LABELS: tuple[int, ...] = (2,)


@dataclass(frozen=True)
class BaselineResult:
    """One baseline segmenter's output on one subject.

    Attributes
    ----------
    label_map
        ``(D, H, W)`` ``uint8`` array of BraTS multi-class labels:
        ``0=BG``, ``1=NCR``, ``2=ED``, ``4=ET``. Both real backends emit
        masks that are remapped (or thresholded) into this scheme inside
        :func:`predict_baseline` so downstream consumers see a single
        canonical label set.
    affine
        ``(4, 4)`` voxel-to-world affine, propagated from the FLAIR
        header so the result can be saved as a NIfTI without re-loading.
    baseline
        ``"raidionics"``, ``"segmentglioma"``, or ``"tumorsynth"``.
        Stored so the on-disk sidecar can be tagged with the engine that
        produced it.
    backend
        ``"docker"`` (real, PR-7e2), ``"command"`` (real TumorSynth
        command-line wrapper), or ``"dummy"`` (CI / smoke).
    version
        Provenance string. For the real backends this is the upstream
        docker image digest pinned in ``configs/baselines.yaml`` (the
        digest is propagated verbatim onto the sidecar). For the dummy
        backend this is ``"dummy"``.
    channel_paths
        Optional record of the input file paths (handy when the caller
        wants to write a sidecar JSON next to the segmentation).
    raw_outputs
        Optional raw backend artefacts preserved before project-specific
        remapping. TumorSynth uses this for the whole-tumor and
        inner-tumor label maps needed to audit label semantics.
    """

    label_map: np.ndarray
    affine: np.ndarray
    baseline: str
    backend: str
    version: str
    channel_paths: dict[str, Path] = field(default_factory=dict)
    raw_outputs: dict[str, Path] = field(default_factory=dict)


def _load_flair_affine(flair: str | Path) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return ``(shape, affine)`` from the FLAIR header.

    Both real backends produce an output mask in the same grid as the
    input FLAIR; we propagate the FLAIR affine onto the result so the
    saved NIfTI lives in subject-native space exactly like the unified
    DL output. The dummy backend uses the same convention so its
    artefact contract is identical.
    """
    data, affine = load_nifti(flair)
    return tuple(int(s) for s in data.shape[:3]), affine.astype(np.float64)


def _dummy_backend(
    *,
    shape: tuple[int, int, int],
    baseline: str,
) -> np.ndarray:
    """Deterministic, biologically implausible BraTS label-map.

    Mirrors the dummy-backend strategy used by
    :func:`hpgs.segment.unified.predict_brats3` so that the PR-7e
    cohort runner and its test suite can exercise the artefact contract
    (label dtype, BraTS label set, affine propagation, sidecar schema)
    without docker / network / GPU. Each baseline gets a slightly
    different synthetic lesion so that the per-subject metric files
    ``metrics_dl_vs_raidionics.json`` and
    ``metrics_dl_vs_segmentglioma.json`` have non-trivial *distinct*
    values per baseline -- otherwise both files would be identical and
    the test would not catch a confused output mapping at the
    runner-loop level.
    """
    depth, height, width = shape
    cz, cy, cx = depth // 2, height // 2, width // 2
    rz, ry, rx = max(2, depth // 8), max(2, height // 8), max(2, width // 8)
    # Per-baseline jitter: distinct in *both* centroid and radius so the
    # resulting paired metrics (DL vs raidionics) and (DL vs
    # segmentglioma) are unambiguously different against any reasonable
    # DL reference, even on small synthetic test volumes where a single-
    # voxel translational shift could yield identical Dice/HD95 by
    # symmetry.
    if baseline == "raidionics":
        cx += max(1, width // 16)
        rx = max(2, rx + 1)
    elif baseline == "segmentglioma":
        cy += max(1, height // 16)
        ry = max(2, ry - 1)
    elif baseline == "tumorsynth":
        cz += max(1, depth // 16)
        rz = max(2, rz + 2)
    z = np.arange(depth)[:, None, None]
    y = np.arange(height)[None, :, None]
    x = np.arange(width)[None, None, :]
    r = np.sqrt(((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    label = np.zeros(shape, dtype=np.uint8)
    # Order matters: outer shell first (ED), then core (NCR), then
    # enhancing (ET) writes last and wins all ties. Same convention as
    # :func:`hpgs.segment.unified.brats3_to_label_map`.
    label[r < 1.5] = LABEL_ED
    label[r < 1.2] = LABEL_NCR
    label[r < 0.9] = LABEL_ET
    return label


def _resolve_command(command: str | Path) -> str:
    """Return an executable path for a local command-line backend."""
    command_s = str(command)
    if os.sep in command_s:
        path = Path(command_s)
        if not path.is_file():
            raise FileNotFoundError(f"TumorSynth command not found: {path}")
        return str(path)
    resolved = shutil.which(command_s)
    if resolved is None:
        raise FileNotFoundError(
            f"TumorSynth command {command_s!r} not found on PATH. "
            "Install TumorSynth or pass --tumorsynth-command /path/to/mri_TumorSynth.",
        )
    return resolved


def _run_tumorsynth_command(cmd: list[str]) -> None:
    """Run TumorSynth and surface stdout/stderr in a compact exception."""
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        detail = "\n".join(part for part in (stdout[-2000:], stderr[-2000:]) if part)
        raise RuntimeError(
            f"TumorSynth command failed with exit code {proc.returncode}: {' '.join(cmd)}"
            + (f"\n{detail}" if detail else ""),
        )


def _save_roi_images(
    *,
    channel_paths: dict[str, Path],
    whole_tumor_mask: np.ndarray,
    out_dir: Path,
) -> dict[str, Path]:
    """Create TumorSynth inner-tumor ROI images from the whole-tumor mask."""
    roi_paths: dict[str, Path] = {}
    for name in TUMORSYNTH_CHANNEL_ORDER:
        data, affine = load_nifti(channel_paths[name])
        if data.shape[:3] != whole_tumor_mask.shape:
            raise ValueError(
                f"TumorSynth ROI shape mismatch for {name}: "
                f"{data.shape[:3]} vs whole-tumor mask {whole_tumor_mask.shape}",
            )
        roi = np.asarray(data, dtype=np.float32) * whole_tumor_mask.astype(np.float32)
        out_path = out_dir / f"roi_{name}.nii.gz"
        nib.save(nib.Nifti1Image(roi, affine), str(out_path))
        roi_paths[name] = out_path
    return roi_paths


def _preserve_tumorsynth_raw_outputs(
    *,
    whole_path: Path,
    inner_path: Path,
    preserve_raw_dir: str | Path | None,
    preserve_raw_stem: str,
) -> dict[str, Path]:
    """Copy raw TumorSynth outputs before BraTS remapping, when requested."""
    if preserve_raw_dir is None:
        return {}

    raw_dir = Path(preserve_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_whole = raw_dir / f"{preserve_raw_stem}_wholetumor_raw.nii.gz"
    raw_inner = raw_dir / f"{preserve_raw_stem}_innertumor_raw.nii.gz"
    shutil.copy2(whole_path, raw_whole)
    shutil.copy2(inner_path, raw_inner)
    return {
        "tumorsynth_whole_tumor_raw": raw_whole,
        "tumorsynth_inner_tumor_raw": raw_inner,
    }


def _tumorsynth_command_backend(
    *,
    channels: dict[str, Path],
    affine: np.ndarray,
    command: str | Path,
    nnunet_dir: str | Path | None,
    threads: int,
    cpu: bool,
    work_dir: str | Path | None,
    whole_tumor_label: int,
    inner_ncr_labels: tuple[int, ...],
    inner_ed_labels: tuple[int, ...],
    inner_et_labels: tuple[int, ...],
    preserve_raw_dir: str | Path | None = None,
    preserve_raw_stem: str = "tumorsynth",
) -> tuple[np.ndarray, str, dict[str, Path]]:
    """Run upstream ``mri_TumorSynth`` in whole-tumor + inner-tumor mode.

    TumorSynth's whole-tumor model emits healthy-tissue labels plus a
    single tumor label (18 in the upstream wrapper). Its inner-tumor
    model emits three consecutive tumor labels inside an ROI. We remap
    those labels into the project-wide BraTS convention: NCR/NET,
    ED, and ET. The whole-tumor mask is also used as an ED fallback so
    the detected WT extent is preserved if the inner model leaves any
    ROI voxels unlabeled.
    """
    if threads == 0 or threads < -1:
        raise ValueError("--tumorsynth-threads must be -1 or a positive integer")
    nnunet_root = nnunet_dir or os.environ.get("NNUNET_ENV_DIR")
    if not nnunet_root:
        raise ValueError(
            "TumorSynth requires an nnU-Net v1.7 model root. Pass "
            "--tumorsynth-nnunet-dir or set NNUNET_ENV_DIR.",
        )

    exe = _resolve_command(command)
    tmp_parent = Path(work_dir) if work_dir is not None else None
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hpgs_tumorsynth_", dir=tmp_parent) as tmp_s:
        tmp = Path(tmp_s)
        whole_path = tmp / "tumorsynth_wholetumor.nii.gz"
        inner_path = tmp / "tumorsynth_innertumor.nii.gz"

        whole_inputs = ",".join(str(channels[name]) for name in TUMORSYNTH_CHANNEL_ORDER)
        base_cmd = [exe, "--threads", str(threads), "--nnUnet", str(nnunet_root)]
        if cpu:
            base_cmd.append("--cpu")
        _run_tumorsynth_command(
            [*base_cmd, "--i", whole_inputs, "--o", str(whole_path), "--wholetumor"],
        )

        whole_lm, _ = load_nifti(whole_path)
        whole_lm = np.asarray(whole_lm)
        whole_mask = whole_lm == int(whole_tumor_label)
        if not np.any(whole_mask):
            raise RuntimeError(
                f"TumorSynth whole-tumor output has no voxels with label {whole_tumor_label}. "
                "Check --tumorsynth-whole-tumor-label before trusting the remap.",
            )

        roi_paths = _save_roi_images(
            channel_paths=channels,
            whole_tumor_mask=whole_mask,
            out_dir=tmp,
        )
        inner_inputs = ",".join(str(roi_paths[name]) for name in TUMORSYNTH_CHANNEL_ORDER)
        _run_tumorsynth_command(
            [*base_cmd, "--i", inner_inputs, "--o", str(inner_path), "--innertumor"],
        )

        inner_lm, _ = load_nifti(inner_path)
        inner_lm = np.asarray(inner_lm)
        if inner_lm.shape[:3] != whole_mask.shape:
            raise ValueError(
                "TumorSynth inner/whole shape mismatch: "
                f"{inner_lm.shape[:3]} vs {whole_mask.shape}",
            )

        raw_outputs = _preserve_tumorsynth_raw_outputs(
            whole_path=whole_path,
            inner_path=inner_path,
            preserve_raw_dir=preserve_raw_dir,
            preserve_raw_stem=preserve_raw_stem,
        )

    label_map = np.zeros(whole_mask.shape, dtype=np.uint8)
    label_map[whole_mask] = LABEL_ED
    ncr = np.isin(inner_lm, list(inner_ncr_labels)) & whole_mask
    ed = np.isin(inner_lm, list(inner_ed_labels)) & whole_mask
    et = np.isin(inner_lm, list(inner_et_labels)) & whole_mask
    label_map[ncr] = LABEL_NCR
    label_map[ed] = LABEL_ED
    label_map[et] = LABEL_ET

    # The upstream command preserves geometry; we still use the FLAIR affine
    # to keep the baseline contract identical to the unified DL output.
    if label_map.shape != tuple(int(s) for s in label_map.shape[:3]):
        raise ValueError("TumorSynth produced a non-3D label map")
    _ = affine  # documented contract: caller propagates FLAIR affine
    version = (
        f"mri_TumorSynth command={exe}; nnunet_dir={nnunet_root}; "
        f"whole_label={whole_tumor_label}; ncr_labels={inner_ncr_labels}; "
        f"ed_labels={inner_ed_labels}; "
        f"et_labels={inner_et_labels}"
    )
    return label_map, version, raw_outputs


def predict_baseline(
    *,
    baseline: Baseline,
    t1: str | Path,
    t1c: str | Path,
    t2: str | Path,
    flair: str | Path,
    backend: Backend = "docker",
    image_digest: str | None = None,
    tumorsynth_command: str | Path = "mri_TumorSynth",
    tumorsynth_nnunet_dir: str | Path | None = None,
    tumorsynth_threads: int = 8,
    tumorsynth_cpu: bool = False,
    tumorsynth_work_dir: str | Path | None = None,
    tumorsynth_whole_tumor_label: int = TUMORSYNTH_WHOLE_TUMOR_LABEL,
    tumorsynth_inner_ncr_labels: tuple[int, ...] = TUMORSYNTH_INNER_NCR_LABELS,
    tumorsynth_inner_ed_labels: tuple[int, ...] = TUMORSYNTH_INNER_ED_LABELS,
    tumorsynth_inner_et_labels: tuple[int, ...] = TUMORSYNTH_INNER_ET_LABELS,
    tumorsynth_preserve_raw_dir: str | Path | None = None,
    tumorsynth_preserve_raw_stem: str = "tumorsynth",
) -> BaselineResult:
    """Run one baseline glioma segmenter on one subject.

    Parameters
    ----------
    baseline
        ``"raidionics"``, ``"segmentglioma"``, or ``"tumorsynth"``.
    t1, t1c, t2, flair
        Paths to the four bias-corrected, skull-stripped, RAS,
        1mm-isotropic NIfTI volumes (the same inputs accepted by
        :func:`hpgs.segment.unified.predict_brats3`).
    backend
        ``"docker"`` (real, stubbed for Raidionics / segment_glioma),
        ``"command"`` (real TumorSynth command-line wrapper), or
        ``"dummy"`` (deterministic synthetic field for tests).
    image_digest
        Optional pinned upstream docker image digest. Recorded verbatim
        on the result for provenance; ignored by the dummy backend.

    Raises
    ------
    ValueError
        If ``baseline`` or ``backend`` is not one of the known values.
    NotImplementedError
        If ``backend == "docker"`` -- PR-7e2 will land the real glue.
    """
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline: {baseline!r} (expected one of {list(BASELINES)})",
        )

    channels: dict[str, str | Path] = {"T1": t1, "T1c": t1c, "T2": t2, "FLAIR": flair}
    channel_paths = {k: Path(v) for k, v in channels.items()}
    shape, affine = _load_flair_affine(flair)

    if backend == "dummy":
        label_map = _dummy_backend(shape=shape, baseline=baseline)
        version = "dummy"
        raw_outputs: dict[str, Path] = {}
    elif backend == "command":
        if baseline != "tumorsynth":
            raise NotImplementedError(
                "backend='command' is currently implemented only for "
                f"'tumorsynth', not {baseline!r}",
            )
        label_map, version, raw_outputs = _tumorsynth_command_backend(
            channels=channel_paths,
            affine=affine,
            command=tumorsynth_command,
            nnunet_dir=tumorsynth_nnunet_dir,
            threads=tumorsynth_threads,
            cpu=tumorsynth_cpu,
            work_dir=tumorsynth_work_dir,
            whole_tumor_label=tumorsynth_whole_tumor_label,
            inner_ncr_labels=tumorsynth_inner_ncr_labels,
            inner_ed_labels=tumorsynth_inner_ed_labels,
            inner_et_labels=tumorsynth_inner_et_labels,
            preserve_raw_dir=tumorsynth_preserve_raw_dir,
            preserve_raw_stem=tumorsynth_preserve_raw_stem,
        )
    elif backend == "docker":
        raise NotImplementedError(
            f"Real {baseline!r} docker backend is not wired in this repository (PR-7e2); "
            "use backend='dummy' for contract tests or backend='command' for TumorSynth.",
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r} (expected 'docker', 'command', or 'dummy')",
        )

    return BaselineResult(
        label_map=label_map,
        affine=affine,
        baseline=baseline,
        backend=backend,
        version=image_digest if (backend == "docker" and image_digest) else version,
        channel_paths=channel_paths,
        raw_outputs=raw_outputs,
    )
