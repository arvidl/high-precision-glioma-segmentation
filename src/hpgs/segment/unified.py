"""Unified deep-learning BraTS three-compartment segmenter.

This module implements the architectural commitment described in
``docs/design_segmentation_role.md``: a single state-of-the-art pretrained
brain-tumor segmenter (the MONAI Bundle BraTS three-class SegResNet) is the
canonical engine for *all* downstream Hit-Plot computations on every cohort
(LUMIERE, UCSF-PDGM, BGO). Dataset-provided masks are kept as the
*reference* against which the unified engine is benchmarked --- they are no
longer fed into the Hit-Plot pipeline.

The public API is intentionally narrow::

    from hpgs.segment.unified import predict_brats3, SegmentationResult

    result = predict_brats3(
        t1=...,
        t1c=...,
        t2=...,
        flair=...,
        backend="monai_bundle",
    )

    result.label_map     # (D, H, W) uint8, BraTS labels {0, 1, 2, 4}
    result.probabilities # (3, D, H, W) float32, sigmoid p for (TC, WT, ET)
    result.affine        # (4, 4) float, copied from the FLAIR header

Heavy dependencies (``torch``, ``monai``, network downloads) are imported
lazily so that ``import hpgs.segment.unified`` stays cheap on
documentation builds and on the CI lint-and-test path.

Backends
--------
``"monai_bundle"``
    Real inference path. Downloads the ``brats_mri_segmentation`` bundle
    from the MONAI model zoo on first use, builds the SegResNet, and runs
    sliding-window inference.
``"dummy"``
    Synthetic backend that returns a deterministic, biologically implausible
    sigmoid field. Used by the unit tests and by the smoke-test script in
    its ``--dry-run`` mode so that the *adapter contract* (channel order,
    label conversion, affine propagation) can be exercised in CI without a
    GPU, without weights, and without network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from hpgs.io import load_nifti
from hpgs.segment import LABEL_ED, LABEL_ET, LABEL_NCR

# Channel order required by the MONAI BraTS bundle.
#
# IMPORTANT: this is *not* the BraTS21 challenge order. The MONAI
# `brats_mri_segmentation` bundle (v0.4.x) was trained on BraTS 2018 data
# and its `metadata.json` declares `channel_def = {0: T1c, 1: T1, 2: T2,
# 3: FLAIR}` --- i.e. T1c precedes T1. Feeding the BraTS21 order
# `(T1, T1c, T2, FLAIR)` here silently swaps T1 and T1c, which inverts
# the contrast difference the model uses to find enhancing tumor: WT
# Dice survives, TC degrades, and ET collapses to near zero (verified on
# UCSF-PDGM sub-0005, sub-0035, sub-0068; ET-Dice 0.10 / 0.06 / 0.24
# before this fix vs.\ the bundle's reported 0.79 on its own validation).
#
# Future backends (Swin-UNETR, nnU-Net BraTS21, …) may use the BraTS21
# order; when one is added we will turn this into a per-backend constant.
CHANNEL_ORDER: tuple[str, ...] = ("T1c", "T1", "T2", "FLAIR")

# Names of the three sigmoid output channels of the BraTS three-class head.
PROB_CHANNEL_ORDER: tuple[str, ...] = ("TC", "WT", "ET")

Backend = Literal["monai_bundle", "dummy"]
Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class SegmentationResult:
    """Result of one call to :func:`predict_brats3`.

    Attributes
    ----------
    label_map
        ``(D, H, W)`` ``uint8`` array of BraTS multi-class labels:
        ``0=BG``, ``1=NCR``, ``2=ED``, ``4=ET``.
    probabilities
        ``(3, D, H, W)`` ``float32`` array of per-voxel sigmoid
        probabilities, in the order :data:`PROB_CHANNEL_ORDER`. These feed
        the *probabilistic* member of the Hit-Plot family (R2.8).
    affine
        ``(4, 4)`` voxel-to-world affine, propagated from the FLAIR header
        so the result can be saved as a NIfTI without re-loading.
    backend
        Provenance string (``"monai_bundle"`` or ``"dummy"``). Stored so a
        Hit-Plot can be tagged with the engine that produced it.
    channel_paths
        Optional record of the input file paths (handy when the caller
        wants to write a sidecar JSON next to the segmentation).
    bundle_version
        Optional MONAI bundle version string (only set for the real
        backend; ``None`` for the dummy backend).
    """

    label_map: np.ndarray
    probabilities: np.ndarray
    affine: np.ndarray
    backend: str
    channel_paths: dict[str, Path] = field(default_factory=dict)
    bundle_version: str | None = None


# ---------------------------------------------------------------------------
# Pre/post-processing helpers (pure numpy, fully testable in CI).
# ---------------------------------------------------------------------------


def _zscore_brain(volume: np.ndarray) -> np.ndarray:
    """Brain-masked per-channel z-score normalization.

    UCSF-PDGM volumes are skull-stripped, so the brain mask is well
    approximated by ``volume > 0``. Background voxels are zeroed out after
    normalisation, matching the MONAI BraTS bundle's preprocessing.

    When the foreground has zero variance (a degenerate, all-constant
    volume) we subtract the mean only and skip the division --- that
    yields a clean zero field, which is the sensible "no information"
    encoding and matches what the bundle's normalizer falls back to.
    """
    mask = volume > 0
    if not mask.any():
        return volume.astype(np.float32)
    fg = volume[mask].astype(np.float64)
    mu = float(fg.mean())
    sd = float(fg.std())
    out = volume.astype(np.float32) - np.float32(mu)
    if sd > 0.0:
        out = out / np.float32(sd)
    out[~mask] = 0.0
    return out


def _stack_channels(channels: dict[str, str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load ``CHANNEL_ORDER`` channels, brain-z-score, return stack + affine.

    Returns
    -------
    stack
        ``(4, D, H, W)`` float32 with channels in :data:`CHANNEL_ORDER`.
    affine
        ``(4, 4)`` float, taken from the FLAIR volume.
    """
    arrays: list[np.ndarray] = []
    affine: np.ndarray | None = None
    shape: tuple[int, ...] | None = None
    for name in CHANNEL_ORDER:
        if name not in channels:
            raise KeyError(f"predict_brats3 requires channel {name!r}; got {list(channels)}")
        data, aff = load_nifti(channels[name])
        if data.ndim != 3:
            raise ValueError(
                f"Channel {name!r} must be 3-D (D,H,W); got shape {data.shape}",
            )
        if shape is None:
            shape = tuple(int(s) for s in data.shape)
        elif tuple(int(s) for s in data.shape) != shape:
            raise ValueError(
                f"Channel {name!r} shape {data.shape} disagrees with reference {shape}",
            )
        arrays.append(_zscore_brain(np.asarray(data)))
        if name == "FLAIR":
            affine = aff
    assert affine is not None  # FLAIR is in CHANNEL_ORDER
    return np.stack(arrays, axis=0).astype(np.float32), affine.astype(np.float64)


def brats3_to_label_map(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert (TC, WT, ET) sigmoid probabilities to BraTS multi-class labels.

    Per BraTS convention the three binary subregions decompose as::

        ET  = ET                         -> label 4
        NCR = TC \\ ET                    -> label 1
        ED  = WT \\ TC                    -> label 2
        BG  = everything else            -> label 0

    Order of assignment matters: ED is laid down first, then NCR is written
    over the part of TC that is not enhancing, then ET is written last and
    therefore wins all ties.
    """
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError(
            f"probabilities must have shape (3, D, H, W); got {probabilities.shape}",
        )
    tc = probabilities[0] > threshold
    wt = probabilities[1] > threshold
    et = probabilities[2] > threshold
    label = np.zeros(probabilities.shape[1:], dtype=np.uint8)
    label[wt & ~tc] = LABEL_ED
    label[tc & ~et] = LABEL_NCR
    label[et] = LABEL_ET
    return label


# ---------------------------------------------------------------------------
# Backends.
# ---------------------------------------------------------------------------


def _resolve_device(device: Device) -> str:
    """Resolve ``"auto"`` to a concrete torch device string."""
    if device != "auto":
        return device
    # Lazy import: keep the lint / dummy-backend paths torch-free.
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dummy_backend(stack: np.ndarray) -> np.ndarray:
    """Deterministic, biologically implausible sigmoid field.

    Used to exercise the adapter contract (channel order, conversion,
    affine propagation) without needing torch / monai weights / network.
    The field is designed so that the resulting ``label_map`` contains
    voxels of every BraTS class, which makes downstream tests meaningful.
    """
    _, depth, height, width = stack.shape
    probs = np.zeros((3, depth, height, width), dtype=np.float32)
    # A small "tumor" centred in the volume:
    cz, cy, cx = depth // 2, height // 2, width // 2
    rz, ry, rx = max(2, depth // 8), max(2, height // 8), max(2, width // 8)
    z = np.arange(depth)[:, None, None]
    y = np.arange(height)[None, :, None]
    x = np.arange(width)[None, None, :]
    r = np.sqrt(((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    probs[1] = np.clip(1.5 - r, 0.0, 1.0)  # WT
    probs[0] = np.clip(1.2 - r, 0.0, 1.0)  # TC ⊂ WT
    probs[2] = np.clip(0.9 - r, 0.0, 1.0)  # ET ⊂ TC
    return probs


def _monai_bundle_backend(
    stack: np.ndarray,
    *,
    device: str,
    bundle_dir: Path,
) -> tuple[np.ndarray, str]:
    """Real backend: SegResNet from the MONAI BraTS bundle.

    Returns ``(probabilities, bundle_version)``.

    The     bundle is downloaded into ``bundle_dir`` on first use. We bypass the
    bundle's data loaders deliberately --- our preprocessing is already done
    in ``_stack_channels`` and matches the bundle's expected input
    distribution (RAS, 1mm isotropic, brain-masked z-score per channel).
    """
    # Lazy imports: torch + monai pull in CUDA / cuDNN probing on import,
    # which we do not want to pay for on the lint-only path or when callers
    # use the dummy backend (e.g. CI, smoke-test dry runs).
    import torch  # noqa: PLC0415
    from monai.bundle import download, load  # noqa: PLC0415
    from monai.inferers import sliding_window_inference  # noqa: PLC0415

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # `monai.bundle.download` is idempotent: it is a no-op if the bundle is
    # already on disk. We pin to the "github" source so behaviour is
    # reproducible across MONAI minor versions.
    download(
        name="brats_mri_segmentation",
        bundle_dir=str(bundle_dir),
        source="github",
    )
    # `monai.bundle.load` builds the network from `configs/inference.json`
    # (verified for brats_mri_segmentation v0.4.8: SegResNet, blocks_down
    # [1,2,2,4], blocks_up [1,1,1], 16 init filters, 4-in / 3-out, 0.2
    # dropout) and loads `models/model.pt` into it via
    # `monai.networks.utils.copy_model_state`. We deliberately pass no
    # `copy_model_args` --- the kwarg is forwarded verbatim to
    # `copy_model_state` and is *not* the same surface as torch.load's
    # `weights_only`; passing extra kwargs raises TypeError.
    model = load(
        name="brats_mri_segmentation",
        bundle_dir=str(bundle_dir),
        source="github",
    )
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"MONAI bundle load returned {type(model)!r}, expected torch.nn.Module",
        )
    model.to(device).eval()

    # Read the bundle's metadata to record provenance on the result.
    meta_path = next(
        (p for p in bundle_dir.rglob("metadata.json") if "brats" in p.as_posix().lower()),
        None,
    )
    bundle_version = "unknown"
    if meta_path is not None:
        try:
            bundle_version = str(json.loads(meta_path.read_text()).get("version", "unknown"))
        except (OSError, json.JSONDecodeError):
            bundle_version = "unknown"

    x = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, 4, D, H, W)
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            padding_mode="constant",
        )
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()
    if probs.shape[0] != 3:
        raise RuntimeError(
            f"MONAI bundle produced {probs.shape[0]} channels, expected 3 (TC, WT, ET)",
        )
    return probs.astype(np.float32), bundle_version


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def predict_brats3(
    *,
    t1: str | Path,
    t1c: str | Path,
    t2: str | Path,
    flair: str | Path,
    backend: Backend = "monai_bundle",
    device: Device = "auto",
    bundle_dir: str | Path | None = None,
    threshold: float = 0.5,
) -> SegmentationResult:
    """Run the unified BraTS three-class segmenter on one subject.

    Parameters
    ----------
    t1, t1c, t2, flair
        Paths to the four bias-corrected, skull-stripped, RAS, 1mm-isotropic
        NIfTI volumes. The UCSF-PDGM extracted layout (``*_T1_bias.nii.gz``
        etc.) and the LUMIERE pre-processed layout both satisfy this.
    backend
        ``"monai_bundle"`` (default, real inference) or ``"dummy"``
        (deterministic synthetic field for tests).
    device
        ``"auto"`` picks ``cuda``, ``mps``, or ``cpu`` in that order. Only
        consulted by the ``monai_bundle`` backend.
    bundle_dir
        Where to cache the MONAI bundle. Defaults to
        ``~/.cache/hpgs/monai_bundles``. Only consulted by the
        ``monai_bundle`` backend.
    threshold
        Sigmoid threshold used to binarise (TC, WT, ET) before converting
        to the BraTS multi-class label map. ``0.5`` matches the bundle's
        default inference recipe.
    """
    channels: dict[str, str | Path] = {"T1": t1, "T1c": t1c, "T2": t2, "FLAIR": flair}
    stack, affine = _stack_channels(channels)

    if backend == "dummy":
        probs = _dummy_backend(stack)
        bundle_version: str | None = None
    elif backend == "monai_bundle":
        cache = (
            Path(bundle_dir)
            if bundle_dir is not None
            else Path.home() / ".cache" / "hpgs" / "monai_bundles"
        )
        resolved_device = _resolve_device(device)
        probs, bundle_version = _monai_bundle_backend(
            stack,
            device=resolved_device,
            bundle_dir=cache,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'monai_bundle' or 'dummy')")

    label_map = brats3_to_label_map(probs, threshold=threshold)
    return SegmentationResult(
        label_map=label_map,
        probabilities=probs,
        affine=affine,
        backend=backend,
        channel_paths={k: Path(v) for k, v in channels.items()},
        bundle_version=bundle_version,
    )
