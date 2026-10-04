"""Render the eight panels (a..h) of manuscript Figure 2 for a UCSF-PDGM subject.

Produces eight PNG files, one per Freeview-style panel, each laid out as a
1x3 sagittal/coronal/axial strip at a fixed voxel coordinate:

    a) Original T1
    b) Original Gadolinium-contrast T1 (bias-corrected channel from UCSF-PDGM)
    c) Synthetic super-resolution T1 (mri_synthsr)
    d) Original FLAIR (bias-corrected channel from UCSF-PDGM)
    e) Tumor segmentation overlaid on the bias-corrected T1
    f) aparc+aseg overlaid on the bias-corrected T1 (recon-all-clinical output)
    g) FreeSurfer 8.2.0 SynthSeg parcellation overlaid on the bias-corrected T1
    h) wmparc overlaid on the bias-corrected T1 (recon-all-clinical output)

All volumes are reoriented to canonical RAS on load and displayed in the
conventional *radiological* orientation (image-left = patient-right for the
axial and coronal planes; anterior-left for the sagittal plane; superior up
for axial/sagittal). The manuscript crosshair voxel coordinate is specified
in the original (LPS) array frame and translated to RAS internally so the
pinned landmarks (e.g. 0020 at [159, 147, 95]) continue to match Freeview.

The output PNGs are suitable as drop-in includes for the 4x2 tabular layout
of Figure 2 in ``paper/main.tex``. Label panels (f/g/h) are colored
with the FreeSurferColorLUT to visually match Freeview exports. Panel e uses
the three-compartment tumor LUT (NCR=1 red, ED=2 green, ET=4 yellow).

Usage::

    uv run python scripts/render_figure2_panels.py --subject 0020 \
        --data-root data/ucsf_pdgm_legacy5 \
        --derivatives-root data/derivatives_legacy5 \
        --output-dir outputs/legacy5/figure2 \
        --coord 159 147 95
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hpgs.io import load_nifti_canonical, orig_voxel_to_ras
from hpgs.viz import colorize_label_volume, parse_freesurfer_color_lut

TUMOR_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 60, 60),
    2: (60, 220, 80),
    4: (255, 215, 0),
}

PLANE_TITLES = ("Sagittal", "Coronal", "Axial")


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    title: str
    base_key: str
    overlay_key: str | None
    overlay_kind: str  # "none" | "tumor" | "lut"


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("a", "Original T1", "t1_raw", None, "none"),
    PanelSpec("b", "T1c (Gd)", "t1c_bias", None, "none"),
    PanelSpec("c", "SynthSR", "synthsr", None, "none"),
    PanelSpec("d", "FLAIR", "flair_bias", None, "none"),
    PanelSpec("e", "Tumor segmentation", "t1_bias", "tumor", "tumor"),
    PanelSpec("f", "aparc+aseg", "t1_bias", "aparc_aseg", "lut"),
    PanelSpec("g", "FS 8.2.0 SynthSeg", "t1_bias", "synthseg", "lut"),
    PanelSpec("h", "wmparc", "t1_bias", "wmparc", "lut"),
)


def _normalize_intensity(volume: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(volume, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _radiological(view: np.ndarray) -> np.ndarray:
    """Apply the radiological display transform ``fliplr(rot90(...))``.

    On an RAS-reoriented volume this produces the conventional radiological
    display for all three canonical planes (image-left = patient-right for
    axial/coronal; anterior-left for sagittal; superior up for axial/sagittal).
    Broadcasts safely over trailing channel dimensions so LUT-colorised RGB
    slices keep their colors intact.
    """
    return np.fliplr(np.rot90(view))


def _triptych_base(volume_ras: np.ndarray, coord_ras: tuple[int, int, int]) -> list[np.ndarray]:
    i, j, k = coord_ras
    return [
        _radiological(volume_ras[i, :, :]),
        _radiological(volume_ras[:, j, :]),
        _radiological(volume_ras[:, :, k]),
    ]


def _triptych_labels(volume_ras: np.ndarray, coord_ras: tuple[int, int, int]) -> list[np.ndarray]:
    return _triptych_base(volume_ras, coord_ras)


def _draw_crosshair(
    ax,
    plane_idx: int,
    coord_ras: tuple[int, int, int],
    ras_shape: tuple[int, int, int],
) -> None:
    """Overlay crosshair lines on a radiological triptych panel.

    Given RAS voxel coords (i_R, j_A, k_S) and the RAS volume shape (R, A, S),
    project the target voxel onto each of the three displayed planes. The
    radiological transform places the crosshair at::

        sagittal (R-fixed):  x = A-1 - j,  y = S-1 - k
        coronal  (A-fixed):  x = R-1 - i,  y = S-1 - k
        axial    (S-fixed):  x = R-1 - i,  y = A-1 - j
    """
    i, j, k = coord_ras
    size_r, size_a, size_s = ras_shape
    if plane_idx == 0:  # sagittal
        x = size_a - 1 - j
        y = size_s - 1 - k
    elif plane_idx == 1:  # coronal
        x = size_r - 1 - i
        y = size_s - 1 - k
    else:  # axial
        x = size_r - 1 - i
        y = size_a - 1 - j
    ax.axhline(y, color="white", linewidth=0.5, alpha=0.85)
    ax.axvline(x, color="white", linewidth=0.5, alpha=0.85)


def _overlay_views_for(
    spec: PanelSpec,
    overlays: dict[str, np.ndarray | None],
    coord_ras: tuple[int, int, int],
    lut: dict[int, tuple[int, int, int]] | None,
) -> list[np.ndarray] | None:
    if spec.overlay_key is None:
        return None
    labels = overlays.get(spec.overlay_key)
    if labels is None:
        return None
    if spec.overlay_kind == "tumor":
        return _triptych_labels(labels, coord_ras)
    if spec.overlay_kind == "lut":
        if lut is None:
            raise ValueError("LUT required for label overlays.")
        rgb = colorize_label_volume(labels.astype(np.int32), lut)
        i, j, k = coord_ras
        return [
            _radiological(rgb[i, :, :, :]),
            _radiological(rgb[:, j, :, :]),
            _radiological(rgb[:, :, k, :]),
        ]
    raise ValueError(f"Unknown overlay kind: {spec.overlay_kind}")


def _draw_tumor_overlay(ax, view: np.ndarray) -> None:
    for label, color in TUMOR_COLORS.items():
        mask = view == label
        if not mask.any():
            continue
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = color[0] / 255.0
        rgba[..., 1] = color[1] / 255.0
        rgba[..., 2] = color[2] / 255.0
        rgba[..., 3] = mask.astype(np.float32) * 0.6
        ax.imshow(rgba, interpolation="nearest")


def _draw_lut_overlay(ax, view: np.ndarray, alpha: float) -> None:
    rgba = np.zeros((*view.shape[:2], 4), dtype=np.float32)
    rgba[..., :3] = view.astype(np.float32) / 255.0
    nonzero = (view.sum(axis=-1) > 0).astype(np.float32)
    rgba[..., 3] = nonzero * alpha
    ax.imshow(rgba, interpolation="nearest")


def _render_panel(
    spec: PanelSpec,
    bases: dict[str, np.ndarray],
    overlays: dict[str, np.ndarray | None],
    coord_ras: tuple[int, int, int],
    ras_shape: tuple[int, int, int],
    subject_label: str,
    output: Path,
    *,
    lut: dict[int, tuple[int, int, int]] | None,
    overlay_alpha: float = 0.45,
) -> None:
    base = bases.get(spec.base_key)
    if base is None:
        raise KeyError(f"Missing base volume for panel {spec.letter}: {spec.base_key}")
    base_views = _triptych_base(_normalize_intensity(base), coord_ras)
    overlay_views = _overlay_views_for(spec, overlays, coord_ras, lut)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for idx, plane in enumerate(PLANE_TITLES):
        ax = axes[idx]
        ax.imshow(base_views[idx], cmap="gray", interpolation="nearest")
        if overlay_views is not None:
            ov = overlay_views[idx]
            if spec.overlay_kind == "tumor":
                _draw_tumor_overlay(ax, ov)
            else:
                _draw_lut_overlay(ax, ov, overlay_alpha)
        _draw_crosshair(ax, idx, coord_ras, ras_shape)
        ax.set_title(plane, fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{subject_label} — {spec.letter}) {spec.title}", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_canonical_only(path: Path) -> np.ndarray:
    """Load a NIfTI and return only its canonical-RAS data."""
    data, _orig_shape, _transform = load_nifti_canonical(path)
    return data


def _gather_inputs(
    subject_id: str,
    data_root: Path,
    derivatives_root: Path,
    coord_orig: tuple[int, int, int],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray | None],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    """Load and reorient all inputs for one subject; return canonical-RAS data.

    The reference frame is the raw T1 (``sub-XXXX_T1.nii.gz``) since that is the
    volume the manuscript crosshair was originally picked on in Freeview. All
    other volumes are co-registered on the same voxel grid (UCSF-PDGM ships
    co-registered mpMRI + tumor segmentation, and the FS clinical derivatives
    are resampled back onto that grid via
    :func:`hpgs.parcellate.resample_like`), so a single RAS reorientation and
    crosshair apply to every panel.
    """
    subject_dir = data_root / f"sub-{subject_id}"
    deriv = derivatives_root / f"sub-{subject_id}"
    pre = deriv / "preprocess"
    par = deriv / "parcellation"

    t1_raw_path = subject_dir / f"sub-{subject_id}_T1.nii.gz"
    t1_raw_ras, t1_raw_shape, t1_raw_transform = load_nifti_canonical(t1_raw_path)
    coord_ras = orig_voxel_to_ras(coord_orig, t1_raw_shape, t1_raw_transform)
    ras_shape = (
        int(t1_raw_ras.shape[0]),
        int(t1_raw_ras.shape[1]),
        int(t1_raw_ras.shape[2]),
    )

    bases: dict[str, np.ndarray] = {"t1_raw": t1_raw_ras}
    bases["t1_bias"] = _load_canonical_only(subject_dir / f"sub-{subject_id}_T1_bias.nii.gz")
    bases["t1c_bias"] = _load_canonical_only(subject_dir / f"sub-{subject_id}_T1c_bias.nii.gz")
    bases["flair_bias"] = _load_canonical_only(subject_dir / f"sub-{subject_id}_FLAIR_bias.nii.gz")

    synthsr_candidates = [
        pre / "synthSR_standalone.nii.gz",
        pre / "synthSR_recon.nii.gz",
    ]
    synthsr_path = next((p for p in synthsr_candidates if p.is_file()), None)
    if synthsr_path is not None:
        bases["synthsr"] = _load_canonical_only(synthsr_path)
    else:
        print(
            f"[warning] no SynthSR volume found for sub-{subject_id}; panel c will fall back to T1."
        )
        bases["synthsr"] = bases["t1_raw"]

    overlays: dict[str, np.ndarray | None] = {}
    overlays["tumor"] = _load_canonical_only(
        subject_dir / f"sub-{subject_id}_tumor_segmentation.nii.gz"
    )
    overlays["synthseg"] = (
        _load_canonical_only(par / "synthseg.nii.gz")
        if (par / "synthseg.nii.gz").is_file()
        else None
    )
    overlays["aparc_aseg"] = (
        _load_canonical_only(par / "aparc+aseg.nii.gz")
        if (par / "aparc+aseg.nii.gz").is_file()
        else None
    )
    overlays["wmparc"] = (
        _load_canonical_only(par / "wmparc.nii.gz") if (par / "wmparc.nii.gz").is_file() else None
    )
    for key in ("aparc_aseg", "synthseg", "wmparc"):
        if overlays[key] is None:
            print(
                f"[warning] {key} missing for sub-{subject_id}; "
                "panel will be rendered without overlay."
            )
    return bases, overlays, coord_ras, ras_shape


def render_subject_panels(
    *,
    subject_id: str,
    data_root: Path,
    derivatives_root: Path,
    coord: tuple[int, int, int],
    output_dir: Path,
    name_prefix: str,
) -> list[Path]:
    bases, overlays, coord_ras, ras_shape = _gather_inputs(
        subject_id, data_root, derivatives_root, coord
    )
    lut = parse_freesurfer_color_lut()
    subject_label = f"UCSF-PDGM-{subject_id}"

    outputs: list[Path] = []
    for spec in PANELS:
        out_path = output_dir / f"{name_prefix}{subject_id}_panel_{spec.letter}.png"
        _render_panel(
            spec,
            bases=bases,
            overlays=overlays,
            coord_ras=coord_ras,
            ras_shape=ras_shape,
            subject_label=subject_label,
            output=out_path,
            lut=lut,
        )
        outputs.append(out_path)
    return outputs


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--derivatives-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coord",
        nargs=3,
        type=int,
        metavar=("I", "J", "K"),
        required=True,
        help=(
            "Voxel coordinate (in the ORIGINAL NIfTI array frame, typically LPS for "
            "UCSF-PDGM) used for the sagittal/coronal/axial slices. Internally "
            "translated to the canonical RAS frame before rendering."
        ),
    )
    parser.add_argument(
        "--name-prefix",
        default="glioma-localization-figs-UCSF-PDGM-",
        help="File-name prefix; the subject id and panel letter are appended.",
    )
    args = parser.parse_args()

    outputs = render_subject_panels(
        subject_id=args.subject,
        data_root=args.data_root,
        derivatives_root=args.derivatives_root,
        coord=tuple(args.coord),
        output_dir=args.output_dir,
        name_prefix=args.name_prefix,
    )
    for p in outputs:
        print(p)


if __name__ == "__main__":
    _cli()
