"""Register all LUMIERE Patient-048 timepoints to the pre-operative reference.

Thin CLI shim over :func:`hpgs.register.lumiere.register_subject_longitudinal`,
mirroring the style of ``scripts/extract_lumiere_p048.py``: the cohort
definition (which timepoints, which channels, which reference, which moving-channel
overrides) lives in ``configs/lumiere_p048_timepoints.yaml`` and the per-file
data layout is taken from ``data/lumiere_p048/`` (produced by the extractor).

Usage
-----

    uv run python scripts/register_lumiere_p048.py

    uv run python scripts/register_lumiere_p048.py \
        --derivatives-root ./data/lumiere_p048/derivatives \
        --include-legacy-seg

Outputs land under ``<stage-root>/derivatives/`` (or ``--derivatives-root``).
The orchestrator writes a ``manifest.json`` at the derivatives root and a
``registration_qc.json`` per timepoint with grid-match, correlation, and
transform-path information; both are intended to be diffable across runs
(deterministic ANTsPyX seed = ``hpgs.register.DEFAULT_SEED``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hpgs.register import DEFAULT_SEED
from hpgs.register.lumiere import register_subject_longitudinal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_STAGE = REPO_ROOT / "data" / "lumiere_p048"
DEFAULT_DERIVATIVES = DEFAULT_STAGE / "derivatives"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Cohort manifest YAML (timepoints + reference + channels). Default: "
            f"{DEFAULT_MANIFEST.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE,
        help=(
            f"Staged data root (extractor output). Default: {DEFAULT_STAGE.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Where to write registered outputs + per-tp QC. Default: "
            f"{DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--include-legacy-seg",
        action="store_true",
        help="Also warp the LUMIERE-team legacy segmentations (HD-GLIO-AUTO, DeepBraTumIA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic ANTsPyX seed. Default: {DEFAULT_SEED} (project-wide).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run registration and overwrite existing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    manifest_path = args.manifest.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    derivatives_root = args.derivatives_root.expanduser().resolve()

    if not manifest_path.is_file():
        print(f"[err] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not stage_root.is_dir():
        print(
            f"[err] stage-root not found: {stage_root}\n"
            "  Run scripts/extract_lumiere_p048.py first to stage the cohort.",
            file=sys.stderr,
        )
        return 2

    provenance = register_subject_longitudinal(
        manifest_path=manifest_path,
        stage_root=stage_root,
        derivatives_root=derivatives_root,
        include_legacy_seg=args.include_legacy_seg,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    n_tps = len(provenance["timepoints"])
    n_warnings = sum(len(tp["warnings"]) for tp in provenance["timepoints"])
    label = (
        derivatives_root.relative_to(REPO_ROOT)
        if derivatives_root.is_relative_to(REPO_ROOT)
        else derivatives_root
    )
    print(f"[ok] Registered {n_tps} timepoint(s); outputs under {label}")
    if n_warnings:
        print(
            f"[warn] {n_warnings} per-timepoint warning(s); see registration_qc.json",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
