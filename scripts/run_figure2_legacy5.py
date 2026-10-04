"""Regenerate the eight panels of Figure 2 for each of the five legacy subjects.

This cohort-level driver iterates over the subjects listed in
``configs/figure2_legacy5.yaml`` and, for each one:

1. Invokes ``scripts/run_fs_clinical.py`` to ensure SynthSR, SynthSeg,
   aparc+aseg and wmparc exist in the derivatives tree (idempotent; skipped
   when outputs already exist unless ``--overwrite`` is passed).
2. Invokes ``scripts/render_figure2_panels.py`` to render the eight panel
   PNGs into ``paths.figure_output``.

Usage::

    uv run python scripts/run_figure2_legacy5.py \
        --config configs/figure2_legacy5.yaml --threads 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUN_FS_CLINICAL = HERE / "run_fs_clinical.py"
RENDER_PANELS = HERE / "render_figure2_panels.py"


def _load_config(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-fs",
        action="store_true",
        help="Skip the FS clinical stage and only regenerate panel PNGs.",
    )
    parser.add_argument(
        "--fs-config",
        type=Path,
        default=Path("configs/legacy5.yaml"),
        help="Config passed to run_fs_clinical.py (supplies data_root, derivatives, "
        "freesurfer_subjects). Defaults to configs/legacy5.yaml.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = cfg["paths"]
    data_root = Path(paths["data_root"]).resolve()
    derivatives_root = Path(paths["derivatives"]).resolve()
    figure_output = Path(paths["figure_output"]).resolve()
    name_prefix = cfg.get("name_prefix", "glioma-localization-figs-UCSF-PDGM-")
    subjects = cfg["subjects"]

    figure_output.mkdir(parents=True, exist_ok=True)

    for entry in subjects:
        sid = str(entry["id"])
        coord = [str(int(x)) for x in entry["coord"]]
        print(f"=== sub-{sid} (coord {','.join(coord)}) ===")
        if not args.skip_fs:
            fs_cmd = [
                sys.executable,
                str(RUN_FS_CLINICAL),
                "--subject",
                sid,
                "--config",
                str(args.fs_config),
                "--threads",
                str(args.threads),
            ]
            if args.overwrite:
                fs_cmd.append("--overwrite")
            _run(fs_cmd)

        render_cmd = [
            sys.executable,
            str(RENDER_PANELS),
            "--subject",
            sid,
            "--data-root",
            str(data_root),
            "--derivatives-root",
            str(derivatives_root),
            "--output-dir",
            str(figure_output),
            "--name-prefix",
            name_prefix,
            "--coord",
            *coord,
        ]
        _run(render_cmd)


if __name__ == "__main__":
    main()
