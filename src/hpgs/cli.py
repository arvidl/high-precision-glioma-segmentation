"""Typer-based CLI entry points for HPGS."""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from hpgs.pipeline import run_subject_pipeline

app = typer.Typer(add_completion=False, help="HPGS command-line interface.")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COHORT_STAGES = ("segment", "parcellate", "hitplot")
_STAGE_SCRIPTS = {
    "segment": "scripts/segment_all_cohort50.py",
    "parcellate": "scripts/parcellate_all_cohort50.py",
    "hitplot": "scripts/compute_hitplot_cohort50.py",
}


def _load_extract_ucsfpdgm_main():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "extract_ucsfpdgm.py"
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    spec = importlib.util.spec_from_file_location("hpgs_extract_ucsfpdgm", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@app.command()
def extract_ucsfpdgm(
    src: Annotated[Path, typer.Option(help="Path to the UCSF-PDGM v5 root.")],
    dst: Annotated[Path, typer.Option(help="Destination root.")] = Path(
        "./data/ucsf_pdgm_cohort50"
    ),
    include_optional: Annotated[
        bool, typer.Option(help="Include ADC/FA/MD/SWI/ASL channels.")
    ] = True,
) -> None:
    """Copy the 50-subject leak-safe stratified cohort into a flat ``data/`` layout."""
    extract_ucsfpdgm_main = _load_extract_ucsfpdgm_main()
    extract_ucsfpdgm_main(src=src, dst=dst, include_optional=include_optional)


@app.command()
def run_subject(
    subject: Annotated[str, typer.Option(help="4-digit subject id, e.g. '0005'.")],
    config: Annotated[Path, typer.Option(help="Pipeline config.")] = Path("configs/default.yaml"),
) -> None:
    """Run the Phase 1 single-subject pipeline."""
    outputs = run_subject_pipeline(subject, config)
    typer.echo(f"[hpgs] finished sub-{subject}")
    for name, path in outputs.items():
        typer.echo(f"[hpgs] {name}: {path}")


def cohort_stage_commands(
    *,
    cohort_yaml: Path,
    metadata_csv: Path,
    cohort_root: Path,
    deriv_root: Path,
    stages: list[str],
    subjects: list[str],
    dry_run: bool,
    device: str,
    segment_backend: str,
    parcellate_backend: str,
    fs_work_dir: Path | None,
    threads: int,
    skip_existing: bool,
) -> list[list[str]]:
    """Argv lists for the n=50 cohort scripts, in ``stages`` order.

    These are the same scripts as ``make segment-all``, ``make parcellate-all``,
    and ``make hitplot-all``.
    """
    unknown = [stage for stage in stages if stage not in _COHORT_STAGES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown cohort stage(s) {unknown}. Choose from {', '.join(_COHORT_STAGES)}."
        )

    common = [
        "--cohort-yaml",
        str(cohort_yaml),
        "--metadata-csv",
        str(metadata_csv),
        "--cohort-root",
        str(cohort_root),
        "--deriv-root",
        str(deriv_root),
        "--dry-run" if dry_run else "--no-dry-run",
    ]
    for subject in subjects:
        common.extend(["--subject", subject])
    if skip_existing and not dry_run:
        common.append("--skip-existing")

    commands: list[list[str]] = []
    for stage in stages:
        script = _REPO_ROOT / _STAGE_SCRIPTS[stage]
        cmd = [sys.executable, str(script), *common]
        if stage == "segment":
            cmd.extend(["--backend", segment_backend, "--device", device])
        elif stage == "parcellate":
            cmd.extend(
                [
                    "--backend",
                    parcellate_backend,
                    "--threads",
                    str(threads),
                ]
            )
            if fs_work_dir is not None:
                cmd.extend(["--fs-work-dir", str(fs_work_dir)])
        else:
            cmd.append("--require-parcellation")
        commands.append(cmd)
    return commands


@app.command()
def run_cohort(
    *,
    cohort_yaml: Annotated[Path, typer.Option(help="Pinned n=50 cohort manifest.")] = Path(
        "configs/cohort_ucsfpdgm_n50.yaml"
    ),
    metadata_csv: Annotated[Path, typer.Option(help="UCSF-PDGM v5 metadata CSV.")] = Path(
        "data/UCSF-PDGM-metadata_v5.csv"
    ),
    cohort_root: Annotated[Path, typer.Option(help="Extracted cohort root.")] = Path(
        "data/ucsf_pdgm_cohort50"
    ),
    deriv_root: Annotated[Path, typer.Option(help="Per-subject derivatives root.")] = Path(
        "data/derivatives_cohort50"
    ),
    stage: Annotated[
        list[str] | None,
        typer.Option(
            "--stage",
            help=(
                "Stage to run: segment, parcellate, or hitplot. "
                "Repeat to select a subset. Default: all three, in that order."
            ),
        ),
    ] = None,
    subject: Annotated[
        list[str] | None,
        typer.Option(
            "--subject",
            help="4-digit subject id. Repeat for a subset. Default: the full cohort.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Enumerate artefact paths. Does not run inference or FreeSurfer.",
        ),
    ] = False,
    device: Annotated[
        str,
        typer.Option(help="Segmentation device: auto, cpu, cuda, or mps."),
    ] = "auto",
    segment_backend: Annotated[
        str,
        typer.Option(help="Segmentation backend: monai_bundle or dummy."),
    ] = "monai_bundle",
    parcellate_backend: Annotated[
        str,
        typer.Option(help="Parcellation backend: freesurfer or dummy."),
    ] = "freesurfer",
    fs_work_dir: Annotated[
        Path | None,
        typer.Option(help="Scratch SUBJECTS_DIR. Required for a real FreeSurfer parcellation."),
    ] = None,
    threads: Annotated[int, typer.Option(help="Threads forwarded to recon-all-clinical.sh.")] = 4,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip subjects whose stage outputs are already on disk.",
        ),
    ] = False,
) -> None:
    """Run the n=50 cohort: segmentation, FreeSurfer parcellation, then Hit-Plots.

    Calls ``scripts/segment_all_cohort50.py``, ``scripts/parcellate_all_cohort50.py``,
    and ``scripts/compute_hitplot_cohort50.py``, the same scripts as
    ``make segment-all``, ``make parcellate-all``, and ``make hitplot-all``.
    """
    stages = list(stage) if stage else list(_COHORT_STAGES)
    subjects = list(subject) if subject else []
    if (
        not dry_run
        and "parcellate" in stages
        and parcellate_backend == "freesurfer"
        and fs_work_dir is None
    ):
        raise typer.BadParameter(
            "A real FreeSurfer parcellation needs --fs-work-dir (a scratch SUBJECTS_DIR). "
            "Pass --dry-run to enumerate paths, or --parcellate-backend dummy "
            "for a synthetic parcellation."
        )

    commands = cohort_stage_commands(
        cohort_yaml=cohort_yaml,
        metadata_csv=metadata_csv,
        cohort_root=cohort_root,
        deriv_root=deriv_root,
        stages=stages,
        subjects=subjects,
        dry_run=dry_run,
        device=device,
        segment_backend=segment_backend,
        parcellate_backend=parcellate_backend,
        fs_work_dir=fs_work_dir,
        threads=threads,
        skip_existing=skip_existing,
    )
    for cmd in commands:
        typer.echo("[hpgs] " + shlex.join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise typer.Exit(code=completed.returncode)
    typer.echo("[hpgs] cohort stages finished: " + ", ".join(stages))


if __name__ == "__main__":
    app()
