"""Typer-based CLI entry points for HPGS."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Annotated

import typer

from hpgs.pipeline import run_subject_pipeline

app = typer.Typer(add_completion=False, help="HPGS command-line interface.")


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


@app.command()
def run_cohort(
    config: Annotated[Path, typer.Option(help="Pipeline config.")] = Path("configs/default.yaml"),
) -> None:
    """Run the pipeline for the full 50-subject cohort (stub — Cursor TODO)."""
    typer.echo(f"[hpgs] would run cohort pipeline with config {config}")


if __name__ == "__main__":
    app()
