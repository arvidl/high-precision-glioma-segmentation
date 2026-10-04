r"""Sync code-generated figures and tables into ``paper/figs/``.

Workflow
--------
1. Notebooks and scripts under ``src/hpgs`` write figures to
   ``outputs/figures/<name>.{png,pdf,svg}`` and auto-generated LaTeX
   tables to ``outputs/tables/<name>.tex``.
2. This script copies (or updates, by mtime) both kinds of artefacts into
   ``paper/figs/``, preserving filenames so ``\includegraphics{figs/<name>}``
   and ``\input{figs/<name>}`` keep working.
3. Stale entries can be pruned with ``--prune``.

Usage
-----
    python scripts/sync_figs_to_paper.py
    python scripts/sync_figs_to_paper.py --pattern "*.pdf"
    python scripts/sync_figs_to_paper.py --prune
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DST_DIR = REPO_ROOT / "paper" / "figs"

# (source_dir, default_glob_patterns) — each source is optional; missing
# directories are skipped with a note rather than raising.
DEFAULT_SOURCES: tuple[tuple[Path, tuple[str, ...]], ...] = (
    (REPO_ROOT / "outputs" / "figures", ("*.png", "*.pdf", "*.svg")),
    (REPO_ROOT / "outputs" / "tables", ("*.tex",)),
)


def main(
    patterns: tuple[str, ...] | None = None,
    *,
    prune: bool = False,
    dry_run: bool = False,
) -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    src_files: dict[str, Path] = {}
    any_source_present = False
    for src_dir, default_pats in DEFAULT_SOURCES:
        if not src_dir.is_dir():
            print(f"[skip] {src_dir.relative_to(REPO_ROOT)} not present")
            continue
        any_source_present = True
        pats = patterns if patterns is not None else default_pats
        for pat in pats:
            for p in src_dir.rglob(pat):
                src_files[p.name] = p

    if not any_source_present:
        raise FileNotFoundError(
            "Neither outputs/figures/ nor outputs/tables/ is present. "
            "Run a notebook or pipeline stage that writes artefacts first."
        )

    n_copied, n_skipped = 0, 0
    for name, src in src_files.items():
        dst = DST_DIR / name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            n_skipped += 1
            continue
        action = "would copy" if dry_run else "copying"
        print(f"[{action}] {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
        if not dry_run:
            shutil.copy2(src, dst)
        n_copied += 1

    if prune:
        keep = set(src_files.keys()) | {"README.md"}
        for p in DST_DIR.iterdir():
            if p.is_file() and p.name not in keep:
                action = "would prune" if dry_run else "pruning"
                print(f"[{action}] {p.relative_to(REPO_ROOT)}")
                if not dry_run:
                    p.unlink()

    print(f"[ok] copied {n_copied}, skipped {n_skipped} (up-to-date).")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Glob to copy (repeat for multiple). "
            "Default: *.png/*.pdf/*.svg from outputs/figures/, *.tex from outputs/tables/."
        ),
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help=(
            "Remove paper/figs/* files that are no longer present in any of the source directories."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    pats = tuple(a.pattern) if a.pattern else None
    main(patterns=pats, prune=a.prune, dry_run=a.dry_run)


if __name__ == "__main__":
    _cli()
