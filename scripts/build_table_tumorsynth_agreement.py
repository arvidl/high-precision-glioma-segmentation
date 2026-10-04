"""Build a TumorSynth agreement table from ``dl_vs_tumorsynth`` metrics sidecars.

This is a bounded-comparator companion to the main Table 2 / Table 3 producers.
It walks

``data/derivatives_cohort50/sub-XXXX/metrics/sub-XXXX_metrics_dl_vs_tumorsynth.json``

for every subject in the locked n=50 UCSF-PDGM cohort, aggregates Dice,
HD95, absolute volumetric error, sensitivity, and specificity by BraTS
compartment (WT, TC, ET), and emits:

* ``outputs/tables/table_tumorsynth_agreement.tex``
* ``outputs/tables/table_tumorsynth_agreement.csv``
* ``outputs/tables/table_tumorsynth_agreement.json``

The table is intentionally separate from ``table3_segmenter_agreement`` because
TumorSynth is framed in the manuscript as an optional segmentation-backbone
sensitivity analysis, not as part of the original Raidionics / segment_glioma
Table 3 design.

Usage::

    uv run python scripts/build_table_tumorsynth_agreement.py

Exit codes:
    0   table built and written; n_used > 0 for every cell
    1   table built but at least one cell had n_used == 0
    2   could not load the cohort YAML / metrics root unreachable
    3   --strict and at least one sidecar missing / malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from build_table2_dl_vs_gt import (
    _METRIC_KEYS,
    _METRIC_LABEL,
    ACCEPTED_METRICS_SCHEMA_VERSIONS,
    COMPARTMENTS,
    METRICS,
    _classify_value,
    _fmt_cell,
    summarise_per_cell,
)
from hpgs.io import load_cohort_metadata, normalize_subject_id

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_METRICS_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_TEX = Path("outputs/tables/table_tumorsynth_agreement.tex")
DEFAULT_OUT_CSV = Path("outputs/tables/table_tumorsynth_agreement.csv")
DEFAULT_OUT_JSON = Path("outputs/tables/table_tumorsynth_agreement.json")

COMPARISON = "dl_vs_tumorsynth"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cohort-yaml",
        type=Path,
        default=DEFAULT_COHORT_YAML,
        help="Cohort YAML (default: %(default)s).",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="UCSF-PDGM v5 metadata CSV (default: %(default)s).",
    )
    p.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help="Per-subject derivatives root with metrics/ subdirs (default: %(default)s).",
    )
    p.add_argument(
        "--out-tex",
        type=Path,
        default=DEFAULT_OUT_TEX,
        help="Where to write the LaTeX tabular block (default: %(default)s).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Where to write the long-format per-subject audit CSV (default: %(default)s).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Where to write the per-cell summary JSON (default: %(default)s).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on the first missing / malformed sidecar.",
    )
    return p.parse_args()


def _metrics_path(metrics_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return metrics_root / f"sub-{sid}" / "metrics" / f"sub-{sid}_metrics_{COMPARISON}.json"


class SidecarError(ValueError):
    """Raised when a TumorSynth metrics sidecar is missing required structure."""


def _validate_sidecar(doc: Mapping[str, Any], *, source: Path | str = "<dict>") -> None:
    """Validate a parsed ``metrics_dl_vs_tumorsynth.json`` document."""
    if "schema_version" not in doc:
        raise SidecarError(f"{source}: missing 'schema_version'")
    sv = str(doc["schema_version"])
    if sv not in ACCEPTED_METRICS_SCHEMA_VERSIONS:
        raise SidecarError(
            f"{source}: schema_version {sv!r} not in {sorted(ACCEPTED_METRICS_SCHEMA_VERSIONS)}",
        )
    if doc.get("comparison") != COMPARISON:
        raise SidecarError(
            f"{source}: comparison={doc.get('comparison')!r}, expected {COMPARISON!r}",
        )
    metrics = doc.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SidecarError(f"{source}: 'metrics' must be a mapping")
    for comp in COMPARTMENTS:
        if comp not in metrics:
            raise SidecarError(f"{source}: missing compartment {comp!r}")
        cell = metrics[comp]
        if not isinstance(cell, Mapping):
            raise SidecarError(f"{source}: metrics[{comp!r}] must be a mapping")
        for mk in _METRIC_KEYS:
            if mk not in cell:
                raise SidecarError(f"{source}: missing metric metrics[{comp!r}][{mk!r}]")


def load_metrics_sidecars(
    metrics_root: Path,
    subject_ids: Iterable[str],
    *,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Load and validate every per-subject TumorSynth metrics sidecar."""
    records: list[dict[str, Any]] = []
    problems: list[tuple[str, str]] = []
    for raw_sid in subject_ids:
        sid = normalize_subject_id(raw_sid)
        path = _metrics_path(metrics_root, sid)
        if not path.is_file():
            reason = f"sidecar not found at {path}"
            problems.append((sid, reason))
            if strict:
                raise FileNotFoundError(reason)
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            reason = f"could not parse JSON ({exc})"
            problems.append((sid, reason))
            if strict:
                raise
            continue
        try:
            _validate_sidecar(doc, source=path)
        except SidecarError as exc:
            problems.append((sid, str(exc)))
            if strict:
                raise
            continue
        doc["subject_id"] = normalize_subject_id(doc.get("subject_id", sid))
        records.append(dict(doc))
    return records, problems


def long_audit_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Long-format per-subject CSV for ``dl_vs_tumorsynth``."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        sid = rec["subject_id"]
        for comp in COMPARTMENTS:
            for mk in _METRIC_KEYS:
                raw = rec["metrics"][comp][mk]
                status, val = _classify_value(raw)
                rows.append(
                    {
                        "subject_id": sid,
                        "comparison": COMPARISON,
                        "compartment": comp,
                        "metric": mk,
                        "value": val if val is not None else pd.NA,
                        "status": status,
                        "raw": raw if not isinstance(raw, float) else float(raw),
                    },
                )
    df = pd.DataFrame(
        rows,
        columns=["subject_id", "comparison", "compartment", "metric", "value", "status", "raw"],
    )
    return df.sort_values(["compartment", "metric", "subject_id"], kind="stable").reset_index(
        drop=True
    )


def _denominator_note(
    summary: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    n_total: int,
) -> str:
    deltas: list[str] = []
    for comp in COMPARTMENTS:
        for mk in _METRIC_KEYS:
            cell = summary[comp][mk]
            n_used = int(cell["n_used"])
            if n_used != n_total:
                deltas.append(
                    f"{comp}/{_METRIC_LABEL[mk]}: n={n_used}/{n_total}"
                    f" (missing={int(cell['n_missing'])},"
                    f" inf={int(cell['n_inf'])})",
                )
    if not deltas:
        return ""
    return (
        r"\textit{Per-cell denominator (when different from n="
        + str(n_total)
        + "):} "
        + "; ".join(deltas)
        + "."
    )


def format_table_tex(
    summary: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    *,
    n_total: int,
    n_used_cohort: int,
) -> str:
    """Render the TumorSynth summary as a self-contained LaTeX tabular block."""
    lines: list[str] = []
    lines.append(
        "% AUTO-GENERATED by scripts/build_table_tumorsynth_agreement.py --- DO NOT EDIT BY HAND.",
    )
    lines.append(
        "% Per-subject sidecars: data/derivatives_cohort50/sub-XXXX/metrics/"
        "sub-XXXX_metrics_dl_vs_tumorsynth.json.",
    )
    lines.append(
        f"% Cohort: configs/cohort_ucsfpdgm_n50.yaml, "
        f"n_total={n_total}, n_with_sidecar={n_used_cohort}.",
    )
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Metric & WT & TC & ET \\")
    lines.append(r"\midrule")
    for mk, label, precision in METRICS:
        cells = " & ".join(_fmt_cell(summary[comp][mk], precision) for comp in COMPARTMENTS)
        lines.append(f"{label} & {cells} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    note = _denominator_note(summary, n_used_cohort)
    if note:
        lines.append(r"%")
        lines.append(f"% Footnote candidate: {note}")
    return "\n".join(lines) + "\n"


def _print_summary_to_stdout(
    summary: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    *,
    n_total: int,
    n_used_cohort: int,
    n_problems: int,
) -> None:
    print()
    print("=" * 86)
    print(
        f"TUMORSYNTH  Table (DL vs mri_TumorSynth)  "
        f"cohort_n={n_total}  loaded={n_used_cohort}  problems={n_problems}",
    )
    print("=" * 86)
    print(f"  {'metric':<28}{'WT':>14}{'TC':>14}{'ET':>14}")
    print("-" * 86)
    for mk, _label, precision in METRICS:
        cells = []
        for comp in COMPARTMENTS:
            cell = summary[comp][mk]
            if cell["n_used"] == 0 or cell["median"] is None:
                cells.append("--")
            else:
                fmt = f"{{:.{precision}f}}"
                cells.append(fmt.format(cell["median"]))
        print(f"  {mk:<28}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")
    print("=" * 86)


def main() -> int:
    args = _parse_args()

    try:
        cm = load_cohort_metadata(args.cohort_yaml, args.metadata_csv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: failed to load cohort: {exc}", file=sys.stderr)
        return 2
    if not args.metrics_root.exists():
        print(
            f"error: metrics root does not exist: {args.metrics_root}",
            file=sys.stderr,
        )
        return 2

    subject_ids = list(cm.df.index)
    try:
        records, problems = load_metrics_sidecars(
            args.metrics_root,
            subject_ids,
            strict=args.strict,
        )
    except (FileNotFoundError, json.JSONDecodeError, SidecarError) as exc:
        print(f"error (--strict): {exc}", file=sys.stderr)
        return 3

    for sid, reason in problems:
        print(f"warning: sub-{sid}: {reason}", file=sys.stderr)

    summary = summarise_per_cell(records)
    audit_df = long_audit_frame(records)

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(
        format_table_tex(summary, n_total=cm.n, n_used_cohort=len(records)),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.out_csv, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cohort_yaml": str(args.cohort_yaml),
                "metrics_root": str(args.metrics_root),
                "comparison": COMPARISON,
                "n_total": int(cm.n),
                "n_with_sidecar": len(records),
                "n_problems": len(problems),
                "compartments": list(COMPARTMENTS),
                "metric_keys": list(_METRIC_KEYS),
                "summary": summary,
                "problems": [{"subject_id": sid, "reason": r} for sid, r in problems],
            },
            indent=2,
        ),
    )

    _print_summary_to_stdout(
        summary,
        n_total=cm.n,
        n_used_cohort=len(records),
        n_problems=len(problems),
    )
    print(f"  wrote: {args.out_tex}")
    print(f"  wrote: {args.out_csv}")
    print(f"  wrote: {args.out_json}")

    any_empty_cell = any(
        summary[comp][mk]["n_used"] == 0 for comp in COMPARTMENTS for mk in _METRIC_KEYS
    )
    return 1 if any_empty_cell else 0


if __name__ == "__main__":
    sys.exit(main())
