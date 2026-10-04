"""PR-7g: build Table 2 (DL vs UCSF-PDGM reference) from the PR-7d sidecars.

Walks ``data/derivatives_cohort50/sub-XXXX/metrics/sub-XXXX_metrics_dl_vs_gt.json``
for every subject in the locked cohort YAML
(``configs/cohort_ucsfpdgm_n50.yaml``), aggregates the per-compartment
Q2-C metric bundle (Dice, HD95, |VE|, sensitivity, specificity) into a
median [Q1, Q3] summary, and emits three artefacts:

* ``outputs/tables/table2_dl_vs_gt.tex`` --- a self-contained
  ``\\begin{tabular}{...}`` block (no float wrapper) ready to
  ``\\input{}`` from ``paper/main.tex``;
* ``outputs/tables/table2_dl_vs_gt.csv`` --- long-format per-subject
  audit trail (one row per (subject, compartment, metric)) with
  ``status \\in {ok, missing, inf}`` so reviewers can recompute any
  cell from the raw sidecars;
* ``outputs/tables/table2_dl_vs_gt.json`` --- the same per-cell summary
  the LaTeX table renders, kept as a machine-readable companion for
  the upcoming Hit-Plot agreement panel and for the response letter.

JSON values written by PR-7d that are ``null`` (NaN) or the string
``"inf"`` are excluded from the median / IQR (NaN means *undefined*,
e.g. empty reference compartment for sensitivity; "inf" means
*catastrophic*, e.g. one mask was empty so HD95 has no finite
distance). Each cell records ``n_used`` (finite samples) and
``n_missing = n_nan + n_inf`` so reviewers can audit the denominator.

sidecars, the producer of the table cell values requested by R2.2,
and the prerequisite for the cross-cell agreement summary in PR-7h
(R2.5 / R2.10).

Usage::

    # Default --- whole cohort:
    uv run python scripts/build_table2_dl_vs_gt.py

    # Equivalent via the Make wrapper:
    make table2-dl-vs-gt

    # Override paths (e.g. for a partial run on a subset):
    uv run python scripts/build_table2_dl_vs_gt.py \
        --cohort-yaml configs/cohort_ucsfpdgm_n50.yaml \
        --metrics-root data/derivatives_cohort50 \
        --out-tex outputs/tables/table2_dl_vs_gt.tex \
        --out-csv outputs/tables/table2_dl_vs_gt.csv \
        --out-json outputs/tables/table2_dl_vs_gt.json \
        --strict   # fail on the first missing / malformed sidecar

Exit codes:
    0   table built and written; n_used > 0 for every cell
    1   table built but at least one cell had n_used == 0 (warning)
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

import numpy as np
import pandas as pd

from hpgs.io import load_cohort_metadata, normalize_subject_id

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_METRICS_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_TEX = Path("outputs/tables/table2_dl_vs_gt.tex")
DEFAULT_OUT_CSV = Path("outputs/tables/table2_dl_vs_gt.csv")
DEFAULT_OUT_JSON = Path("outputs/tables/table2_dl_vs_gt.json")

# Columns of Table 2 (locked by Q2-C in docs/scope_pr7.md § 3).
COMPARTMENTS: tuple[str, ...] = ("WT", "TC", "ET")

# Rows of Table 2: (json key, display label, format precision, format width-hint).
# format precision is the decimals shown in the median [Q1, Q3] cells; the
# upstream sidecars carry full float64 precision so changing this here is
# purely cosmetic.
METRICS: tuple[tuple[str, str, int], ...] = (
    ("dice", r"Dice", 3),
    ("hd95_mm", r"HD$_{95}$ (mm)", 2),
    ("abs_volumetric_error_mm3", r"$|\mathrm{VE}|$ (mm$^3$)", 0),
    ("sensitivity", r"Sensitivity", 3),
    ("specificity", r"Specificity", 3),
)
_METRIC_KEYS: tuple[str, ...] = tuple(k for k, _, _ in METRICS)
_METRIC_PRECISION: dict[str, int] = {k: p for k, _, p in METRICS}
_METRIC_LABEL: dict[str, str] = {k: lbl for k, lbl, _ in METRICS}

# Schema versions of the PR-7d sidecars we know how to read. Bumping
# requires coordinated updates here (and to docs/scope_pr7.md § 4 and
# scripts/segment_all_cohort50.py).
ACCEPTED_METRICS_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})


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


# ---------------------------------------------------------------------------
# Sidecar I/O + schema validation
# ---------------------------------------------------------------------------


def _metrics_path(metrics_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return metrics_root / f"sub-{sid}" / "metrics" / f"sub-{sid}_metrics_dl_vs_gt.json"


class SidecarError(ValueError):
    """Raised when a metrics sidecar is missing required structure."""


def _validate_sidecar(doc: Mapping[str, Any], *, source: Path | str = "<dict>") -> None:
    """Validate a parsed PR-7d ``metrics_dl_vs_gt.json`` document.

    Permissive about *extra* keys (forward-compatibility) but strict
    about the keys we actually consume: schema version must be on the
    accepted list, ``comparison`` must be ``"dl_vs_gt"``, and every
    compartment in :data:`COMPARTMENTS` must carry every metric in
    :data:`_METRIC_KEYS`.
    """
    if "schema_version" not in doc:
        raise SidecarError(f"{source}: missing 'schema_version'")
    sv = str(doc["schema_version"])
    if sv not in ACCEPTED_METRICS_SCHEMA_VERSIONS:
        raise SidecarError(
            f"{source}: schema_version {sv!r} not in {sorted(ACCEPTED_METRICS_SCHEMA_VERSIONS)}",
        )
    if doc.get("comparison") != "dl_vs_gt":
        raise SidecarError(
            f"{source}: comparison={doc.get('comparison')!r}, expected 'dl_vs_gt'",
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
    """Load and validate every per-subject metrics sidecar.

    Returns ``(records, problems)`` where ``records`` is the list of
    successfully loaded sidecars (already augmented with the canonical
    ``subject_id``) and ``problems`` is a list of
    ``(subject_id, reason)`` for subjects whose sidecar was missing,
    unreadable, or schema-invalid.

    If ``strict`` is True, the first problem is re-raised.
    """
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
        # Canonicalise subject_id (the sidecar's own field is authoritative
        # but PR-7d already normalises it; we still pin it here for safety).
        doc["subject_id"] = normalize_subject_id(doc.get("subject_id", sid))
        records.append(dict(doc))
    return records, problems


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _classify_value(v: Any) -> tuple[str, float | None]:  # noqa: PLR0911
    # The cyclomatic complexity here is tightly bounded by the small,
    # exhaustive set of JSON value types we accept; a flat dispatch
    # tree is much easier to audit than a dict-based dispatch.
    """Return ``(status, finite_value)`` for one raw metric cell.

    ``status`` is one of ``"ok"``, ``"missing"`` (was JSON ``null``,
    i.e. NaN in PR-7d), or ``"inf"`` (was the string ``"inf"`` in
    PR-7d). ``finite_value`` is the number when ``status == "ok"``,
    otherwise ``None``.
    """
    if v is None:
        return "missing", None
    if isinstance(v, str):
        if v == "inf":
            return "inf", None
        # Some downstream writers may emit '-inf' or 'nan' as strings.
        if v.lower() in {"-inf", "+inf", "nan", "-nan"}:
            return ("inf" if "inf" in v.lower() else "missing"), None
        # Anything else string is malformed; treat as missing for the
        # aggregation but flag it in the audit CSV.
        return "missing", None
    if isinstance(v, bool):
        return "missing", None
    if isinstance(v, (int, float)):
        x = float(v)
        if not np.isfinite(x):
            return ("inf" if np.isinf(x) else "missing"), None
        return "ok", x
    return "missing", None


def long_audit_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Long-format per-subject CSV: one row per (subject, compartment, metric)."""
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
                        "compartment": comp,
                        "metric": mk,
                        "value": val if val is not None else np.nan,
                        "status": status,
                        "raw": raw if not isinstance(raw, float) else float(raw),
                    },
                )
    df = pd.DataFrame(
        rows, columns=["subject_id", "compartment", "metric", "value", "status", "raw"]
    )
    return df.sort_values(["compartment", "metric", "subject_id"], kind="stable").reset_index(
        drop=True
    )


def summarise_per_cell(
    records: list[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Per-(compartment, metric) median / Q1 / Q3 / n_used / n_missing / n_inf.

    Returned shape::

        {
            "WT": {
                "dice": {"median": 0.89, "q1": 0.85, "q3": 0.93,
                         "n_used": 48, "n_missing": 0, "n_inf": 0},
                ...
            },
            "TC": {...},
            "ET": {...},
        }

    NaN-only cells (n_used == 0) report ``median``/``q1``/``q3`` as
    ``None`` so the JSON is valid.
    """
    out: dict[str, dict[str, dict[str, float | int | None]]] = {comp: {} for comp in COMPARTMENTS}
    for comp in COMPARTMENTS:
        for mk in _METRIC_KEYS:
            finite: list[float] = []
            n_missing = 0
            n_inf = 0
            for rec in records:
                status, val = _classify_value(rec["metrics"][comp][mk])
                if status == "ok" and val is not None:
                    finite.append(val)
                elif status == "inf":
                    n_inf += 1
                else:
                    n_missing += 1
            if finite:
                arr = np.asarray(finite, dtype=np.float64)
                cell = {
                    "median": float(np.median(arr)),
                    "q1": float(np.quantile(arr, 0.25)),
                    "q3": float(np.quantile(arr, 0.75)),
                    "n_used": int(arr.size),
                    "n_missing": int(n_missing),
                    "n_inf": int(n_inf),
                }
            else:
                cell = {
                    "median": None,
                    "q1": None,
                    "q3": None,
                    "n_used": 0,
                    "n_missing": int(n_missing),
                    "n_inf": int(n_inf),
                }
            out[comp][mk] = cell
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_cell(cell: Mapping[str, float | int | None], precision: int) -> str:
    """Render one (median, Q1, Q3) cell as ``median [q1, q3]``.

    Reports ``--`` when n_used == 0.
    """
    if cell.get("n_used", 0) == 0 or cell.get("median") is None:
        return r"--"
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(cell['median'])} [{fmt.format(cell['q1'])}, {fmt.format(cell['q3'])}]"


def _denominator_note(
    summary: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    n_total: int,
) -> str:
    """Emit a one-line ``n=...`` note iff any cell has n_used != n_total."""
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
    """Render Table 2 as a self-contained ``\\begin{tabular}`` block.

    The block contains no float wrapper / caption; those live in
    ``paper/main.tex`` so the manuscript can move the table
    around without regenerating the producer's output.
    """
    lines: list[str] = []
    lines.append(
        "% AUTO-GENERATED by scripts/build_table2_dl_vs_gt.py --- DO NOT EDIT BY HAND.",
    )
    lines.append(
        "% Per-subject sidecars: data/derivatives_cohort50/sub-XXXX/metrics/"
        "sub-XXXX_metrics_dl_vs_gt.json (PR-7d).",
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


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _print_summary_to_stdout(
    summary: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    *,
    n_total: int,
    n_used_cohort: int,
    n_problems: int,
) -> None:
    print()
    print("=" * 78)
    print(
        f"PR-7g  Table 2 (DL vs UCSF-PDGM reference)  "
        f"cohort_n={n_total}  loaded={n_used_cohort}  problems={n_problems}",
    )
    print("=" * 78)
    print(f"  {'metric':<28}{'WT':>14}{'TC':>14}{'ET':>14}")
    print("-" * 78)
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
    print("=" * 78)


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
                "comparison": "dl_vs_gt",
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

    # If any cell ended up with zero finite samples we treat it as a soft
    # failure: the table is still emitted (so reviewers see the structure
    # and the '--' placeholder), but the exit code flags that the table
    # is incomplete.
    any_empty_cell = any(
        summary[comp][mk]["n_used"] == 0 for comp in COMPARTMENTS for mk in _METRIC_KEYS
    )
    return 1 if any_empty_cell else 0


if __name__ == "__main__":
    sys.exit(main())
