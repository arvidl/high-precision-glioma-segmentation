"""PR-12 (R2.12): build Table 4 (per-pipeline-stage wall-clock runtime).

Aggregates the per-subject ``elapsed_s`` values that are already
captured in the existing PR-7c (parcellation) and PR-7d (DL
segmentation) sidecars and emits a self-contained ``\\begin{tabular}``
block ready to ``\\input{}`` from ``paper/main.tex``.

Stages currently surfaced (one row each):

* DL segmentation, first-call latency  (cold start; n=1, the earliest
  ``seg_dl`` sidecar by mtime --- the per-process MONAI Bundle
  weight-load + MPS context-init cost).
* DL segmentation, steady-state         (warm; n = N_seg - 1, all
  remaining ``seg_dl`` sidecars).
* Anatomical parcellation               (FreeSurfer ``recon-all-clinical``
  wmparc; n = number of parcellation sidecars present, may be a
  prefix of the cohort while a long fan-out is still in flight).

Per-stage cell format: ``median [Q1, Q3]`` (auto-formatted as
seconds, MM:SS, or HH:MM:SS depending on magnitude). The CSV/JSON
companions keep the raw float seconds so reviewers can recompute any
cell or replot.

("computational runtime, reproducibility, and software availability
should be expanded"). The host platform fingerprint is captured at
build time so the reader knows on which machine the cohort was
actually run.

Usage::

    # Whole cohort, default paths:
    uv run python scripts/bench_runtime.py

    # Or via the Make wrapper:
    make bench-runtime

    # Override paths / cohort:
    uv run python scripts/bench_runtime.py \\
        --cohort-yaml configs/cohort_ucsfpdgm_n50.yaml \\
        --derivatives-root data/derivatives_cohort50 \\
        --out-tex outputs/tables/table4_runtime.tex \\
        --out-csv outputs/tables/table4_runtime.csv \\
        --out-json outputs/tables/table4_runtime.json

Exit codes:
    0   table built and written; at least one stage has n_used > 0
    1   table built but every stage had n_used == 0 (warning)
    2   could not load the cohort YAML / derivatives root unreachable
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hpgs.io import load_cohort_metadata, normalize_subject_id

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_DERIVATIVES_ROOT = Path("data/derivatives_cohort50")
DEFAULT_OUT_TEX = Path("outputs/tables/table4_runtime.tex")
DEFAULT_OUT_CSV = Path("outputs/tables/table4_runtime.csv")
DEFAULT_OUT_JSON = Path("outputs/tables/table4_runtime.json")

# Schema versions of the upstream sidecars we know how to read. Bumping
# requires coordinated updates here (and to docs/scope_pr7.md § 4 and
# scripts/segment_all_cohort50.py / scripts/parcellate_all_cohort50.py).
ACCEPTED_SEG_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})
ACCEPTED_PARC_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingRecord:
    """One per-subject timing observation pulled from a sidecar."""

    subject_id: str
    stage: str  # "seg" | "parc"
    elapsed_s: float
    backend: str
    device: str
    extra: Mapping[str, Any] = field(default_factory=dict)
    sidecar: Path | None = None
    mtime: float | None = None  # used to detect cold start


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def _seg_sidecar(deriv_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return deriv_root / f"sub-{sid}" / "seg_dl" / f"sub-{sid}_seg_brats3_dl.json"


def _parc_sidecar(deriv_root: Path, subject_id: str) -> Path:
    sid = normalize_subject_id(subject_id)
    return deriv_root / f"sub-{sid}" / "parcellation" / f"sub-{sid}_wmparc.json"


def _coerce_elapsed(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        x = float(v)
        return x if np.isfinite(x) and x >= 0 else None
    return None


def load_seg_records(
    deriv_root: Path,
    subject_ids: Iterable[str],
) -> tuple[list[TimingRecord], list[tuple[str, str]]]:
    """Walk PR-7d ``seg_dl/sub-XXXX_seg_brats3_dl.json`` sidecars."""
    records: list[TimingRecord] = []
    problems: list[tuple[str, str]] = []
    for raw_sid in subject_ids:
        sid = normalize_subject_id(raw_sid)
        path = _seg_sidecar(deriv_root, sid)
        if not path.is_file():
            problems.append((sid, f"seg sidecar not found at {path}"))
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append((sid, f"could not parse seg JSON ({exc})"))
            continue
        sv = str(doc.get("schema_version", "?"))
        if sv not in ACCEPTED_SEG_SCHEMA_VERSIONS:
            problems.append((sid, f"seg schema_version {sv!r} not accepted"))
            continue
        elapsed = _coerce_elapsed(doc.get("elapsed_s"))
        if elapsed is None:
            problems.append((sid, "seg sidecar missing finite 'elapsed_s'"))
            continue
        records.append(
            TimingRecord(
                subject_id=sid,
                stage="seg",
                elapsed_s=elapsed,
                backend=str(doc.get("backend", "?")),
                device=str(doc.get("device", "?")),
                extra={
                    "bundle_version": doc.get("bundle_version"),
                    "channel_order": doc.get("channel_order"),
                },
                sidecar=path,
                mtime=path.stat().st_mtime,
            ),
        )
    return records, problems


def load_parc_records(
    deriv_root: Path,
    subject_ids: Iterable[str],
) -> tuple[list[TimingRecord], list[tuple[str, str]]]:
    """Walk PR-7c ``parcellation/sub-XXXX_wmparc.json`` sidecars."""
    records: list[TimingRecord] = []
    problems: list[tuple[str, str]] = []
    for raw_sid in subject_ids:
        sid = normalize_subject_id(raw_sid)
        path = _parc_sidecar(deriv_root, sid)
        if not path.is_file():
            problems.append((sid, f"parc sidecar not found at {path}"))
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append((sid, f"could not parse parc JSON ({exc})"))
            continue
        sv = str(doc.get("schema_version", "?"))
        if sv not in ACCEPTED_PARC_SCHEMA_VERSIONS:
            problems.append((sid, f"parc schema_version {sv!r} not accepted"))
            continue
        elapsed = _coerce_elapsed(doc.get("elapsed_s"))
        if elapsed is None:
            problems.append((sid, "parc sidecar missing finite 'elapsed_s'"))
            continue
        threads = doc.get("threads")
        device_str = f"CPU/{threads}t" if isinstance(threads, int) else "CPU"
        records.append(
            TimingRecord(
                subject_id=sid,
                stage="parc",
                elapsed_s=elapsed,
                backend=str(doc.get("backend", "?")),
                device=device_str,
                extra={
                    "version": doc.get("version"),
                    "input_channel": doc.get("input_channel"),
                    "threads": threads,
                },
                sidecar=path,
                mtime=path.stat().st_mtime,
            ),
        )
    return records, problems


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class StageSummary:
    """Cohort-level summary for one Table 4 row."""

    stage_label: str
    backend_label: str
    device_label: str
    n_used: int
    n_expected: int
    elapsed_s: list[float] = field(default_factory=list)

    @property
    def median(self) -> float | None:
        return float(np.median(self.elapsed_s)) if self.elapsed_s else None

    @property
    def q1(self) -> float | None:
        return float(np.quantile(self.elapsed_s, 0.25)) if self.elapsed_s else None

    @property
    def q3(self) -> float | None:
        return float(np.quantile(self.elapsed_s, 0.75)) if self.elapsed_s else None

    @property
    def vmin(self) -> float | None:
        return float(np.min(self.elapsed_s)) if self.elapsed_s else None

    @property
    def vmax(self) -> float | None:
        return float(np.max(self.elapsed_s)) if self.elapsed_s else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_label": self.stage_label,
            "backend_label": self.backend_label,
            "device_label": self.device_label,
            "n_used": int(self.n_used),
            "n_expected": int(self.n_expected),
            "elapsed_s": {
                "median": self.median,
                "q1": self.q1,
                "q3": self.q3,
                "min": self.vmin,
                "max": self.vmax,
            },
            "subjects": [],  # filled in main()
        }


def split_seg_cold_warm(
    seg_records: list[TimingRecord],
) -> tuple[TimingRecord | None, list[TimingRecord]]:
    """Split DL segmentation records into (cold, warm).

    ``cold`` is the earliest-mtime record (per-process model-load +
    MPS/CUDA context-init cost paid once per cohort run); ``warm`` is
    everything else, in stable mtime order.

    If the input is empty, returns ``(None, [])``. If only one record
    exists, that single record is returned as ``cold`` with empty warm.
    """
    if not seg_records:
        return None, []
    ordered = sorted(seg_records, key=lambda r: (r.mtime if r.mtime is not None else 0.0))
    return ordered[0], ordered[1:]


def summarise_stages(
    seg_records: list[TimingRecord],
    parc_records: list[TimingRecord],
    *,
    n_total: int,
) -> list[StageSummary]:
    """Build the three Table 4 rows."""
    cold, warm = split_seg_cold_warm(seg_records)

    seg_backend_label = "MONAI Bundle 0.4.8 (BraTS-3 SegResNet)"
    seg_device_label = "MPS"
    if seg_records:
        bundle_versions = {
            r.extra.get("bundle_version") for r in seg_records if r.extra.get("bundle_version")
        }
        if len(bundle_versions) == 1:
            seg_backend_label = f"MONAI Bundle {next(iter(bundle_versions))} (BraTS-3 SegResNet)"
        devices = {r.device for r in seg_records}
        if len(devices) == 1:
            seg_device_label = next(iter(devices)).upper()

    parc_backend_label = "FreeSurfer recon-all-clinical (wmparc)"
    parc_device_label = "CPU"
    if parc_records:
        versions = {r.extra.get("version") for r in parc_records if r.extra.get("version")}
        if len(versions) == 1:
            v = next(iter(versions)) or ""
            # Pull the user-friendly major.minor[.patch] version out of
            # the long FreeSurfer build banner, e.g.
            # "freesurfer-macOS-darwin_arm64-8.2.0-20260314-d932c45"
            # -> "8.2.0".
            m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", v)
            if m:
                parc_backend_label = f"FreeSurfer {m.group(1)} recon-all-clinical (wmparc)"
        devices = {r.device for r in parc_records}
        if len(devices) == 1:
            parc_device_label = next(iter(devices))

    summaries: list[StageSummary] = []

    # Row 1 --- DL cold-start (n = 0 or 1)
    cold_summary = StageSummary(
        stage_label=r"DL segmentation (first-call latency)",
        backend_label=seg_backend_label,
        device_label=seg_device_label,
        n_used=1 if cold is not None else 0,
        n_expected=1,
        elapsed_s=[cold.elapsed_s] if cold is not None else [],
    )
    summaries.append(cold_summary)

    # Row 2 --- DL steady state (n = N_seg - 1)
    warm_summary = StageSummary(
        stage_label=r"DL segmentation (steady-state)",
        backend_label=seg_backend_label,
        device_label=seg_device_label,
        n_used=len(warm),
        n_expected=max(n_total - 1, 0),
        elapsed_s=[r.elapsed_s for r in warm],
    )
    summaries.append(warm_summary)

    # Row 3 --- FreeSurfer parcellation (n = N_parc, may be partial)
    parc_summary = StageSummary(
        stage_label=r"Anatomical parcellation",
        backend_label=parc_backend_label,
        device_label=parc_device_label,
        n_used=len(parc_records),
        n_expected=n_total,
        elapsed_s=[r.elapsed_s for r in parc_records],
    )
    summaries.append(parc_summary)

    return summaries


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _fmt_duration(seconds: float | None) -> str:
    """Smart-format a wall-clock duration for the LaTeX table cell.

    < 60 s          -> ``1.29 s``     (2 decimals)
    60 s - 3600 s   -> ``34:20``       (MM:SS, integer seconds)
    >= 3600 s       -> ``1:02:30``     (HH:MM:SS, integer seconds)
    """
    if seconds is None:
        return "--"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.2f}\\,s"
    total = round(seconds)
    if total < _SECONDS_PER_HOUR:
        m, s = divmod(total, _SECONDS_PER_MINUTE)
        return f"{m:d}:{s:02d}\\,min"
    h, rest = divmod(total, _SECONDS_PER_HOUR)
    m, s = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{h:d}:{m:02d}:{s:02d}\\,hr"


def _fmt_cell(s: StageSummary) -> str:
    """``median [Q1, Q3]`` cell, or ``single value`` for n=1, or ``--``."""
    if s.n_used == 0:
        return "--"
    if s.n_used == 1:
        return _fmt_duration(s.median)
    return f"{_fmt_duration(s.median)} [{_fmt_duration(s.q1)}, {_fmt_duration(s.q3)}]"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _platform_fingerprint() -> dict[str, str]:
    """Best-effort host fingerprint for the LaTeX comment header + JSON."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def format_table_tex(
    summaries: list[StageSummary],
    *,
    n_total: int,
    fingerprint: Mapping[str, str],
) -> str:
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/bench_runtime.py --- DO NOT EDIT BY HAND.")
    lines.append(
        "% Per-subject sidecars: data/derivatives_cohort50/sub-XXXX/"
        "{seg_dl/*_seg_brats3_dl.json (PR-7d), parcellation/*_wmparc.json (PR-7c)}.",
    )
    lines.append(
        f"% Cohort: configs/cohort_ucsfpdgm_n50.yaml, n_total={n_total}; "
        f"per-stage n_used given in the table itself.",
    )
    lines.append(
        f"% Host: {fingerprint.get('platform','?')}; "
        f"machine={fingerprint.get('machine','?')}; "
        f"python={fingerprint.get('python','?')}.",
    )
    # tabularx + \small lets the three long-text columns (Stage, Backend,
    # Wall-clock) wrap inside \textwidth instead of overflowing the right
    # margin (Stage + Backend alone are ~57pt wider than \textwidth at 11pt).
    # Device and n are short and stay as fixed-width l/c. The producer wraps
    # itself in a group so \small does not leak past the table.
    # cf. main.tex preamble: \usepackage{tabularx}.
    lines.append(r"\begingroup\small")
    lines.append(
        r"\begin{tabularx}{\textwidth}{"
        r">{\raggedright\arraybackslash}X "
        r">{\raggedright\arraybackslash}X "
        r"l c "
        r">{\raggedright\arraybackslash}X}",
    )
    lines.append(r"\toprule")
    lines.append(
        r"Stage & Backend & Device & $n$ & Wall-clock per subject " r"(median [Q1, Q3]) \\",
    )
    lines.append(r"\midrule")
    for s in summaries:
        n_cell = f"{s.n_used}/{s.n_expected}" if s.n_used != s.n_expected else f"{s.n_used}"
        lines.append(
            f"{s.stage_label} & {s.backend_label} & {s.device_label} "
            f"& {n_cell} & {_fmt_cell(s)} \\\\",
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def write_csv(records: list[TimingRecord], path: Path) -> None:
    """Long-format per-subject audit CSV (one row per timing observation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(
            ["subject_id", "stage", "elapsed_s", "backend", "device", "sidecar", "mtime"],
        )
        for r in sorted(records, key=lambda x: (x.stage, x.subject_id)):
            writer.writerow(
                [
                    r.subject_id,
                    r.stage,
                    f"{r.elapsed_s:.6f}",
                    r.backend,
                    r.device,
                    str(r.sidecar) if r.sidecar else "",
                    f"{r.mtime:.3f}" if r.mtime is not None else "",
                ],
            )


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-yaml", type=Path, default=DEFAULT_COHORT_YAML)
    p.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    p.add_argument("--derivatives-root", type=Path, default=DEFAULT_DERIVATIVES_ROOT)
    p.add_argument("--out-tex", type=Path, default=DEFAULT_OUT_TEX)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return p.parse_args()


def _fmt_cell_plain(s: StageSummary) -> str:
    """ASCII version of :func:`_fmt_cell` for stdout (no LaTeX ``\\,``)."""
    if s.n_used == 0:
        return "--"
    if s.n_used == 1:
        return _fmt_duration(s.median).replace("\\,", " ")
    median = _fmt_duration(s.median).replace("\\,", " ")
    q1 = _fmt_duration(s.q1).replace("\\,", " ")
    q3 = _fmt_duration(s.q3).replace("\\,", " ")
    return f"{median} [{q1}, {q3}]"


def _print_summary(summaries: list[StageSummary], *, n_total: int) -> None:
    print()
    print("=" * 92)
    print(f"PR-12  Table 4 (per-pipeline-stage wall-clock)  cohort_n={n_total}")
    print("=" * 92)
    print(f"  {'stage':<42}{'n':>6}  {'cell':<40}")
    print("-" * 92)
    for s in summaries:
        n_cell = f"{s.n_used}/{s.n_expected}"
        cell = _fmt_cell_plain(s)
        print(f"  {s.stage_label:<42}{n_cell:>6}  {cell:<40}")
    print("=" * 92)


def main() -> int:
    args = _parse_args()
    try:
        cm = load_cohort_metadata(args.cohort_yaml, args.metadata_csv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: failed to load cohort: {exc}", file=sys.stderr)
        return 2
    if not args.derivatives_root.exists():
        print(
            f"error: derivatives root does not exist: {args.derivatives_root}",
            file=sys.stderr,
        )
        return 2

    subject_ids = list(cm.df.index)
    seg_records, seg_problems = load_seg_records(args.derivatives_root, subject_ids)
    parc_records, parc_problems = load_parc_records(args.derivatives_root, subject_ids)

    fingerprint = _platform_fingerprint()
    summaries = summarise_stages(seg_records, parc_records, n_total=cm.n)

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(
        format_table_tex(summaries, n_total=cm.n, fingerprint=fingerprint),
    )

    write_csv(seg_records + parc_records, args.out_csv)

    out_json = {
        "schema_version": "1.0",
        "cohort_yaml": str(args.cohort_yaml),
        "derivatives_root": str(args.derivatives_root),
        "n_total": int(cm.n),
        "n_seg_with_sidecar": len(seg_records),
        "n_parc_with_sidecar": len(parc_records),
        "n_seg_problems": len(seg_problems),
        "n_parc_problems": len(parc_problems),
        "host": fingerprint,
        "stages": [s.as_dict() for s in summaries],
        "subjects": {
            "seg": [
                {
                    "subject_id": r.subject_id,
                    "elapsed_s": r.elapsed_s,
                    "device": r.device,
                    "mtime": r.mtime,
                }
                for r in sorted(seg_records, key=lambda x: (x.mtime or 0.0))
            ],
            "parc": [
                {
                    "subject_id": r.subject_id,
                    "elapsed_s": r.elapsed_s,
                    "device": r.device,
                    "mtime": r.mtime,
                }
                for r in sorted(parc_records, key=lambda x: (x.mtime or 0.0))
            ],
        },
        "problems": {
            "seg": [{"subject_id": sid, "reason": r} for sid, r in seg_problems],
            "parc": [{"subject_id": sid, "reason": r} for sid, r in parc_problems],
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_json, indent=2))

    _print_summary(summaries, n_total=cm.n)
    print(f"  host: {fingerprint['platform']}")
    print(
        f"  seg: loaded={len(seg_records)}/{cm.n} (problems={len(seg_problems)});  "
        f"parc: loaded={len(parc_records)}/{cm.n} (problems={len(parc_problems)})",
    )
    print(f"  wrote: {args.out_tex}")
    print(f"  wrote: {args.out_csv}")
    print(f"  wrote: {args.out_json}")

    any_used = any(s.n_used > 0 for s in summaries)
    return 0 if any_used else 1


if __name__ == "__main__":
    sys.exit(main())
