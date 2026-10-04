"""Per-timepoint clinical summary for the LUMIERE Patient-048 cohort (read-only).

Cross-joins the cohort manifest at ``configs/lumiere_p048_timepoints.yaml``
with three public LUMIERE manifests and the on-disk stage at
``data/lumiere_p048/``:

* ``LUMIERE-Demographics_Pathology.csv``  -- subject-level demographics + OS
* ``LUMIERE-ExpertRating-v202211.csv``    -- per-timepoint RANO + rationale
* ``LUMIERE-MRinfo.csv``                  -- per-channel scanner / sequence

Two outputs:

1. **Drift check.** Every demographics field declared in the manifest YAML
   is verified against the corresponding column in the demographics CSV;
   every per-timepoint RANO + rationale string is verified against the
   ExpertRating CSV. Mismatches are reported as warnings (non-fatal) so
   that any future hand-edit of the manifest that drifts from the public
   source data is caught before it leaks into a figure caption.

2. **Per-timepoint clinical table.** One row per timepoint with the
   day-axis value, RANO label, scanner manufacturer + model per channel,
   and the count of staged files actually present under
   ``data/lumiere_p048/tp-<name>/``.

A one-line caption-ready summary is also printed -- this is the string we
expect to drop verbatim into the Fig. 7-analog (and Fig. 8-analog)
captions in ``paper/main.tex``.

The script is **read-only**: it never writes into the LUMIERE archive
nor into the staged tree. With ``--out-json`` it emits a single
machine-readable JSON sidecar that downstream Phase 2 producer code can
consume in place of re-parsing the CSVs.

Usage
-----

    uv run python scripts/summarize_lumiere_p048_clinical.py

    uv run python scripts/summarize_lumiere_p048_clinical.py \
        --lumiere-archive ~/Dropbox/Arvid/China_2025/LUMIERE \
        --stage-root ./data/lumiere_p048 \
        --out-json outputs/inventory/lumiere_p048_clinical_summary.json

The default ``--lumiere-archive`` is the author's local mirror under
``~/Dropbox/Arvid/China_2025/LUMIERE``. On any other machine, pass an
explicit ``--lumiere-archive`` pointing at the directory that contains
the three CSVs above (downloaded from
``https://github.com/ysuter/gbm-data-longitudinal``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_STAGE = REPO_ROOT / "data" / "lumiere_p048"
DEFAULT_ARCHIVE = Path("~/Dropbox/Arvid/China_2025/LUMIERE").expanduser()

MRINFO_NAME = "LUMIERE-MRinfo.csv"
EXPERT_RATING_NAME = "LUMIERE-ExpertRating-v202211.csv"
DEMOGRAPHICS_NAME = "LUMIERE-Demographics_Pathology.csv"

PATIENT_LABEL = "Patient-048"


def _norm(value: str | None) -> str:
    """Normalise a CSV cell for tolerant comparison (strip + lower)."""
    return (value or "").strip().lower()


def _load_csv_for_patient(path: Path, patient_label: str) -> list[dict[str, str]]:
    """Load a LUMIERE CSV and return rows whose ``Patient`` cell matches."""
    if not path.is_file():
        raise FileNotFoundError(f"LUMIERE CSV not found: {path}")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader if _norm(row.get("Patient")) == _norm(patient_label)]


def _check_demographics_drift(manifest_demo: dict, demo_rows: list[dict[str, str]]) -> list[str]:
    """Compare manifest demographics against the LUMIERE Demographics CSV row."""
    warnings: list[str] = []
    if not demo_rows:
        return [f"demographics: {PATIENT_LABEL} has no row in {DEMOGRAPHICS_NAME}"]
    if len(demo_rows) > 1:
        warnings.append(
            f"demographics: {PATIENT_LABEL} has {len(demo_rows)} rows in "
            f"{DEMOGRAPHICS_NAME} (expected 1); using the first."
        )
    row = demo_rows[0]

    # CSV uses 'female' / 'male'; manifest uses 'F' / 'M'. Compare first letter.
    csv_sex_initial = _norm(row.get("Sex"))[:1]
    if csv_sex_initial != _norm(str(manifest_demo.get("sex", "")))[:1]:
        warnings.append(
            f"demographics.sex: manifest={manifest_demo.get('sex')!r} CSV={row.get('Sex')!r}"
        )

    try:
        csv_age = int(float(row.get("Age at surgery (years)", "")))
        if csv_age != int(manifest_demo.get("age_at_surgery_years", -1)):
            warnings.append(
                f"demographics.age_at_surgery_years: "
                f"manifest={manifest_demo.get('age_at_surgery_years')} CSV={csv_age}"
            )
    except (TypeError, ValueError):
        raw = row.get("Age at surgery (years)")
        warnings.append(f"demographics.age_at_surgery_years: unparseable CSV value {raw!r}")

    # IDH: CSV cell is verbose ("IDH1 neg, Sequencing required"); manifest uses 'WT' / 'mutant'.
    csv_idh_lower = _norm(row.get("IDH (WT: wild type)"))
    manifest_idh = _norm(str(manifest_demo.get("idh", "")))
    csv_implies_wt = "neg" in csv_idh_lower or "wild" in csv_idh_lower or "wt" in csv_idh_lower
    if manifest_idh == "wt" and not csv_implies_wt:
        warnings.append(
            f"demographics.idh: manifest=WT but CSV={row.get('IDH (WT: wild type)')!r} "
            f"does not imply wild-type"
        )

    csv_mgmt_qual = _norm(row.get("MGMT qualitative"))
    if csv_mgmt_qual != _norm(str(manifest_demo.get("mgmt_qualitative", ""))):
        manifest_mgmt = manifest_demo.get("mgmt_qualitative")
        warnings.append(
            f"demographics.mgmt_qualitative: "
            f"manifest={manifest_mgmt!r} CSV={row.get('MGMT qualitative')!r}"
        )

    try:
        csv_mgmt_quant = float(_norm(row.get("MGMT quantitative")).rstrip("%"))
        manifest_mgmt_quant = float(manifest_demo.get("mgmt_quantitative_pct", -1))
        if abs(csv_mgmt_quant - manifest_mgmt_quant) > 1e-3:
            warnings.append(
                f"demographics.mgmt_quantitative_pct: "
                f"manifest={manifest_mgmt_quant} CSV={csv_mgmt_quant}"
            )
    except (TypeError, ValueError):
        raw = row.get("MGMT quantitative")
        warnings.append(f"demographics.mgmt_quantitative_pct: unparseable CSV value {raw!r}")

    try:
        csv_os = int(float(row.get("Survival time (weeks)", "")))
        if csv_os != int(manifest_demo.get("survival_weeks", -1)):
            warnings.append(
                f"demographics.survival_weeks: "
                f"manifest={manifest_demo.get('survival_weeks')} CSV={csv_os}"
            )
    except (TypeError, ValueError):
        raw = row.get("Survival time (weeks)")
        warnings.append(f"demographics.survival_weeks: unparseable CSV value {raw!r}")

    return warnings


def _rating_columns(row: dict[str, str]) -> tuple[str | None, str | None]:
    """ExpertRating CSV uses long, comma-bearing column titles; pull them by prefix."""
    rating, rationale = None, None
    for key, val in row.items():
        if key is None:
            continue
        key_lower = key.lower()
        if key_lower.startswith("rating ") and "rationale" not in key_lower:
            rating = val
        elif "rationale" in key_lower:
            rationale = val
        elif key_lower == "rating":
            rating = val
    return rating, rationale


def _check_rano_drift(manifest_tps: list[dict], rating_rows: list[dict[str, str]]) -> list[str]:
    """Compare manifest per-timepoint RANO + rationale against the ExpertRating CSV."""
    warnings: list[str] = []
    by_tp = {_norm(row.get("Date")): row for row in rating_rows}
    for tp in manifest_tps:
        name = tp["name"]
        row = by_tp.get(_norm(name))
        if row is None:
            warnings.append(f"rano[{name}]: no row in {EXPERT_RATING_NAME}")
            continue
        csv_rating, csv_rationale = _rating_columns(row)
        if _norm(csv_rating) != _norm(tp.get("rano", "")):
            warnings.append(f"rano[{name}].rano: manifest={tp.get('rano')!r} CSV={csv_rating!r}")
        if _norm(csv_rationale) != _norm(tp.get("rano_rationale", "")):
            warnings.append(
                f"rano[{name}].rano_rationale: "
                f"manifest={tp.get('rano_rationale')!r} CSV={csv_rationale!r}"
            )
    return warnings


def _scanner_per_channel(
    tp_name: str, mr_rows: list[dict[str, str]], channels: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Return {channel: {manufacturer, model, voxel_size}} for one timepoint."""
    out: dict[str, dict[str, str]] = {}
    for channel in channels:
        match = next(
            (
                r
                for r in mr_rows
                if _norm(r.get("Timepoint")) == _norm(tp_name)
                and _norm(r.get("Sequence")) == _norm(channel)
            ),
            None,
        )
        if match is None:
            out[channel] = {"manufacturer": "?", "model": "?", "voxel_size": "?"}
            continue
        out[channel] = {
            "manufacturer": (match.get("Manufacturer") or "?").strip(),
            "model": (match.get("Model") or "?").strip(),
            "voxel_size": (match.get("Voxel size") or "?").strip(),
        }
    return out


def _staged_files(stage_root: Path, tp_name: str) -> dict[str, list[str]]:
    """Inventory the staged tp-<name>/ folder. Returns {present, missing_channels}."""
    tp_dir = stage_root / f"tp-{tp_name}"
    if not tp_dir.is_dir():
        return {"present": [], "stage_dir": str(tp_dir), "exists": "no"}
    present = sorted(p.name for p in tp_dir.iterdir() if p.is_file())
    return {"present": present, "stage_dir": str(tp_dir), "exists": "yes"}


def _print_drift(warnings: list[str]) -> None:
    if not warnings:
        print("Drift check: PASS (manifest matches LUMIERE-Demographics + ExpertRating CSVs).")
        return
    print(f"Drift check: {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"  - {w}")


def _print_per_tp_table(
    manifest: dict,
    mr_rows: list[dict[str, str]],
    stage_root: Path,
) -> list[dict]:
    print("\nPer-timepoint clinical summary:")
    print("-" * 78)
    rows: list[dict] = []
    channels = manifest.get("channels", ["T1", "CT1", "T2", "FLAIR"])
    for tp in manifest["timepoints"]:
        name = tp["name"]
        scanners = _scanner_per_channel(name, mr_rows, channels)
        staged = _staged_files(stage_root, name)
        n_present = len(staged["present"])
        models = sorted({sc["model"] for sc in scanners.values() if sc["model"] != "?"})
        models_str = ", ".join(models) if models else "?"
        print(
            f"  day {tp['days_since_baseline']:>3}  {name:<11}  "
            f"RANO={tp['rano']:<8}  scanner=[{models_str}]  "
            f"staged={n_present} files"
        )
        if tp.get("rano_rationale"):
            print(f"            rationale: {tp['rano_rationale']}")
        rows.append(
            {
                "name": name,
                "days_since_baseline": tp["days_since_baseline"],
                "rano": tp["rano"],
                "rano_rationale": tp.get("rano_rationale", ""),
                "scanners": scanners,
                "staged": staged,
            }
        )
    return rows


def _caption_one_liner(manifest: dict) -> str:
    demo = manifest.get("demographics", {})
    tps = manifest.get("timepoints", [])
    rano_arrow = " -> ".join(tp["rano"] for tp in tps)
    return (
        f"Patient-{manifest.get('patient_id', '?')}: "
        f"{demo.get('sex', '?')}, {demo.get('age_at_surgery_years', '?')}y, "
        f"IDH-{demo.get('idh', '?')}, MGMT {demo.get('mgmt_qualitative', '?')}, "
        f"OS {demo.get('survival_weeks', '?')} wk; "
        f"{len(tps)} timepoints, RANO {rano_arrow}."
    )


def summarize(
    *,
    manifest_path: Path,
    archive_root: Path,
    stage_root: Path,
) -> dict:
    with manifest_path.open() as fh:
        manifest = yaml.safe_load(fh)

    demo_rows = _load_csv_for_patient(archive_root / DEMOGRAPHICS_NAME, PATIENT_LABEL)
    rating_rows = _load_csv_for_patient(archive_root / EXPERT_RATING_NAME, PATIENT_LABEL)
    mr_rows = _load_csv_for_patient(archive_root / MRINFO_NAME, PATIENT_LABEL)

    drift_warnings = _check_demographics_drift(manifest.get("demographics", {}), demo_rows)
    drift_warnings += _check_rano_drift(manifest["timepoints"], rating_rows)

    print("=" * 78)
    print(f"LUMIERE clinical summary  ({PATIENT_LABEL})")
    print("=" * 78)
    _print_drift(drift_warnings)
    rows = _print_per_tp_table(manifest, mr_rows, stage_root)

    one_liner = _caption_one_liner(manifest)
    print("\n" + "-" * 78)
    print("Caption one-liner (paste into Fig. 7-analog / Fig. 8-analog captions):")
    print("-" * 78)
    print(f"  {one_liner}")

    return {
        "schema_version": "1.0",
        "patient_label": PATIENT_LABEL,
        "manifest_path": str(manifest_path),
        "archive_root": str(archive_root),
        "stage_root": str(stage_root),
        "drift_warnings": drift_warnings,
        "caption_one_liner": one_liner,
        "demographics_csv": demo_rows[0] if demo_rows else None,
        "timepoints": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--lumiere-archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=(
            "Directory holding the three LUMIERE CSVs (Demographics, "
            f"ExpertRating, MRinfo). Default: {DEFAULT_ARCHIVE}"
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE,
        help=f"Local LUMIERE Patient-048 stage. Default: {DEFAULT_STAGE.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write a structured clinical-summary JSON sidecar.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the manifest drifts from the source CSVs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize(
        manifest_path=args.manifest.expanduser().resolve(),
        archive_root=args.lumiere_archive.expanduser().resolve(),
        stage_root=args.stage_root.expanduser().resolve(),
    )
    if args.out_json is not None:
        out = args.out_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\nClinical summary JSON written to: {out}")
    if args.strict and summary["drift_warnings"]:
        print("\n[strict] Drift detected; exiting non-zero.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
