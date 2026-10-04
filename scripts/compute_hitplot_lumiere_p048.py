"""Compute per-timepoint Hit-Plot CSVs for LUMIERE Patient-048.

Longitudinal analogue of ``scripts/compute_hitplot_cohort50.py`` (PR-7f). Drives
:func:`hpgs.hitplot.compartment_region_matrix` and
:func:`hpgs.hitplot.probabilistic_hitplot` across the six Patient-048
timepoints, joining the round-3 Step G.2 FreeSurfer wmparc with the unified
DL segmentation (Step D) and the LUMIERE legacy comparators (HD-GLIO-AUTO and
DeepBraTumIA, both staged into the registered RAS frame by Step B).

Sources written (one CSV per ``(timepoint, source)`` pair):

* ``hitplot/tp-<week>_hitplot_dl.csv``        (unified BraTS-3 SegResNet hard mask)
* ``hitplot/tp-<week>_hitplot_dl_prob.csv``   (unified BraTS-3 SegResNet sigmoid probs)
* ``hitplot/tp-<week>_hitplot_hdglio.csv``    (LUMIERE HD-GLIO-AUTO 2-class mask)
* ``hitplot/tp-<week>_hitplot_dbt.csv``       (LUMIERE DeepBraTumIA-native 3-class mask)

There is no ``gt`` source: LUMIERE Patient-048 has no manual ground truth (the
two legacy comparators *are* the public reference); ``hdglio`` and ``dbt``
play the comparator role, and the unified DL engine is the headline producer.

Comparator label remapping (validated empirically by
``scripts/summarize_lumiere_p048_segmentation.py``)
---------------------------------------------------
The two comparator NIfTIs use their own label schemes; we recode each into
the BraTS-3 scheme (NCR=1, ED=2, ET=4, BG=0) before handing it to the same
``compartment_region_matrix`` API the cohort runner uses, so the output
CSV is directly comparable across sources.

* HD-GLIO-AUTO (2-class):
    1 = T2-hyperintense / non-enhancing core+edema  -> ED  (BraTS code 2)
    2 = contrast-enhancing tumour                   -> ET  (BraTS code 4)
  HD-GLIO does not separately resolve necrosis; its TC is meaningless, so
  we only emit the WT and ET rows (compartments=("WT", "ET")).
* DeepBraTumIA-native (3-class, ``ct1_seg_mask.nii.gz``):
    1 = enhancing tumour  -> ET   (BraTS code 4)
    2 = necrosis          -> NCR  (BraTS code 1)
    3 = edema             -> ED   (BraTS code 2)
  All three BraTS compartments are well-defined; emits ("WT", "TC", "ET").

Outputs (under ``--derivatives-root``, default ``data/lumiere_p048/derivatives``)
--------------------------------------------------------------------------------
::

    derivatives_root/
        hitplot_summary.csv        # one row per (tp, source) (Figure 11 input)
        hitplot_summary.json       # full provenance for the run
        tp-<week>/
            hitplot/
                tp-<week>_hitplot_dl.csv
                tp-<week>_hitplot_dl_prob.csv
                tp-<week>_hitplot_hdglio.csv     (when comparator NIfTI present)
                tp-<week>_hitplot_dbt.csv        (when comparator NIfTI present)

Each per-(tp, source) CSV is the long-format ``(label_id, region, compartment,
parcel_voxels, compartment_voxels, overlap_voxels, overlap_volume_mm3,
pct_of_parcel, pct_of_compartment)`` table (or its Poisson-binomial
probabilistic counterpart for ``dl_prob``) the cohort PR-7f computer
already documents in Section 4 of ``docs/scope_pr7.md``.

Usage
-----

    # Real run on every (tp, source) pair (default; no FreeSurfer required
    # at this step -- Step G.2 already produced the wmparc).
    uv run python scripts/compute_hitplot_lumiere_p048.py

    # Only the DL hard + sigmoid sources for one timepoint:
    uv run python scripts/compute_hitplot_lumiere_p048.py \\
        --timepoint week-013 --source dl --source dl_prob

    # Re-aggregate hitplot_summary.csv from existing per-tp CSVs without
    # recomputing them:
    uv run python scripts/compute_hitplot_lumiere_p048.py --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from hpgs.hitplot import compartment_region_matrix, probabilistic_hitplot
from hpgs.segment.unified import PROB_CHANNEL_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "lumiere_p048_timepoints.yaml"
DEFAULT_DERIVATIVES = REPO_ROOT / "data" / "lumiere_p048" / "derivatives"

SCHEMA_VERSION = "1.0"

# Source identifiers; appear as ``--source <id>`` on the CLI and in the
# on-disk filename. ``all`` is a virtual alias.
SOURCES: tuple[str, ...] = ("dl", "dl_prob", "hdglio", "dbt")
_PROBABILISTIC_SOURCES: frozenset[str] = frozenset({"dl_prob"})

# BraTS-3 codes (mirror of hpgs.segment.unified.LABEL_*).
_BRATS_NCR = 1
_BRATS_ED = 2
_BRATS_ET = 4

# Comparator label -> BraTS-3 code remappings. Each value is a {comparator
# label code -> BraTS code} dict; codes not in the dict become BG (0).
_COMPARATOR_REMAP: dict[str, dict[int, int]] = {
    "hdglio": {
        1: _BRATS_ED,  # HD-GLIO non-enhancing T2-hyperintense -> BraTS edema
        2: _BRATS_ET,  # HD-GLIO contrast-enhancing -> BraTS enhancing
    },
    "dbt": {
        1: _BRATS_ET,  # DeepBraTumIA enhancing -> BraTS enhancing
        2: _BRATS_NCR,  # DeepBraTumIA necrosis -> BraTS necrosis
        3: _BRATS_ED,  # DeepBraTumIA edema -> BraTS edema
    },
}

# Per-source BraTS-3 compartments to emit. HD-GLIO has no TC equivalent
# (does not resolve NCR), so we drop the TC row to avoid emitting a
# meaningless "TC == ET" duplicate.
_SOURCE_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "dl": ("WT", "TC", "ET"),
    "dl_prob": tuple(PROB_CHANNEL_ORDER),  # ("TC", "WT", "ET")
    "hdglio": ("WT", "ET"),
    "dbt": ("WT", "TC", "ET"),
}


# ---------------------------------------------------------------------------
# Manifest + path helpers.
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or "timepoints" not in manifest:
        raise ValueError(f"Invalid Patient-048 manifest: {path}")
    return manifest


def _tp_root(derivatives_root: Path, tp: str) -> Path:
    return derivatives_root / f"tp-{tp}"


def _hitplot_dir(derivatives_root: Path, tp: str) -> Path:
    return _tp_root(derivatives_root, tp) / "hitplot"


def _hitplot_csv_path(derivatives_root: Path, tp: str, source: str) -> Path:
    return _hitplot_dir(derivatives_root, tp) / f"tp-{tp}_hitplot_{source}.csv"


def _wmparc_paths(derivatives_root: Path, tp: str) -> tuple[Path, Path]:
    """Return ``(wmparc_native.nii.gz, wmparc_lut.json)``."""
    parc = _tp_root(derivatives_root, tp) / "parcellation"
    return (
        parc / f"tp-{tp}_wmparc_native.nii.gz",
        parc / f"tp-{tp}_wmparc_lut.json",
    )


def _seg_dl_paths(derivatives_root: Path, tp: str) -> dict[str, Path]:
    seg_dir = _tp_root(derivatives_root, tp) / "seg_dl"
    base = f"tp-{tp}_seg_brats3_dl"
    return {
        "label_map": seg_dir / f"{base}.nii.gz",
        "probs": seg_dir / f"{base}_probs.nii.gz",
        "sidecar": seg_dir / f"{base}.json",
    }


def _comparator_path(derivatives_root: Path, tp: str, source: str) -> Path:
    """Registered LUMIERE legacy comparator NIfTI for one (tp, source)."""
    suffix = {
        "hdglio": "hdglio_to_ref_segmentation",
        "dbt": "deepbratumia_to_ref_segmentation",
    }[source]
    return _tp_root(derivatives_root, tp) / f"tp-{tp}_{suffix}.nii.gz"


def _load_wmparc_lut(lut_path: Path) -> dict[int, str]:
    raw = json.loads(lut_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{lut_path}: expected top-level JSON object")
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            lab = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{lut_path}: non-integer label key {k!r}") from exc
        out[lab] = str(v)
    if not out:
        raise ValueError(f"{lut_path}: empty LUT")
    return out


def _read_prob_channel_order(sidecar_path: Path) -> tuple[str, ...]:
    """Pull ``probability_channel_order`` from the seg_dl sidecar (else fallback)."""
    if not sidecar_path.is_file():
        return tuple(PROB_CHANNEL_ORDER)
    try:
        doc = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError:
        return tuple(PROB_CHANNEL_ORDER)
    order = doc.get("probability_channel_order")
    if not (isinstance(order, list) and all(isinstance(x, str) for x in order)):
        return tuple(PROB_CHANNEL_ORDER)
    return tuple(order)


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


# ---------------------------------------------------------------------------
# Comparator -> BraTS-3 remapping.
# ---------------------------------------------------------------------------


def _remap_comparator_to_brats(label_map: np.ndarray, source: str) -> np.ndarray:
    """Recode a comparator label map into the BraTS-3 scheme.

    Codes not in :data:`_COMPARATOR_REMAP` become BG (0). The output is
    always int32 so it slots straight into ``compartment_region_matrix``.
    """
    if source not in _COMPARATOR_REMAP:
        raise ValueError(f"No comparator remap for source {source!r}")
    remap = _COMPARATOR_REMAP[source]
    out = np.zeros_like(label_map, dtype=np.int32)
    for src_code, brats_code in remap.items():
        out[label_map == src_code] = brats_code
    return out


# ---------------------------------------------------------------------------
# Per-(timepoint, source) processor.
# ---------------------------------------------------------------------------


def _compute_one(  # noqa: PLR0911
    *,
    timepoint: str,
    source: str,
    derivatives_root: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Produce one Hit-Plot CSV for one (timepoint, source) pair.

    Returns a record dict with at minimum ``{timepoint, source, status}``
    where ``status in {"ok", "skipped", "missing_input", "error"}``. Per-pair
    failures are reported, never raised; the caller decides exit-code policy.
    The early-return-per-failure shape mirrors
    ``compute_hitplot_cohort50.py::_real_process_subject_source`` (which
    carries the same ``noqa: PLR0911``).
    """
    out_csv = _hitplot_csv_path(derivatives_root, timepoint, source)
    record: dict[str, Any] = {
        "timepoint": timepoint,
        "source": source,
        "out_csv": str(out_csv),
    }

    if skip_existing and out_csv.is_file():
        record.update(status="skipped", reason=f"--skip-existing and {out_csv.name} present")
        return record

    wmparc_path, lut_path = _wmparc_paths(derivatives_root, timepoint)
    if not (wmparc_path.is_file() and lut_path.is_file()):
        record.update(
            status="missing_input",
            reason=(
                f"parcellation absent (expected {wmparc_path.name} "
                f"+ {lut_path.name}); run scripts/parcellate_lumiere_p048.py"
            ),
        )
        return record

    try:
        parc_img = nib.load(str(wmparc_path))
        parcellation = np.asarray(parc_img.dataobj).astype(np.int32)
        affine = np.asarray(parc_img.affine, dtype=np.float64)
        label_lut = _load_wmparc_lut(lut_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        record.update(status="error", reason=f"failed to load parcellation: {exc}")
        return record

    voxel_vol = _voxel_volume_mm3(affine)
    compartments = _SOURCE_COMPARTMENTS[source]
    t0 = time.time()

    try:
        if source == "dl_prob":
            probs_path = _seg_dl_paths(derivatives_root, timepoint)["probs"]
            sidecar_path = _seg_dl_paths(derivatives_root, timepoint)["sidecar"]
            if not probs_path.is_file():
                record.update(status="missing_input", reason=f"DL probs absent: {probs_path}")
                return record
            probs_img = nib.load(str(probs_path))
            probs_xyzt = np.asarray(probs_img.dataobj)
            if probs_xyzt.ndim != 4:
                record.update(
                    status="error",
                    reason=f"expected 4-D probs, got shape {probs_xyzt.shape}",
                )
                return record
            channel_order = _read_prob_channel_order(sidecar_path)
            df = probabilistic_hitplot(
                probs_xyzt,
                parcellation,
                label_lut,
                voxel_volume_mm3=voxel_vol,
                compartments=channel_order,
                compartment_axis=-1,
            )
        else:
            label_map = _load_source_hard_label_map(derivatives_root, timepoint, source, record)
            if label_map is None:
                return record
            df = compartment_region_matrix(
                label_map,
                parcellation,
                label_lut,
                voxel_volume_mm3=voxel_vol,
                compartments=compartments,
            )
    except (FileNotFoundError, ValueError) as exc:
        record.update(status="error", reason=f"hitplot producer failed: {exc}")
        return record

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    record.update(
        status="ok",
        n_rows=len(df),
        elapsed_s=round(time.time() - t0, 3),
        voxel_volume_mm3=voxel_vol,
        compartments=list(compartments),
    )
    return record


def _load_source_hard_label_map(
    derivatives_root: Path,
    timepoint: str,
    source: str,
    record: dict[str, Any],
) -> np.ndarray | None:
    """Load the hard label map for one source, BraTS-recoding comparators.

    Mutates ``record`` (writes ``status`` / ``reason``) and returns ``None``
    if the input NIfTI is absent. Returns the int32 BraTS-3 array on success.
    """
    if source == "dl":
        path = _seg_dl_paths(derivatives_root, timepoint)["label_map"]
        if not path.is_file():
            record.update(status="missing_input", reason=f"DL label map absent: {path}")
            return None
        return np.asarray(nib.load(str(path)).dataobj).astype(np.int32)

    if source not in _COMPARATOR_REMAP:
        record.update(status="error", reason=f"unknown source {source!r}")
        return None

    path = _comparator_path(derivatives_root, timepoint, source)
    if not path.is_file():
        record.update(
            status="missing_input",
            reason=f"comparator NIfTI absent: {path}",
        )
        return None
    raw = np.asarray(nib.load(str(path)).dataobj).astype(np.int32)
    return _remap_comparator_to_brats(raw, source)


# ---------------------------------------------------------------------------
# Cohort-level aggregation.
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS: tuple[str, ...] = (
    "timepoint",
    "days_since_baseline",
    "rano",
    "source",
    "status",
    "n_rows",
    "elapsed_s",
    "voxel_volume_mm3",
    "out_csv",
)


def _write_summary(
    *,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    derivatives_root: Path,
    sources: list[str],
) -> tuple[Path, Path]:
    csv_path = derivatives_root / "hitplot_summary.csv"
    json_path = derivatives_root / "hitplot_summary.json"

    by_tp = {tp_meta["name"]: tp_meta for tp_meta in manifest["timepoints"]}
    rows: list[dict[str, Any]] = []
    for r in records:
        tp = r["timepoint"]
        meta = by_tp.get(tp, {})
        rows.append(
            {
                "timepoint": tp,
                "days_since_baseline": meta.get("days_since_baseline"),
                "rano": meta.get("rano"),
                "source": r["source"],
                "status": r.get("status", "?"),
                "n_rows": r.get("n_rows", ""),
                "elapsed_s": r.get("elapsed_s", ""),
                "voxel_volume_mm3": r.get("voxel_volume_mm3", ""),
                "out_csv": r.get("out_csv", ""),
            }
        )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": manifest.get("cohort_id"),
        "patient_id": manifest.get("patient_id"),
        "manifest_path": str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        "derivatives_root": str(derivatives_root),
        "sources": sources,
        "comparator_label_remap": {
            k: {str(src): int(dst) for src, dst in v.items()} for k, v in _COMPARATOR_REMAP.items()
        },
        "source_compartments": {k: list(v) for k, v in _SOURCE_COMPARTMENTS.items()},
        "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Cohort manifest YAML. Default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES,
        help=(
            "Root containing the registered + segmented + parcellated tree. "
            f"Default: {DEFAULT_DERIVATIVES.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--timepoint",
        action="append",
        dest="timepoints",
        default=None,
        metavar="NAME",
        help="Only process this timepoint name (e.g. 'week-013'); repeat for several.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        choices=(*SOURCES, "all"),
        help="Hit-Plot source to compute; repeat for several. Default: all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (timepoint, source) pairs whose CSV already exists.",
    )
    return parser.parse_args(argv)


def _select_timepoints(
    manifest: dict[str, Any], requested: list[str] | None
) -> list[dict[str, Any]]:
    all_tps = manifest["timepoints"]
    if not requested:
        return all_tps
    by_name = {tp["name"]: tp for tp in all_tps}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(
            f"error: --timepoint values not in manifest: {missing}; available={list(by_name)}"
        )
    return [by_name[name] for name in requested]


def _select_sources(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(SOURCES)
    seen: set[str] = set()
    out: list[str] = []
    for raw in requested:
        if raw == "all":
            for s in SOURCES:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            continue
        if raw not in SOURCES:
            raise SystemExit(f"error: unknown --source {raw!r}; valid={list(SOURCES)}")
        if raw not in seen:
            seen.add(raw)
            out.append(raw)
    return out


def _print_table(records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 96)
    print("LUMIERE Patient-048 Hit-Plot computation report")
    print("=" * 96)
    print(f"{'timepoint':<14}{'source':<10}{'status':<14}" f"{'time(s)':>9}{'n_rows':>9}  reason")
    print("-" * 96)
    for r in records:
        tp = r["timepoint"]
        src = r["source"]
        status = r.get("status", "?")
        elapsed = r.get("elapsed_s", "-")
        n_rows = r.get("n_rows", "-")
        reason = r.get("reason", "")
        elapsed_str = f"{elapsed:>9.2f}" if isinstance(elapsed, int | float) else f"{elapsed:>9}"
        n_rows_str = f"{n_rows:>9}"
        print(f"{tp:<14}{src:<10}{status:<14}{elapsed_str}{n_rows_str}  {reason}")
    print("=" * 96)
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_skip = sum(1 for r in records if r.get("status") == "skipped")
    n_miss = sum(1 for r in records if r.get("status") == "missing_input")
    n_err = sum(1 for r in records if r.get("status") == "error")
    print(f"  ok={n_ok}  skipped={n_skip}  missing_input={n_miss}  error={n_err}")
    print("=" * 96)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    derivatives_root = args.derivatives_root.expanduser().resolve()

    if not manifest_path.is_file():
        print(f"[err] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not derivatives_root.is_dir():
        print(
            f"[err] derivatives-root not found: {derivatives_root}\n"
            "  Run scripts/register_lumiere_p048.py + segment_lumiere_p048.py + "
            "parcellate_lumiere_p048.py first.",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(manifest_path)
    timepoints = _select_timepoints(manifest, args.timepoints)
    sources = _select_sources(args.sources)

    print(
        f"LUMIERE Patient-048 Hit-Plot  manifest={manifest_path}\n"
        f"                              derivatives_root={derivatives_root}\n"
        f"                              n_timepoints={len(timepoints)}  sources={sources}  "
        f"skip_existing={args.skip_existing}"
    )
    print("-" * 96)

    records: list[dict[str, Any]] = []
    n_failed = 0

    for tp_meta in timepoints:
        tp = tp_meta["name"]
        for source in sources:
            print(f"[run] {tp:<14} {source:<10} ...", flush=True)
            try:
                rec = _compute_one(
                    timepoint=tp,
                    source=source,
                    derivatives_root=derivatives_root,
                    skip_existing=args.skip_existing,
                )
            except (RuntimeError, OSError) as exc:
                rec = {
                    "timepoint": tp,
                    "source": source,
                    "status": "error",
                    "reason": str(exc),
                    "out_csv": str(_hitplot_csv_path(derivatives_root, tp, source)),
                }
            if rec.get("status") == "error":
                n_failed += 1
            records.append(rec)

    csv_path, _json_path = _write_summary(
        manifest=manifest,
        records=records,
        derivatives_root=derivatives_root,
        sources=sources,
    )
    _print_table(records)
    csv_label = csv_path.relative_to(REPO_ROOT) if csv_path.is_relative_to(REPO_ROOT) else csv_path
    print(f"[ok] cohort summary: {csv_label}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
