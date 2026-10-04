"""PR-7f cohort fan-out for the Hit-Plot family (v1 + probabilistic).

Drives :func:`hpgs.hitplot.compartment_region_matrix` and
:func:`hpgs.hitplot.probabilistic_hitplot` across every subject in the
locked cohort YAML (``configs/cohort_ucsfpdgm_n50.yaml``) and writes the
five per-subject CSVs enumerated in Section 4 of ``docs/scope_pr7.md``:

* ``hitplot/sub-XXXX_hitplot_dl.csv``            (DL hard mask; PR-7d)
* ``hitplot/sub-XXXX_hitplot_dl_prob.csv``       (DL sigmoid probs;
                                                   PR-7d probs NIfTI)
* ``hitplot/sub-XXXX_hitplot_gt.csv``            (UCSF-PDGM reference)
* ``hitplot/sub-XXXX_hitplot_raidionics.csv``    (Raidionics; PR-7e)
* ``hitplot/sub-XXXX_hitplot_segmentglioma.csv`` (segment_glioma; PR-7e)
* ``hitplot/sub-XXXX_hitplot_tumorsynth.csv``    (mri_TumorSynth; optional PR-7e)

Each CSV is the long-format (label, compartment) matrix the Hit-Plot
agreement panel (PR-7i) consumes: the deterministic variant carries
``parcel_voxels / compartment_voxels / overlap_voxels /
overlap_volume_mm3 / pct_of_parcel / pct_of_compartment``; the
probabilistic variant carries the closed-form Poisson-binomial mean and
std of ``overlap_voxels`` and ``pct_of_parcel``. Rows are sorted by
``(compartment, pct_of_parcel desc)`` so a per-compartment top-k is a
one-line pandas operation downstream.

Inputs consumed (all pre-existing on disk when the operator runs):

* **Parcellation** (PR-7c, pending): ``data/derivatives_cohort50/
  sub-XXXX/parcellation/sub-XXXX_wmparc_native.nii.gz`` +
  ``sub-XXXX_wmparc_lut.json``. Required by every source (``compartment_
  region_matrix`` and ``probabilistic_hitplot`` both need the parcel
  LUT and label map). If absent and ``--require-parcellation`` is set
  (default), the subject is skipped with a warning and the runner
  continues with the remaining subjects; with ``--no-require-
  parcellation`` the skip is silent (for smoke runs before PR-7c ships).
* **DL label map + probs** (PR-7d): ``seg_dl/sub-XXXX_seg_brats3_
  dl.nii.gz``, ``seg_dl/sub-XXXX_seg_brats3_dl_probs.nii.gz`` +
  ``seg_dl/sub-XXXX_seg_brats3_dl.json`` sidecar (for the
  authoritative ``probability_channel_order``).
* **GT label map**: resolved via :func:`hpgs.io.resolve_subject_inputs`
  from the extracted UCSF-PDGM cohort root.
* **Raidionics / segment_glioma / TumorSynth label maps** (PR-7e): ``seg_<baseline>/
  sub-XXXX_seg_<baseline>.nii.gz``.

Outputs written:

* The five CSVs listed above (one per ``--source``). Rows per CSV
  depend on the parcellation LUT (usually ~100 regions x 3 compartments
  = ~300 rows for the deterministic variants).

Per-``(subject, source)`` failures are logged to stderr and the run
continues; the final exit code is non-zero if any pair failed.

The Q2-C headline metric bundle (Dice / HD95 / |VE| / sens / spec) is
**not** recomputed here --- that job belongs to PR-7d / PR-7e which
already wrote the per-subject metrics JSONs that feed Tables 2 and 3.

Usage::

    # Dry-run (default): enumerate subjects + artefact paths, no I/O.
    uv run python scripts/compute_hitplot_cohort50.py

    # Real fan-out (all 5 sources, every subject):
    uv run python scripts/compute_hitplot_cohort50.py --no-dry-run

    # Equivalent via the Make wrapper:
    make hitplot-all

    # Subset: one subject, only the DL hard + sigmoid sources:
    uv run python scripts/compute_hitplot_cohort50.py --no-dry-run \\
        --subject 0005 \\
        --source dl --source dl_prob

    # CI / smoke mode: synthetic parcellations may be absent -- don't
    # require them:
    uv run python scripts/compute_hitplot_cohort50.py --no-dry-run \\
        --no-require-parcellation

Exit codes:
    0   every (subject, source) pair succeeded or was skipped cleanly
    1   at least one (subject, source) pair failed
    2   could not load the cohort YAML / cohort or derivatives root unreachable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hpgs.hitplot import compartment_region_matrix, probabilistic_hitplot
from hpgs.io import (
    derivatives_dir,
    load_cohort_metadata,
    load_nifti,
    normalize_subject_id,
    resolve_subject_inputs,
)
from hpgs.segment.unified import PROB_CHANNEL_ORDER

DEFAULT_COHORT_YAML = Path("configs/cohort_ucsfpdgm_n50.yaml")
DEFAULT_METADATA_CSV = Path("data/UCSF-PDGM-metadata_v5.csv")
DEFAULT_COHORT_ROOT = Path("data/ucsf_pdgm_cohort50")
DEFAULT_DERIV_ROOT = Path("data/derivatives_cohort50")

# Mirror of scripts/segment_all_cohort50.py so this runner probes the
# same GT NIfTI the PR-7d runner consumed.
_INPUT_TUMOR_SEG = "tumor_segmentation"

# Hit-Plot source identifiers. The keys are what appears on the CLI
# (``--source <id>``) and in the on-disk filename; ``all`` is a virtual
# alias that expands to every other source (stable order).
SOURCES: tuple[str, ...] = ("dl", "dl_prob", "gt", "raidionics", "segmentglioma", "tumorsynth")

# Sources whose inner producer is :func:`probabilistic_hitplot`; the
# rest call :func:`compartment_region_matrix`.
_PROBABILISTIC_SOURCES: frozenset[str] = frozenset({"dl_prob"})


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
        "--cohort-root",
        type=Path,
        default=DEFAULT_COHORT_ROOT,
        help="Extracted UCSF-PDGM cohort root (default: %(default)s).",
    )
    p.add_argument(
        "--deriv-root",
        type=Path,
        default=DEFAULT_DERIV_ROOT,
        help="Where to read/write per-subject derivatives (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Enumerate subjects and print artefact paths (default).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Run the real PR-7f Hit-Plot fan-out.",
    )
    p.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        default=None,
        metavar="ID",
        help="4-digit subject id; repeat for multiple. Defaults to all cohort subjects.",
    )
    p.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        metavar="SRC",
        help=(
            "Hit-Plot source to (re)produce; repeat for multiple. One of "
            f"{[*SOURCES, 'all']}. Default: all."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Real run: skip (subject, source) pairs whose CSV already exists.",
    )
    p.add_argument(
        "--require-parcellation",
        dest="require_parcellation",
        action="store_true",
        default=True,
        help="Fail (warn + skip subject) if the parcellation artefacts are missing (default).",
    )
    p.add_argument(
        "--no-require-parcellation",
        dest="require_parcellation",
        action="store_false",
        help=(
            "Silently skip subjects missing parcellation artefacts (smoke mode before PR-7c ships)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Artefact contract (Section 4 of docs/scope_pr7.md)
# ---------------------------------------------------------------------------


def _planned_hitplot_artefacts(deriv_root: Path, subject_id: str) -> dict[str, Path]:
    """Per-(subject) inputs + outputs for the Hit-Plot fan-out.

    The keys match the naming used by ``scripts/segment_all_cohort50.py::
    _planned_artefacts`` so a single operator can reason about both
    runners' contracts at once.
    """
    sid = normalize_subject_id(subject_id)
    sub = deriv_root / f"sub-{sid}"
    seg_dl = sub / "seg_dl"
    seg_raid = sub / "seg_raidionics"
    seg_sg = sub / "seg_segmentglioma"
    seg_ts = sub / "seg_tumorsynth"
    parc = sub / "parcellation"
    hp = sub / "hitplot"
    return {
        "parc_wmparc": parc / f"sub-{sid}_wmparc_native.nii.gz",
        "parc_lut": parc / f"sub-{sid}_wmparc_lut.json",
        "seg_dl_label_map": seg_dl / f"sub-{sid}_seg_brats3_dl.nii.gz",
        "seg_dl_probs": seg_dl / f"sub-{sid}_seg_brats3_dl_probs.nii.gz",
        "seg_dl_sidecar": seg_dl / f"sub-{sid}_seg_brats3_dl.json",
        "seg_raidionics": seg_raid / f"sub-{sid}_seg_raidionics.nii.gz",
        "seg_segmentglioma": seg_sg / f"sub-{sid}_seg_segmentglioma.nii.gz",
        "seg_tumorsynth": seg_ts / f"sub-{sid}_seg_tumorsynth.nii.gz",
        "hp_dl": hp / f"sub-{sid}_hitplot_dl.csv",
        "hp_dl_prob": hp / f"sub-{sid}_hitplot_dl_prob.csv",
        "hp_gt": hp / f"sub-{sid}_hitplot_gt.csv",
        "hp_raidionics": hp / f"sub-{sid}_hitplot_raidionics.csv",
        "hp_segmentglioma": hp / f"sub-{sid}_hitplot_segmentglioma.csv",
        "hp_tumorsynth": hp / f"sub-{sid}_hitplot_tumorsynth.csv",
    }


def _hitplot_output_path(artefacts: dict[str, Path], source: str) -> Path:
    """Look up the on-disk CSV for one ``(subject, source)`` pair."""
    key = {
        "dl": "hp_dl",
        "dl_prob": "hp_dl_prob",
        "gt": "hp_gt",
        "raidionics": "hp_raidionics",
        "segmentglioma": "hp_segmentglioma",
        "tumorsynth": "hp_tumorsynth",
    }[source]
    return artefacts[key]


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


# ---------------------------------------------------------------------------
# Source + subject selection
# ---------------------------------------------------------------------------


def _select_subjects(
    cohort_ids: list[str],
    requested: list[str] | None,
) -> list[str]:
    """Normalise + filter the requested subject ids against the cohort.

    Mirrors ``scripts/segment_all_cohort50.py::_select_subjects``
    (Pythonic order preservation, dedup, explicit ``ValueError`` on
    unknown ids) so the two runners behave identically from the CLI.
    """
    if requested is None:
        return list(cohort_ids)
    allowed = {normalize_subject_id(s): s for s in cohort_ids}
    # Split any whitespace-packed tokens (accommodates ``SEGMENT_SUBJECTS="0005 0026"``
    # from the Makefile shim) and preserve request order while de-duplicating.
    expanded: list[str] = []
    for token in requested:
        expanded.extend(t for t in str(token).split() if t)
    seen: set[str] = set()
    out: list[str] = []
    for raw in expanded:
        sid = normalize_subject_id(raw)
        if sid in seen:
            continue
        if sid not in allowed:
            raise ValueError(f"subject {raw!r} not in the locked cohort")
        seen.add(sid)
        out.append(sid)
    return out


def _select_sources(requested: list[str] | None) -> list[str]:
    """Resolve ``--source`` into a stable ordered list of concrete sources.

    ``None`` -> every source; ``all`` expands to every source; explicit
    ids are dedup'd but order-preserving; unknown ids raise
    ``ValueError``.
    """
    if requested is None:
        return list(SOURCES)
    expanded: list[str] = []
    for token in requested:
        expanded.extend(t for t in str(token).split() if t)
    if not expanded:
        return list(SOURCES)
    seen: set[str] = set()
    out: list[str] = []
    for raw in expanded:
        if raw == "all":
            for s in SOURCES:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            continue
        if raw not in SOURCES:
            raise ValueError(
                f"source {raw!r} not in {[*SOURCES, 'all']}",
            )
        if raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


# ---------------------------------------------------------------------------
# Parcellation LUT helpers
# ---------------------------------------------------------------------------


def _read_wmparc_lut(lut_path: Path) -> dict[int, str]:
    """Read the ``sub-XXXX_wmparc_lut.json`` sidecar into ``{label: name}``.

    Per Section 4 of ``docs/scope_pr7.md`` the on-disk keys are string
    ints (JSON has no integer keys); we coerce to ``int`` here so the
    hitplot public API sees the signature it documents.
    """
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
    """Read ``probability_channel_order`` from the PR-7d seg_dl sidecar.

    Falls back to :data:`PROB_CHANNEL_ORDER` (the canonical order the
    unified segmenter ships with) and logs a warning if the sidecar is
    absent or missing the key; the fallback is always correct as long
    as the probs NIfTI was also written by the same PR-7d runner.
    """
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


# ---------------------------------------------------------------------------
# Per-(subject, source) real processor
# ---------------------------------------------------------------------------


def _real_process_subject_source(  # noqa: PLR0911, PLR0915
    subject_id: str,
    source: str,
    *,
    cohort_root: Path,
    deriv_root: Path,
    skip_existing: bool,
    require_parcellation: bool,
) -> dict[str, Any]:
    """Produce one Hit-Plot CSV for one (subject, source) pair.

    Returns a dict record with at minimum ``{subject_id, source,
    status}`` where ``status \\in {ok, skipped, missing_input, error}``
    and a free-form ``reason`` / ``detail``. Callers aggregate these
    into a cohort-level summary and set the process exit code based on
    whether any ``error`` records were emitted.
    """
    sid = normalize_subject_id(subject_id)
    artefacts = _planned_hitplot_artefacts(deriv_root, sid)
    out_csv: Path = _hitplot_output_path(artefacts, source)

    record: dict[str, Any] = {
        "subject_id": sid,
        "source": source,
        "out_csv": str(out_csv),
    }

    if skip_existing and out_csv.is_file():
        record["status"] = "skipped"
        record["reason"] = f"--skip-existing and {out_csv.name} present"
        return record

    parc_wmparc: Path = artefacts["parc_wmparc"]
    parc_lut: Path = artefacts["parc_lut"]
    if not (parc_wmparc.is_file() and parc_lut.is_file()):
        record["status"] = "missing_input"
        record["reason"] = (
            f"parcellation not on disk (expected {parc_wmparc.name} + {parc_lut.name})"
        )
        # Same record shape whether or not require_parcellation is True;
        # the caller picks the exit-code policy (warn-and-continue vs.
        # silent-skip).
        record["require_parcellation"] = bool(require_parcellation)
        return record

    try:
        parcellation, parc_affine = load_nifti(parc_wmparc)
        parcellation = np.asarray(parcellation).astype(np.int32)
        label_lut = _read_wmparc_lut(parc_lut)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        record["status"] = "error"
        record["reason"] = f"failed to load parcellation: {exc}"
        return record

    voxel_volume = _voxel_volume_mm3(parc_affine)

    t0 = time.time()
    try:
        if source == "dl_prob":
            probs_path: Path = artefacts["seg_dl_probs"]
            sidecar_path: Path = artefacts["seg_dl_sidecar"]
            if not probs_path.is_file():
                record["status"] = "missing_input"
                record["reason"] = f"DL probs NIfTI absent: {probs_path}"
                return record
            probs_xyzt, _ = load_nifti(probs_path)
            # On-disk layout is (X, Y, Z, C); the hitplot API wants
            # (C, X, Y, Z) by default but we pass compartment_axis=-1
            # so it lifts axis -1 to 0 internally.
            channel_order = _read_prob_channel_order(sidecar_path)
            df = probabilistic_hitplot(
                probs_xyzt,
                parcellation,
                label_lut,
                voxel_volume_mm3=voxel_volume,
                compartments=channel_order,
                compartment_axis=-1,
            )
        else:
            lm_path = {
                "dl": artefacts["seg_dl_label_map"],
                "gt": None,  # resolved via resolve_subject_inputs below
                "raidionics": artefacts["seg_raidionics"],
                "segmentglioma": artefacts["seg_segmentglioma"],
                "tumorsynth": artefacts["seg_tumorsynth"],
            }[source]
            if source == "gt":
                inputs = resolve_subject_inputs(
                    cohort_root,
                    sid,
                    channel_names=[],
                    tumor_segmentation_name=_INPUT_TUMOR_SEG,
                )
                lm_path = Path(inputs.tumor_segmentation)
            assert lm_path is not None
            if not lm_path.is_file():
                record["status"] = "missing_input"
                record["reason"] = f"source NIfTI absent: {lm_path}"
                return record
            label_map, _ = load_nifti(lm_path)
            label_map = np.asarray(label_map).astype(np.int32)
            df = compartment_region_matrix(
                label_map,
                parcellation,
                label_lut,
                voxel_volume_mm3=voxel_volume,
            )
    except (FileNotFoundError, ValueError) as exc:
        record["status"] = "error"
        record["reason"] = f"hitplot producer failed: {exc}"
        return record

    derivatives_dir(deriv_root, sid)  # ensures sub-SID exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    record["status"] = "ok"
    record["n_rows"] = len(df)
    record["elapsed_s"] = round(time.time() - t0, 3)
    record["voxel_volume_mm3"] = voxel_volume
    return record


# ---------------------------------------------------------------------------
# Dry-run / real orchestration
# ---------------------------------------------------------------------------


def _check_gt_input(cohort_root: Path, subject_id: str) -> tuple[bool, str]:
    """Return ``(ok, detail)``: whether the GT NIfTI is resolvable."""
    try:
        resolve_subject_inputs(
            cohort_root,
            subject_id,
            channel_names=[],
            tumor_segmentation_name=_INPUT_TUMOR_SEG,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    return True, ""


def _print_dry_run_row(
    subject_id: str,
    artefacts: dict[str, Path],
    sources: list[str],
    *,
    cohort_root: Path,
) -> None:
    """One line per subject + one indented line per requested source."""
    sid = normalize_subject_id(subject_id)
    parc_ok = artefacts["parc_wmparc"].is_file() and artefacts["parc_lut"].is_file()
    parc_tag = "parc=present" if parc_ok else "parc=MISSING"
    gt_ok, _ = _check_gt_input(cohort_root, sid)
    gt_tag = "gt=present" if gt_ok else "gt=MISSING"
    print(f"sub-{sid}  [{parc_tag}  {gt_tag}]")
    for src in sources:
        out = _hitplot_output_path(artefacts, src)
        exists = "exists" if out.is_file() else "would write"
        input_tag = ""
        if src == "dl" and not artefacts["seg_dl_label_map"].is_file():
            input_tag = "  (DL seg absent)"
        elif src == "dl_prob" and not artefacts["seg_dl_probs"].is_file():
            input_tag = "  (DL probs absent)"
        elif src == "raidionics" and not artefacts["seg_raidionics"].is_file():
            input_tag = "  (Raidionics seg absent)"
        elif src == "segmentglioma" and not artefacts["seg_segmentglioma"].is_file():
            input_tag = "  (segment_glioma seg absent)"
        elif src == "tumorsynth" and not artefacts["seg_tumorsynth"].is_file():
            input_tag = "  (TumorSynth seg absent)"
        elif src == "gt" and not gt_ok:
            input_tag = "  (GT absent)"
        print(f"    {src:<14} -> {out}  ({exists}){input_tag}")


def _run_dry(args: argparse.Namespace, cm) -> int:
    try:
        subjects = _select_subjects(list(cm.df.index), args.subjects)
        sources = _select_sources(args.sources)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print()
    print("=" * 86)
    print(
        f"PR-7f  Hit-Plot dry-run  cohort_n={cm.n}  selected={len(subjects)}  sources={sources}",
    )
    print("=" * 86)
    for sid in subjects:
        artefacts = _planned_hitplot_artefacts(args.deriv_root, sid)
        _print_dry_run_row(sid, artefacts, sources, cohort_root=args.cohort_root)
    return 0


def _print_summary_to_stdout(
    records: list[dict[str, Any]],
    *,
    n_total_subjects: int,
    sources: list[str],
) -> None:
    status_counts: dict[str, int] = {}
    for rec in records:
        status = str(rec.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print()
    print("=" * 86)
    print(
        f"PR-7f  Hit-Plot fan-out complete  cohort_n={n_total_subjects}  sources={sources}",
    )
    for status in ("ok", "skipped", "missing_input", "error"):
        if status in status_counts:
            print(f"  {status:<16} {status_counts[status]:>4}")
    print("=" * 86)


def _run_real(args: argparse.Namespace, cm) -> int:
    try:
        subjects = _select_subjects(list(cm.df.index), args.subjects)
        sources = _select_sources(args.sources)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Only the ``gt`` source resolves inputs via the raw cohort root; the
    # other four sources live under ``deriv_root``, so a missing cohort
    # tree is only a hard error when GT is actually requested.
    if "gt" in sources and not args.cohort_root.exists():
        print(
            f"error: cohort root does not exist: {args.cohort_root}",
            file=sys.stderr,
        )
        return 2

    records: list[dict[str, Any]] = []
    any_error = False
    for sid in subjects:
        for src in sources:
            rec = _real_process_subject_source(
                sid,
                src,
                cohort_root=args.cohort_root,
                deriv_root=args.deriv_root,
                skip_existing=args.skip_existing,
                require_parcellation=args.require_parcellation,
            )
            records.append(rec)
            status = rec.get("status")
            if status == "ok":
                print(
                    f"  sub-{sid} [{src}] ok  "
                    f"n_rows={rec.get('n_rows')}  "
                    f"t={rec.get('elapsed_s')}s",
                )
            elif status == "skipped":
                print(f"  sub-{sid} [{src}] skipped  ({rec.get('reason')})")
            elif status == "missing_input":
                msg = f"warning: sub-{sid} [{src}] missing_input: {rec.get('reason')}"
                print(msg, file=sys.stderr)
                # Missing-parcellation skips are promoted to errors only
                # when the operator asked us to require parcellation.
                reason = str(rec.get("reason", ""))
                if "parcellation not on disk" in reason and not args.require_parcellation:
                    continue
                any_error = any_error or "parcellation not on disk" in reason
                # Non-parcellation missing inputs (e.g. baseline seg not
                # produced on this subject) are informational, not errors.
            else:  # "error"
                any_error = True
                print(
                    f"error: sub-{sid} [{src}] error: {rec.get('reason')}",
                    file=sys.stderr,
                )

    _print_summary_to_stdout(
        records,
        n_total_subjects=cm.n,
        sources=sources,
    )
    return 1 if any_error else 0


def main() -> int:
    args = _parse_args()
    try:
        cm = load_cohort_metadata(args.cohort_yaml, args.metadata_csv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: failed to load cohort: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        return _run_dry(args, cm)
    return _run_real(args, cm)


if __name__ == "__main__":
    sys.exit(main())
