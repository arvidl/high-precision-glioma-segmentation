"""Join the n=50 leak-safe UCSF-PDGM cohort with the v5 metadata CSV.

The cohort YAML at ``configs/cohort_ucsfpdgm_n50.yaml`` is the *source of
truth* for which subjects are in the JMET revision study; the TCIA
``UCSF-PDGM-metadata_v5.csv`` is consulted *only* to attach per-subject
clinical metadata (Sex, Age, MGMT, OS, EOR, BraTS21 cohort, ...) to those
50 IDs.

The seed is locked to ``hpgs.SEED = 20260415`` and the cohort YAML must not
change without coordinated updates to the manuscript Methods.

Note on CSV row filtering: UCSF-PDGM v5 ships baseline exams with IDs of
the form ``UCSF-PDGM-NNN`` and a small number of follow-up exams with IDs
of the form ``UCSF-PDGM-NNN_FU<DDDd>`` (e.g. ``UCSF-PDGM-0429_FU003d``).
The cohort YAML only lists baseline IDs, so we filter the CSV down to
baseline rows before joining; otherwise the ID mapping would raise on the
follow-up rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_BASELINE_ID_RE = re.compile(r"^UCSF-PDGM-(\d+)$")


@dataclass(frozen=True)
class CohortMetadata:
    """Joined view of the n=50 cohort and the UCSF-PDGM v5 metadata CSV.

    Attributes
    ----------
    df:
        Per-subject DataFrame indexed by the four-digit ``subject_id``
        (e.g. ``"0005"``). Columns are the raw CSV columns plus
        ``OS_tertile`` (``"short" | "mid" | "long"``) computed from the cohort
        YAML cuts.
    cohort_yaml:
        Path to the cohort YAML used to select subjects.
    metadata_csv:
        Path to the metadata CSV used to attach metadata.
    seed:
        Seed of the cohort draw (must equal ``hpgs.SEED``).
    n:
        Cohort size (must equal ``len(df) == 50`` for the locked cohort).
    """

    df: pd.DataFrame
    cohort_yaml: Path
    metadata_csv: Path
    seed: int
    n: int


def _to_subject_id(raw: str) -> str:
    """Convert ``"UCSF-PDGM-005"`` (CSV form) to ``"0005"`` (YAML form).

    The UCSF-PDGM CSV uses zero-padded four-digit, hyphen-prefixed baseline
    IDs (e.g. ``UCSF-PDGM-0005``) while the cohort YAML and the on-disk
    ``sub-XXXX/`` layout use the same four-digit form without prefix
    (e.g. ``"0005"``). Raises ``ValueError`` if ``raw`` is not a baseline
    ID; callers should pre-filter follow-up exams (``..._FU<DDD>d``).
    """
    match = _BASELINE_ID_RE.match(raw.strip())
    if match is None:
        raise ValueError(f"Unexpected UCSF-PDGM id: {raw!r}")
    return match.group(1).zfill(4)


def _assign_os_tertile(os_days: float, cuts_days: tuple[float, float]) -> str:
    """Bin overall survival in days into the cohort YAML tertiles.

    ``cuts_days`` is ``(short_max_inclusive, mid_max_inclusive)``: a subject
    with ``os_days <= cuts[0]`` is ``"short"``; ``cuts[0] < os_days <= cuts[1]``
    is ``"mid"``; ``os_days > cuts[1]`` is ``"long"``.
    """
    if os_days <= cuts_days[0]:
        return "short"
    if os_days <= cuts_days[1]:
        return "mid"
    return "long"


def load_cohort_metadata(
    cohort_yaml: str | Path,
    metadata_csv: str | Path,
) -> CohortMetadata:
    """Load the cohort YAML and join it with the UCSF-PDGM v5 metadata CSV.

    Parameters
    ----------
    cohort_yaml:
        Path to ``configs/cohort_ucsfpdgm_n50.yaml`` (or another cohort YAML
        with the same schema).
    metadata_csv:
        Path to ``data/UCSF-PDGM-metadata_v5.csv``.

    Returns
    -------
    CohortMetadata
        A joined view; ``cm.df.index`` are zero-padded four-digit subject IDs
        and ``cm.df`` carries the CSV columns plus ``OS_tertile``.

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    KeyError
        If the cohort YAML lists a subject ID that is absent from the CSV
        (this would indicate cohort/CSV drift; abort loudly rather than
        silently dropping subjects).
    """
    cohort_yaml = Path(cohort_yaml)
    metadata_csv = Path(metadata_csv)
    if not cohort_yaml.is_file():
        raise FileNotFoundError(cohort_yaml)
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)

    with cohort_yaml.open() as f:
        cohort = yaml.safe_load(f)
    seed = int(cohort["seed"])
    n_expected = int(cohort["n"])
    cohort_ids = [str(s).zfill(4) for s in cohort["subjects"]]
    if len(cohort_ids) != n_expected:
        raise ValueError(
            f"Cohort YAML declares n={n_expected} but lists {len(cohort_ids)} subjects."
        )
    cuts = tuple(float(c) for c in cohort["stratification"]["os_tertile_cuts_days"])
    if len(cuts) != 2:
        raise ValueError(f"Expected exactly 2 OS-tertile cut points, got {cuts!r}")

    raw = pd.read_csv(metadata_csv)
    is_baseline = raw["ID"].astype(str).str.match(_BASELINE_ID_RE)
    raw = raw.loc[is_baseline].copy()
    raw["subject_id"] = raw["ID"].map(_to_subject_id)
    raw = raw.set_index("subject_id", drop=True)

    missing = sorted(set(cohort_ids) - set(raw.index))
    if missing:
        raise KeyError(
            f"Cohort YAML lists {len(missing)} subject(s) absent from "
            f"{metadata_csv.name}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    df = raw.loc[cohort_ids].copy()
    df["OS_tertile"] = (
        df["OS"]
        .astype(float)
        .map(
            lambda d: _assign_os_tertile(d, cuts)  # type: ignore[arg-type]
        )
    )

    return CohortMetadata(
        df=df,
        cohort_yaml=cohort_yaml,
        metadata_csv=metadata_csv,
        seed=seed,
        n=n_expected,
    )


def cohort_summary(cm: CohortMetadata) -> dict[str, dict[str, int | float]]:
    """Return a small dict summary suitable for Table 1 / printing.

    Counts are integers; medians are floats. Only fields the manuscript
    reports in Methods § Cohort are computed; the full per-subject DataFrame
    is available on ``cm.df`` for any other slicing.
    """
    df = cm.df
    sex_counts = df["Sex"].value_counts().to_dict()
    os_tertile_counts = df["OS_tertile"].value_counts().to_dict()
    mgmt_counts = df["MGMT status"].value_counts().to_dict()
    eor_counts = df["EOR"].value_counts().to_dict()
    vital_counts = df["1-dead 0-alive"].value_counts().to_dict()
    brats21_seg = df["BraTS21 Segmentation Cohort"].fillna("unused").value_counts().to_dict()

    return {
        "n": {"total": len(df)},
        "sex": {str(k): int(v) for k, v in sex_counts.items()},
        "age_years": {
            "median": float(df["Age at MRI"].median()),
            "mean": float(df["Age at MRI"].mean()),
            "min": float(df["Age at MRI"].min()),
            "max": float(df["Age at MRI"].max()),
        },
        "os_days": {
            "median": float(df["OS"].median()),
            "min": float(df["OS"].min()),
            "max": float(df["OS"].max()),
        },
        "os_tertile": {str(k): int(v) for k, v in os_tertile_counts.items()},
        "mgmt_status": {str(k): int(v) for k, v in mgmt_counts.items()},
        "eor": {str(k): int(v) for k, v in eor_counts.items()},
        "vital_dead": {str(k): int(v) for k, v in vital_counts.items()},
        "brats21_segmentation_cohort": {str(k): int(v) for k, v in brats21_seg.items()},
    }
