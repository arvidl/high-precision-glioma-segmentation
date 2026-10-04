"""Tests for the n=50 cohort / UCSF-PDGM v5 metadata join.

These tests guard the invariants that the manuscript Methods cohort
section hinges on:

* 50 subjects total;
* every subject ID in ``configs/cohort_ucsfpdgm_n50.yaml`` resolves in
  ``data/UCSF-PDGM-metadata_v5.csv``;
* zero subjects from the BraTS21 *Training* segmentation cohort (leak-safe);
* every subject is GBM IDH-wildtype WHO grade 4;
* OS tertile assignment respects the YAML cut points.

The tests are skipped if either input file is missing (which is the case
in the CI matrix until the metadata CSV is committed; the determinism /
draw test still runs on the YAML alone).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hpgs import SEED
from hpgs.io import cohort_summary, load_cohort_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
COHORT_YAML = REPO_ROOT / "configs" / "cohort_ucsfpdgm_n50.yaml"
METADATA_CSV = REPO_ROOT / "data" / "UCSF-PDGM-metadata_v5.csv"


@pytest.fixture(scope="module")
def cm():
    if not METADATA_CSV.is_file():
        pytest.skip(f"Metadata CSV not present: {METADATA_CSV}")
    if not COHORT_YAML.is_file():
        pytest.skip(f"Cohort YAML not present: {COHORT_YAML}")
    return load_cohort_metadata(COHORT_YAML, METADATA_CSV)


def test_cohort_size_is_50(cm) -> None:
    assert cm.n == 50
    assert len(cm.df) == 50
    assert cm.df.index.is_unique


def test_seed_matches_hpgs(cm) -> None:
    assert cm.seed == SEED == 20260415


def test_all_subjects_resolve_in_csv(cm) -> None:
    assert not cm.df["Sex"].isna().any(), (
        "Some cohort IDs returned NaN columns from the join — likely an ID "
        "format mismatch between the YAML (4-digit) and the CSV (UCSF-PDGM-NNN)."
    )


def test_no_brats21_training_subjects(cm) -> None:
    seg_cohort = cm.df["BraTS21 Segmentation Cohort"].fillna("unused").str.lower()
    n_training = int((seg_cohort == "training").sum())
    assert n_training == 0, (
        f"Cohort is supposed to be leak-safe wrt BraTS21 Training, but "
        f"found {n_training} BraTS21-Training subject(s). Check the cohort "
        f"draw and the manuscript Methods § Cohort wording."
    )


def test_all_gbm_idh_wildtype_grade4(cm) -> None:
    diag = cm.df["Final pathologic diagnosis (WHO 2021)"].str.lower()
    grades = cm.df["WHO CNS Grade"].astype(int)
    idh = cm.df["IDH"].str.lower()
    assert diag.str.contains("glioblastoma").all()
    assert idh.eq("wildtype").all()
    assert grades.eq(4).all()


def test_os_tertile_assignment_respects_yaml_cuts(cm) -> None:
    df = cm.df
    s = df.loc[df["OS_tertile"] == "short", "OS"].astype(float)
    m = df.loc[df["OS_tertile"] == "mid", "OS"].astype(float)
    long_ = df.loc[df["OS_tertile"] == "long", "OS"].astype(float)
    if len(s):
        assert s.max() <= 213.0
    if len(m):
        assert m.min() > 213.0 and m.max() <= 484.0
    if len(long_):
        assert long_.min() > 484.0


def test_summary_keys_are_sufficient_for_table1(cm) -> None:
    s = cohort_summary(cm)
    expected_keys = {
        "n",
        "sex",
        "age_years",
        "os_days",
        "os_tertile",
        "mgmt_status",
        "eor",
        "vital_dead",
        "brats21_segmentation_cohort",
    }
    assert expected_keys.issubset(s.keys())
    assert s["n"]["total"] == 50
