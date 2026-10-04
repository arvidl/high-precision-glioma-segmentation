"""Unit tests for ``hpgs.parcellate.lobes`` (PR-7l).

The lobe map is the single source of truth shared by
``scripts/build_lobe_compartment_heatmap.py`` and any downstream
sunburst / heatmap successor to Figure 15. These tests pin the
contract:

* every Desikan-Killiany 2006 stem (the 34 cortical regions per
  hemisphere FreeSurfer 8.2.0 ships in ``aparc.annot``) maps to a
  recognised lobe;
* every wmparc sub-cortical structure that appears in our cohort's
  parcellation LUT maps to one of the four sub-cortical / infratentorial
  buckets;
* hemisphere prefix stripping is symmetric for the four wmparc cortical
  prefixes (``ctx-{lh,rh}-`` and ``wm-{lh,rh}-``) and the two sub-
  cortical prefixes (``Left-`` / ``Right-``);
* unknown / blank names fall through to ``"other"`` rather than raising
  -- a property the producer relies on so a rogue LUT entry does not
  abort a whole-cohort run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpgs.parcellate.lobes import (
    LOBE_OF_CORTICAL_STEM,
    LOBE_OF_SUBCORTICAL_NAME,
    LOBES,
    assign_lobe,
    assign_lobe_from_label,
    strip_hemisphere_prefix,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_lobes_tuple_is_unique_and_anatomically_ordered() -> None:
    assert len(LOBES) == len(set(LOBES))
    # Frontal must appear before occipital (rostro-caudal).
    assert LOBES.index("frontal") < LOBES.index("occipital")
    # "other" is the catch-all and must come last so heatmap rows
    # render with the catch-all bucket at the bottom.
    assert LOBES[-1] == "other"


def test_every_cortical_value_is_in_lobes_tuple() -> None:
    for stem, lobe in LOBE_OF_CORTICAL_STEM.items():
        assert lobe in LOBES, f"{stem!r} maps to unknown lobe {lobe!r}"


def test_every_subcortical_value_is_in_lobes_tuple() -> None:
    for stem, lobe in LOBE_OF_SUBCORTICAL_NAME.items():
        assert lobe in LOBES, f"{stem!r} maps to unknown lobe {lobe!r}"


def test_cortical_stem_count_matches_desikan_killiany() -> None:
    # FreeSurfer's aparc.annot ships 34 cortical labels per hemisphere
    # (Desikan-Killiany 2006). The bin must cover all of them.
    assert len(LOBE_OF_CORTICAL_STEM) == 34


# ---------------------------------------------------------------------------
# strip_hemisphere_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_stem", "expected_hemi"),
    [
        ("ctx-lh-superiorfrontal", "superiorfrontal", "L"),
        ("ctx-rh-superiorfrontal", "superiorfrontal", "R"),
        ("wm-lh-precentral", "precentral", "L"),
        ("wm-rh-precentral", "precentral", "R"),
        ("Left-Hippocampus", "Hippocampus", "L"),
        ("Right-Thalamus", "Thalamus", "R"),
        ("Brain-Stem", "Brain-Stem", None),
        ("CC_Anterior", "CC_Anterior", None),
        ("Unknown", "Unknown", None),
    ],
)
def test_strip_hemisphere_prefix_table(
    name: str,
    expected_stem: str,
    expected_hemi: str | None,
) -> None:
    stem, hemi = strip_hemisphere_prefix(name)
    assert (stem, hemi) == (expected_stem, expected_hemi)


def test_strip_hemisphere_prefix_strips_outer_whitespace() -> None:
    assert strip_hemisphere_prefix("  ctx-lh-insula  ") == ("insula", "L")


def test_strip_hemisphere_prefix_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="expected str"):
        strip_hemisphere_prefix(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assign_lobe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "lobe"),
    [
        # Cortical ribbon (lh + rh symmetry)
        ("ctx-lh-superiorfrontal", "frontal"),
        ("ctx-rh-superiorfrontal", "frontal"),
        ("ctx-lh-precentral", "frontal"),
        ("ctx-rh-frontalpole", "frontal"),
        ("ctx-lh-parsopercularis", "frontal"),
        # Cingulate is its own lobe (not lumped into frontal).
        ("ctx-lh-rostralanteriorcingulate", "cingulate"),
        ("ctx-rh-posteriorcingulate", "cingulate"),
        # Insula is its own lobe.
        ("ctx-lh-insula", "insula"),
        ("ctx-rh-insula", "insula"),
        # Parietal
        ("ctx-lh-superiorparietal", "parietal"),
        ("ctx-rh-precuneus", "parietal"),
        # Temporal
        ("ctx-lh-superiortemporal", "temporal"),
        ("ctx-rh-fusiform", "temporal"),
        ("ctx-lh-entorhinal", "temporal"),
        ("ctx-rh-parahippocampal", "temporal"),
        # Occipital
        ("ctx-lh-lateraloccipital", "occipital"),
        ("ctx-rh-cuneus", "occipital"),
        # Juxta-cortical white matter mirrors the cortical ribbon.
        ("wm-lh-superiorfrontal", "frontal"),
        ("wm-rh-fusiform", "temporal"),
        # Sub-cortical deep
        ("Left-Thalamus", "subcortical-deep"),
        ("Right-Caudate", "subcortical-deep"),
        ("Left-Putamen", "subcortical-deep"),
        ("Right-Pallidum", "subcortical-deep"),
        ("Left-Accumbens-area", "subcortical-deep"),
        ("Right-VentralDC", "subcortical-deep"),
        # Sub-cortical limbic
        ("Left-Hippocampus", "subcortical-limbic"),
        ("Right-Amygdala", "subcortical-limbic"),
        # Cerebellum
        ("Left-Cerebellum-Cortex", "cerebellum"),
        ("Right-Cerebellum-White-Matter", "cerebellum"),
        # Brainstem
        ("Brain-Stem", "brainstem"),
        # Other (CSF / ventricles / CC)
        ("Left-Lateral-Ventricle", "other"),
        ("3rd-Ventricle", "other"),
        ("CSF", "other"),
        ("CC_Anterior", "other"),
        ("Left-UnsegmentedWhiteMatter", "other"),
        ("Unknown", "other"),
    ],
)
def test_assign_lobe_table(name: str, lobe: str) -> None:
    assert assign_lobe(name) == lobe


def test_assign_lobe_unknown_falls_through_to_other() -> None:
    assert assign_lobe("ctx-lh-nonexistent-region") == "other"
    assert assign_lobe("Left-NewStructureFromFutureFreeSurfer") == "other"
    assert assign_lobe("totally-bogus") == "other"


def test_assign_lobe_handles_empty_string() -> None:
    assert assign_lobe("") == "other"


def test_assign_lobe_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="expected str"):
        assign_lobe(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assign_lobe_from_label (LUT mapping)
# ---------------------------------------------------------------------------


def test_assign_lobe_from_label_with_int_keys() -> None:
    lut: dict[int, str] = {1001: "ctx-lh-superiorfrontal", 17: "Left-Hippocampus"}
    assert assign_lobe_from_label(1001, lut) == "frontal"
    assert assign_lobe_from_label(17, lut) == "subcortical-limbic"


def test_assign_lobe_from_label_with_string_keys_from_json() -> None:
    raw = json.loads('{"1001": "ctx-lh-superiorfrontal", "17": "Left-Hippocampus"}')
    assert assign_lobe_from_label(1001, raw) == "frontal"
    assert assign_lobe_from_label(17, raw) == "subcortical-limbic"


def test_assign_lobe_from_label_missing_id_returns_other() -> None:
    assert assign_lobe_from_label(99999, {}) == "other"


# ---------------------------------------------------------------------------
# Coverage against a real cohort LUT (when available)
# ---------------------------------------------------------------------------


_COHORT_LUT_CANDIDATE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "derivatives_cohort50"
    / "sub-0005"
    / "parcellation"
    / "sub-0005_wmparc_lut.json"
)


@pytest.mark.skipif(
    not _COHORT_LUT_CANDIDATE.is_file(),
    reason="cohort LUT not extracted; this is a developer-machine guard test",
)
def test_every_real_cohort_lut_entry_is_classified() -> None:
    lut = json.loads(_COHORT_LUT_CANDIDATE.read_text())
    unmapped: list[str] = []
    classified_into_named_lobe = 0
    for _label, name in lut.items():
        lobe = assign_lobe(name)
        if lobe == "other":
            # "other" is acceptable for ventricles / CC / Unknown, but
            # we want >=80 % of LUT entries (excluding Unknown) to be
            # classified into a *named* lobe.
            unmapped.append(name)
        else:
            classified_into_named_lobe += 1
    n = len(lut)
    # Hard requirement: every name returns *some* lobe (no exceptions).
    assert classified_into_named_lobe + len(unmapped) == n
    # Soft requirement: most LUT entries are real anatomy, so classify
    # into a named lobe.
    assert classified_into_named_lobe >= 0.80 * n, (
        f"only {classified_into_named_lobe}/{n} names classified into a named lobe; "
        f"unmapped sample: {sorted(set(unmapped))[:10]}"
    )
