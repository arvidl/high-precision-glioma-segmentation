"""Canonical FreeSurfer wmparc/aparc-name to anatomical-lobe assignment.

This is the single source of truth for the lobe binning used by the
PR-7l cohort lobe x compartment heatmap producer
(``scripts/build_lobe_compartment_heatmap.py``) and any downstream
sunburst / heatmap successor to Figure 15. It deliberately lives in
``hpgs.parcellate`` next to :func:`predict_wmparc` so the parcellation
producer and the consumer share one mapping.

Design choices (aligned with the Desikan-Killiany 2006 atlas as shipped
in FreeSurfer 8.2.0's ``aparc.annot`` + the wmparc subcortical roll-up):

* The **34 cortical Desikan-Killiany stems** are partitioned into six
  cortical lobes (frontal, parietal, temporal, occipital, cingulate,
  insula). The cingulate is broken out as its own lobe rather than
  being lumped into the frontal lobe -- this matches the aparc
  *lobesStrict* convention used by FreeSurfer's
  ``mri_annotation2label --lobesStrict`` and is the convention used by
  the BraTS-relevant glioma-distribution literature (Larjavaara 2007;
  Duffau & Capelle 2004). The insula is a separate lobe for the same
  reason -- merging it into the frontal lobe would discard one of the
  most clinically meaningful tumor locations (insular glioma).
* Both the cortical ribbon (``ctx-{lh,rh}-<stem>``) and the
  superficial juxtacortical white matter (``wm-{lh,rh}-<stem>``) map
  to the same lobe. wmparc's ``wm-...`` labels are by construction the
  voxels within ~5 mm of the corresponding ``ctx-...`` ribbon, so this
  is the conventional grouping when answering "where in the brain is
  this tumor?".
* **Sub-cortical** structures are coarsened into four buckets:
  ``subcortical-deep`` (thalamus, basal ganglia, ventral DC),
  ``subcortical-limbic`` (hippocampus, amygdala),
  ``cerebellum`` (cortex + WM), and ``brainstem``. This is finer than
  a single "sub-cortical" row but coarser than the per-structure list
  that would dominate any cohort-level heatmap with mostly-empty cells.
* CSF / ventricles, the corpus callosum (``CC_*``), the global
  ``Unknown`` and per-hemisphere ``UnsegmentedWhiteMatter`` placeholders
  are mapped to ``other`` so the producer can either drop or display
  them separately. Their voxels are still accounted for in the producer's
  reconciliation footer.

Public API (`from hpgs.parcellate.lobes import ...`)::

    LOBES                     -- ordered tuple of canonical lobe names
    LOBE_OF_CORTICAL_STEM     -- {Desikan stem (str): lobe (str)}
    LOBE_OF_SUBCORTICAL_NAME  -- {wmparc subcortical name: lobe (str)}
    assign_lobe(name)         -- (str) -> (str), the workhorse
    assign_lobe_from_label(label_id, lut) -> (str)

The function ``assign_lobe(name)`` accepts any string that appears as a
*value* in a wmparc LUT JSON (``data/derivatives_cohort50/sub-XXXX/
parcellation/sub-XXXX_wmparc_lut.json``) and returns the lobe name. It
is intentionally permissive about case and whitespace so it round-trips
between the FreeSurfer LUT, our JSON sidecar, and any downstream CSV.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

__all__ = [
    "LOBES",
    "LOBE_OF_CORTICAL_STEM",
    "LOBE_OF_SUBCORTICAL_NAME",
    "assign_lobe",
    "assign_lobe_from_label",
    "strip_hemisphere_prefix",
]

# Display-ordered tuple of every lobe ``assign_lobe`` may return.
# Order is anatomical / clinical, *not* alphabetical: cortical lobes
# first (rostro-caudal), then deep / limbic, then infratentorial, then
# the catch-all bucket. Heatmap row order follows this.
LOBES: Final[tuple[str, ...]] = (
    "frontal",
    "cingulate",
    "insula",
    "parietal",
    "temporal",
    "occipital",
    "subcortical-deep",
    "subcortical-limbic",
    "cerebellum",
    "brainstem",
    "other",
)

# Desikan-Killiany 2006 -> lobe. Keys are the 34 *stems* that follow
# the ``ctx-{lh,rh}-`` and ``wm-{lh,rh}-`` prefixes in the wmparc LUT.
LOBE_OF_CORTICAL_STEM: Final[Mapping[str, str]] = {
    "superiorfrontal": "frontal",
    "rostralmiddlefrontal": "frontal",
    "caudalmiddlefrontal": "frontal",
    "parsopercularis": "frontal",
    "parstriangularis": "frontal",
    "parsorbitalis": "frontal",
    "lateralorbitofrontal": "frontal",
    "medialorbitofrontal": "frontal",
    "precentral": "frontal",
    "paracentral": "frontal",
    "frontalpole": "frontal",
    "rostralanteriorcingulate": "cingulate",
    "caudalanteriorcingulate": "cingulate",
    "posteriorcingulate": "cingulate",
    "isthmuscingulate": "cingulate",
    "insula": "insula",
    "superiorparietal": "parietal",
    "inferiorparietal": "parietal",
    "supramarginal": "parietal",
    "postcentral": "parietal",
    "precuneus": "parietal",
    "superiortemporal": "temporal",
    "middletemporal": "temporal",
    "inferiortemporal": "temporal",
    "bankssts": "temporal",
    "fusiform": "temporal",
    "transversetemporal": "temporal",
    "entorhinal": "temporal",
    "temporalpole": "temporal",
    "parahippocampal": "temporal",
    "lateraloccipital": "occipital",
    "lingual": "occipital",
    "cuneus": "occipital",
    "pericalcarine": "occipital",
}

# Sub-cortical wmparc names (as they appear in the LUT JSON values),
# stripped of the ``Left-`` / ``Right-`` prefix. ``assign_lobe`` does
# the prefix strip before looking up here.
LOBE_OF_SUBCORTICAL_NAME: Final[Mapping[str, str]] = {
    "Thalamus": "subcortical-deep",
    "Caudate": "subcortical-deep",
    "Putamen": "subcortical-deep",
    "Pallidum": "subcortical-deep",
    "Accumbens-area": "subcortical-deep",
    "VentralDC": "subcortical-deep",
    "Hippocampus": "subcortical-limbic",
    "Amygdala": "subcortical-limbic",
    "Cerebellum-Cortex": "cerebellum",
    "Cerebellum-White-Matter": "cerebellum",
    "Brain-Stem": "brainstem",
}

_CORTICAL_PREFIXES: Final[tuple[str, ...]] = (
    "ctx-lh-",
    "ctx-rh-",
    "wm-lh-",
    "wm-rh-",
)

_HEMI_PREFIXES: Final[tuple[str, ...]] = (
    "Left-",
    "Right-",
)

_OTHER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "Unknown",
        "CSF",
        "Lateral-Ventricle",
        "Inf-Lat-Vent",
        "3rd-Ventricle",
        "4th-Ventricle",
        "5th-Ventricle",
        "choroid-plexus",
        "Choroid-Plexus",
        "UnsegmentedWhiteMatter",
        "WM-hypointensities",
        "non-WM-hypointensities",
        "Optic-Chiasm",
        "CC_Anterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Mid_Posterior",
        "CC_Posterior",
    },
)


def strip_hemisphere_prefix(name: str) -> tuple[str, str | None]:
    """Strip the ``ctx-{lh,rh}-``, ``wm-{lh,rh}-``, ``Left-`` or ``Right-`` prefix.

    Returns ``(stem, hemisphere)`` where ``hemisphere`` is one of
    ``"L"`` / ``"R"`` / ``None`` (for prefix-less names like
    ``"Brain-Stem"``).
    """
    if not isinstance(name, str):
        raise TypeError(f"strip_hemisphere_prefix: expected str, got {type(name).__name__}")
    s = name.strip()
    for pref in _CORTICAL_PREFIXES:
        if s.startswith(pref):
            hemi = "L" if "lh" in pref else "R"
            return s[len(pref) :], hemi
    for pref in _HEMI_PREFIXES:
        if s.startswith(pref):
            hemi = "L" if pref == "Left-" else "R"
            return s[len(pref) :], hemi
    return s, None


def assign_lobe(name: str) -> str:
    """Map a single wmparc LUT name to one of :data:`LOBES`.

    Recognises the four cortical prefix patterns and the ``Left-``/
    ``Right-`` sub-cortical prefix. Unknown / unmapped names fall
    through to ``"other"`` rather than raising, so the producer can
    keep going if FreeSurfer ever ships a new structure name. Use
    :func:`assign_lobe_from_label` if you have a label id + LUT and
    want to short-circuit the missing-label case.
    """
    if not isinstance(name, str):
        raise TypeError(f"assign_lobe: expected str, got {type(name).__name__}")
    stem, _hemi = strip_hemisphere_prefix(name)
    if not stem:
        return "other"
    # Cortical / juxtacortical WM ribbon: stem is a Desikan label.
    if stem in LOBE_OF_CORTICAL_STEM:
        return LOBE_OF_CORTICAL_STEM[stem]
    # Sub-cortical: stem matches a hemisphere-stripped name.
    if stem in LOBE_OF_SUBCORTICAL_NAME:
        return LOBE_OF_SUBCORTICAL_NAME[stem]
    # Known catch-all (ventricles, CC, hypointensities, ...).
    if stem in _OTHER_NAMES or name in _OTHER_NAMES:
        return "other"
    return "other"


def assign_lobe_from_label(label_id: int, lut: Mapping[int | str, str]) -> str:
    """Map a numeric wmparc label id to a lobe via a LUT mapping.

    ``lut`` is the JSON sidecar loaded from ``..._wmparc_lut.json``;
    its keys may be ``int`` *or* ``str`` (json.load gives strings).
    Missing ids return ``"other"``.
    """
    name = lut.get(int(label_id)) or lut.get(str(int(label_id)))
    if not name:
        return "other"
    return assign_lobe(name)
