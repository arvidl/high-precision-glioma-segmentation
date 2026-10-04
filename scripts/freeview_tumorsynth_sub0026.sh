#!/usr/bin/env bash
set -euo pipefail

# Interactive Freeview QC for the mri_TumorSynth comparator on UCSF-PDGM sub-0026.
#
# The script opens the same staged mpMRI inputs used by the MONAI Bundle and
# TumorSynth runs, with TumorSynth as the primary overlay and MONAI/reference
# masks loaded as toggleable comparators.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBJECT_ID="${SUBJECT_ID:-0026}"
SUBJECT_TAG="sub-${SUBJECT_ID}"
COHORT_ROOT="${COHORT_ROOT:-${REPO_ROOT}/data/ucsf_pdgm_cohort50}"
DERIV_ROOT="${DERIV_ROOT:-${REPO_ROOT}/data/derivatives_cohort50}"
FREEVIEW="${FREEVIEW:-}"

if [[ -z "${FREEVIEW}" ]]; then
  if command -v freeview >/dev/null 2>&1; then
    FREEVIEW="$(command -v freeview)"
  elif [[ -x /Applications/freesurfer/8.2.0/bin/freeview ]]; then
    FREEVIEW="/Applications/freesurfer/8.2.0/bin/freeview"
  else
    echo "ERROR: freeview not found. Set FREEVIEW=/path/to/freeview." >&2
    exit 1
  fi
fi

SUBJECT_DIR="${COHORT_ROOT}/${SUBJECT_TAG}"
DERIV_DIR="${DERIV_ROOT}/${SUBJECT_TAG}"

T1="${SUBJECT_DIR}/${SUBJECT_TAG}_T1_bias.nii.gz"
T1C="${SUBJECT_DIR}/${SUBJECT_TAG}_T1c_bias.nii.gz"
T2="${SUBJECT_DIR}/${SUBJECT_TAG}_T2_bias.nii.gz"
FLAIR="${SUBJECT_DIR}/${SUBJECT_TAG}_FLAIR_bias.nii.gz"
GT="${SUBJECT_DIR}/${SUBJECT_TAG}_tumor_segmentation.nii.gz"
DL="${DERIV_DIR}/seg_dl/${SUBJECT_TAG}_seg_brats3_dl.nii.gz"
TUMORSYNTH="${DERIV_DIR}/seg_tumorsynth/${SUBJECT_TAG}_seg_tumorsynth.nii.gz"

missing=0
for path in "${T1}" "${T1C}" "${T2}" "${FLAIR}" "${GT}" "${DL}" "${TUMORSYNTH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing file: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

LUT="${DERIV_DIR}/seg_tumorsynth/${SUBJECT_TAG}_brats_tumor_lut.txt"
mkdir -p "$(dirname "${LUT}")"
cat > "${LUT}" <<'EOF'
0 Background 0 0 0 0
1 NCR_or_NET 255 0 0 255
2 Edema 0 180 0 255
4 Enhancing_Tumor 255 220 0 255
EOF

CMD=(
  "${FREEVIEW}"
  -v "${T1C}:name=T1c:grayscale=0,400"
  -v "${FLAIR}:name=FLAIR:grayscale=0,400"
  -v "${T2}:name=T2:grayscale=0,400:visible=0"
  -v "${T1}:name=T1:grayscale=0,400:visible=0"
  -v "${TUMORSYNTH}:name=TumorSynth_BRATS:colormap=lut:lut=${LUT}:opacity=0.45"
  -v "${DL}:name=MONAI_BRATS:colormap=lut:lut=${LUT}:opacity=0.35:visible=0"
  -v "${GT}:name=Reference_BRATS:colormap=lut:lut=${LUT}:opacity=0.35:visible=0"
)

echo "Launching Freeview for ${SUBJECT_TAG}"
echo "TumorSynth overlay : ${TUMORSYNTH}"
echo "MONAI overlay      : ${DL}"
echo "Reference overlay  : ${GT}"
echo "LUT                : ${LUT}"
echo
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
