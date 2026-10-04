#!/usr/bin/env bash
set -euo pipefail
cd ~/GitHub/high-precision-glioma-segmentation

echo "=== Pre-flight ==="
which recon-all-clinical.sh
echo "FREESURFER_HOME=$FREESURFER_HOME"

SCRATCH="$HOME/fs_scratch"
mkdir -p "$SCRATCH" logs

uv run python scripts/parcellate_all_cohort50.py --dry-run | tail -5

echo
echo "=== Blocking sleep ==="
caffeinate -i -s &
CAFF_PID=$!
echo "caffeinate PID=$CAFF_PID"

echo
echo "=== Launching parcellate-all (FreeSurfer, CPU) ==="
LOG_PARC="logs/parcellate_$(date +%Y%m%d_%H%M%S).log"
nohup make parcellate-all \
    PARCELLATE_FS_WORK_DIR="$SCRATCH" \
    PARCELLATE_THREADS=10 \
    PARCELLATE_SKIP_EXISTING=1 \
    > "$LOG_PARC" 2>&1 &
PID_PARC=$!
disown $PID_PARC
echo "parcellate-all PID=$PID_PARC  log=$LOG_PARC"

echo
echo "=== Launching segment-all (MONAI Bundle, MPS GPU) ==="
LOG_SEG="logs/segment_$(date +%Y%m%d_%H%M%S).log"
nohup make segment-all \
    SEGMENT_DEVICE=mps \
    SEGMENT_SKIP_EXISTING=1 \
    > "$LOG_SEG" 2>&1 &
PID_SEG=$!
disown $PID_SEG
echo "segment-all PID=$PID_SEG  log=$LOG_SEG"

echo
echo "=== Both runs launched. You can close this terminal. ==="
echo "Morning checklist:"
echo "  tail -40 $LOG_PARC"
echo "  tail -40 $LOG_SEG"
echo "  ls data/derivatives_cohort50/sub-*/parcellation/*wmparc_native.nii.gz | wc -l"
echo "  ls data/derivatives_cohort50/sub-*/seg_dl/*seg_brats3_dl.nii.gz | wc -l"
echo "  kill $CAFF_PID    # release sleep block"
