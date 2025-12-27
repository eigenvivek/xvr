#!/bin/bash
#SBATCH --job-name=xvr-wbct-deepfluoro-finetuned-eval
#SBATCH --output=logs/xvr_wbct_deepfluoro_finetuned_eval_%A_%a.out
#SBATCH --error=logs/xvr_wbct_deepfluoro_finetuned_eval_%A_%a.err
#SBATCH --array=0-185
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Get all checkpoint files as an array
CKPTDIR="models/deepfluoro/finetuned-ckpts"
CKPTS=($(ls "$CKPTDIR"/*/*.pth | sort))

# Calculate checkpoint index and subject ID from SLURM_ARRAY_TASK_ID
SUBJECT_ID=$((SLURM_ARRAY_TASK_ID / 31 + 1))

# Get the checkpoint path
CKPTPATH="${CKPTS[$SLURM_ARRAY_TASK_ID]}"

# Extract checkpoint index
CKPTFILE=$(basename "$CKPTPATH")
CKPT_IDX="${CKPTFILE%.pth}"

# Format subject ID with leading zeros (subject01, subject02, etc.)
SUBJECT=$(printf "subject%02d" $SUBJECT_ID)

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing checkpoint: $CKPTFILE (index: $CKPT_IDX)"
echo "Subject: $SUBJECT"

# Create output directory
mkdir -p "results/deepfluoro/evaluate/finetuned/$SUBJECT/$CKPT_IDX"

source .venv/bin/activate

xvr register model \
    "data/deepfluoro/$SUBJECT/xrays" \
    -v "data/deepfluoro/$SUBJECT/volume.nii.gz" \
    -m "data/deepfluoro/$SUBJECT/mask.nii.gz" \
    -c "$CKPTPATH" \
    -o "results/deepfluoro/evaluate/finetuned/$SUBJECT/$CKPT_IDX" \
    --crop 100 \
    --linearize \
    --warp "data/deepfluoro/$SUBJECT/warp2template.txt" \
    --init_only \
    --verbose 0
