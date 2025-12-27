#!/bin/bash
#SBATCH --job-name=xvr-wbct-ljubljana-eval
#SBATCH --output=logs/xvr_wbct_ljubljana_eval_%A_%a.out
#SBATCH --error=logs/xvr_wbct_ljubljana_eval_%A_%a.err
#SBATCH --array=0-2999
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
CKPTDIR="models/wbct/resnet34-quaternion_adjugate"
CKPTS=($(ls "$CKPTDIR"/*.pth | sort))

# Calculate checkpoint index and subject ID from SLURM_ARRAY_TASK_ID
# Total array indices: 300 checkpoints × 10 subjects = 3000 tasks (0-2999)
CKPT_ARRAY_IDX=$((SLURM_ARRAY_TASK_ID / 10))
SUBJECT_ID=$((SLURM_ARRAY_TASK_ID % 10 + 1))

# Get the checkpoint path
CKPTPATH="${CKPTS[$CKPT_ARRAY_IDX]}"

# Extract checkpoint index
CKPTFILE=$(basename "$CKPTPATH")
CKPT_IDX="${CKPTFILE%.pth}"

# Format subject ID with leading zeros (subject01, subject02, etc.)
SUBJECT=$(printf "subject%02d" $SUBJECT_ID)

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing checkpoint: $CKPTFILE (index: $CKPT_IDX)"
echo "Subject: $SUBJECT"

# Create output directory
mkdir -p "results/ljubljana/evaluate/foundation/$SUBJECT/$CKPT_IDX"

source /data/vision/polina/users/vivekg/xvr/.venv/bin/activate

xvr register model \
    "data/ljubljana/$SUBJECT/xrays" \
    -v "data/ljubljana/$SUBJECT/volume.nii.gz" \
    -c "$CKPTPATH" \
    -o "results/ljubljana/evaluate/foundation/$SUBJECT/$CKPT_IDX" \
    --linearize \
    --subtract_background \
    --warp "data/ljubljana/$SUBJECT/warp2template.txt" \
    --init_only \
    --pattern *[!_max].dcm \
    --verbose 0
