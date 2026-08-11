#!/bin/bash
#SBATCH --job-name=xvr-ttopt-deepfluoro-foundation
#SBATCH --output=logs/xvr_deepfluoro_foundation_ttopt_%A_%a.out
#SBATCH --error=logs/xvr_deepfluoro_foundation_ttopt_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=05:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

OUTDIR=experiments/results/deepfluoro/foundation/$SUBJECT
ANTIPODAL_OUTDIR=experiments/results/deepfluoro/foundation_antipodal/$SUBJECT

rm -rf "$OUTDIR" "$ANTIPODAL_OUTDIR"
mkdir -p "$OUTDIR" "$ANTIPODAL_OUTDIR"

xvr register model \
    experiments/data/deepfluoro/$SUBJECT/xrays \
    -v experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c experiments/models/wbct.pth \
    -o $OUTDIR \
    --crop 100 \
    --linearize \
    --labels 1,2,3,4,7 \
    --scales 24,12,6 \
    --n_itrs 500,500,500 \
    --warp experiments/data/deepfluoro/$SUBJECT/warp.txt

xvr register model \
    experiments/data/deepfluoro/$SUBJECT/xrays \
    -v experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c experiments/models/wbct.pth \
    -o $ANTIPODAL_OUTDIR \
    --crop 100 \
    --linearize \
    --labels 1,2,3,4,7 \
    --scales 24,12,6 \
    --n_itrs 500,500,500 \
    --warp experiments/data/deepfluoro/$SUBJECT/warp.txt \
    --antipodal
