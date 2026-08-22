#!/bin/bash
#SBATCH --job-name=xvr-register-deepfluoro-de-novo
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

CKPT=experiments/models/deepfluoro/de_novo/$SUBJECT.pth
OUTDIR=experiments/results/deepfluoro/de_novo/$SUBJECT
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

xvr register model \
    --files experiments/data/deepfluoro/$SUBJECT/xrays/*.dcm \
    --ckpt "$CKPT" \
    --imagepath experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    --labelpath experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    --labels 1 2 3 4 7 \
    --scales 24 12 6 \
    --n-itrs 500 500 500 \
    --patience 10 10 10 \
    --crop 100 \
    --linearize \
    --savepath "$OUTDIR"
