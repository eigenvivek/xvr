#!/bin/bash
#SBATCH --job-name=xvr-register-ljubljana-de-novo
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-10
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

CKPT=experiments/models/ljubljana/de_novo/$SUBJECT.pth
OUTDIR=experiments/results/ljubljana/de_novo/$SUBJECT
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# the glob skips the *_max.dcm maximum-intensity projections, which are not registered
xvr register model \
    --files experiments/data/ljubljana/$SUBJECT/xrays/*[!_max].dcm \
    --ckpt "$CKPT" \
    --imagepath experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    --scales 16 8 4 \
    --n-itrs 500 500 500 \
    --patience 10 10 10 \
    --linearize \
    --subtract-background \
    --savepath "$OUTDIR"
