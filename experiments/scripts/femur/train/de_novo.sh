#!/bin/bash
#SBATCH --job-name=xvr-train-femur-de-novo
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=24:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr train \
    -v experiments/data/femur/$SUBJECT/volume.nii.gz \
    -m experiments/data/femur/$SUBJECT/mask.nii.gz \
    -o experiments/models/femur/de_novo/$SUBJECT \
    --r1 75.0 270.0 \
    --r2 -20.0 20.0 \
    --r3 -20.0 20.0 \
    --tx -75.0 75.0 \
    --ty 650.0 950.0 \
    --tz 0.0 100.0 \
    --sdd 1150.0 \
    --height 128 \
    --delx 2.31796875 \
    --model-name resnet34 \
    --batch-size 116 \
    --lr 0.001 \
    --weight-haus 0.0 \
    --n-total-itrs 30000 \
    --n-save-every-itrs 250 \
    --project xvr \
    --name femur-$SUBJECT-de-novo

FINAL=$(ls experiments/models/femur/de_novo/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/femur/de_novo/$SUBJECT.pth
rm -rf experiments/models/femur/de_novo/$SUBJECT
