#!/bin/bash
#SBATCH --job-name=xvr-train-deepfluoro-de-novo
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-6
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
    -v experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    -o experiments/models/deepfluoro/de_novo/$SUBJECT \
    --r1 135.0 225.0 \
    --r2 -45.0 45.0 \
    --r3 -15.0 15.0 \
    --tx -150.0 150.0 \
    --ty 450.0 1000.0 \
    --tz -150.0 150.0 \
    --sdd 1020.0 \
    --height 128 \
    --delx 2.1764375 \
    --model-name resnet34 \
    --batch-size 116 \
    --lr 0.001 \
    --weight-haus 0.0 \
    --n-total-itrs 30000 \
    --n-save-every-itrs 250 \
    --project xvr \
    --name deepfluoro-$SUBJECT-de-novo

FINAL=$(ls experiments/models/deepfluoro/de_novo/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/deepfluoro/de_novo/$SUBJECT.pth
rm -rf experiments/models/deepfluoro/de_novo/$SUBJECT
