#!/bin/bash
#SBATCH --job-name=xvr-train-deepfluoro-finetuned-cnxv2
#SBATCH --output=logs/deepfluoro_finetuned_cnxv2_%A_%a.out
#SBATCH --error=logs/deepfluoro_finetuned_cnxv2_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia_rtx_a6000|nvidia_rtx_6000_ada_generation"
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

xvr train \
    -v experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    --ckptpath experiments/models/foundation/cnvx2.pth \
    -o experiments/models/deepfluoro/finetuned_cnxv2/$SUBJECT \
    --orientation AP \
    --r1 135.0 225.0 \
    --r2 -45.0 45.0 \
    --r3 -15.0 15.0 \
    --tx -150.0 150.0 \
    --ty 450.0 1000.0 \
    --tz -150.0 150.0 \
    --sdd 1020.0 \
    --height 128 \
    --delx 2.1764375 \
    --model-name convnextv2_femto \
    --norm-layer default \
    --batch-size 116 \
    --p-augmentation 0.333 \
    --lr 0.0005 \
    --geodesic-only \
    --n-total-itrs 1500 \
    --n-save-every-itrs 1500 \
    --n-warmup-itrs 10 \
    --n-grad-accum-itrs 1 \
    --project xvr-final \
    --group deepfluoro \
    --name $SUBJECT-finetuned-cnxv2

FINAL=$(ls experiments/models/deepfluoro/finetuned_cnxv2/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/deepfluoro/finetuned_cnxv2/$SUBJECT.pth
rm -rf experiments/models/deepfluoro/finetuned_cnxv2/$SUBJECT
