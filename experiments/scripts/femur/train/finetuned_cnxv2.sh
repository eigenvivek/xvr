#!/bin/bash
#SBATCH --job-name=xvr-train-femur-finetuned-cnxv2
#SBATCH --output=logs/femur_finetuned_cnxv2_%A_%a.out
#SBATCH --error=logs/femur_finetuned_cnxv2_%A_%a.err
#SBATCH --array=1-5
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
    -v data/femur/$SUBJECT/volume.nii.gz \
    -m data/femur/$SUBJECT/mask_all.nii.gz \
    --ckptpath experiments/models/foundation/cnvx2.pth \
    -o experiments/models/femur/finetuned_cnxv2/$SUBJECT \
    --orientation AP \
    --r1 75.0 270.0 \
    --r2 -20.0 20.0 \
    --r3 -20.0 20.0 \
    --tx -75.0 75.0 \
    --ty 650.0 950.0 \
    --tz 0.0 100.0 \
    --sdd 1150.0 \
    --height 128 \
    --delx 2.31796875 \
    --model-name convnextv2_femto \
    --norm-layer default \
    --batch-size 116 \
    --p-augmentation 0.333 \
    --lr 0.0005 \
    --geodesic-only \
    --n-total-itrs 1500 \
    --n-warmup-itrs 10 \
    --n-grad-accum-itrs 1 \
    --n-save-every-itrs 1500 \
    --project xvr-final \
    --group femur \
    --name $SUBJECT-finetuned-cnxv2

FINAL=$(ls experiments/models/femur/finetuned_cnxv2/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/femur/finetuned_cnxv2/$SUBJECT.pth
rm -rf experiments/models/femur/finetuned_cnxv2/$SUBJECT
