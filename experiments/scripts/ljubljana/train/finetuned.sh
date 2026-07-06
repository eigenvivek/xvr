#!/bin/bash
#SBATCH --job-name=xvr-train-ljubljana-finetuned
#SBATCH --output=logs/ljubljana_finetuned_%A_%a.out
#SBATCH --error=logs/ljubljana_finetuned_%A_%a.err
#SBATCH --array=1-10
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
    -v data/ljubljana/$SUBJECT/volume.nii.gz \
    --ckptpath experiments/models/foundation/resnet34.pth \
    -o experiments/models/ljubljana/finetuned/$SUBJECT \
    --orientation AP \
    --r1 -45.0 105.0 \
    --r2 -5.0 5.0 \
    --r3 -5.0 5.0 \
    --tx -25.0 25.0 \
    --ty 700.0 800.0 \
    --tz -25.0 25.0 \
    --sdd 1250.0 \
    --height 128 \
    --delx 2.31 \
    --model-name resnet34 \
    --batch-size 116 \
    --p-augmentation 0.333 \
    --lr 0.005 \
    --geodesic-only \
    --n-total-itrs 1500 \
    --n-warmup-itrs 10 \
    --n-grad-accum-itrs 1 \
    --n-save-every-itrs 1500 \
    --project xvr-final \
    --group ljubljana \
    --name $SUBJECT-finetuned

FINAL=$(ls experiments/models/ljubljana/finetuned/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/ljubljana/finetuned/$SUBJECT.pth
rm -rf experiments/models/ljubljana/finetuned/$SUBJECT
