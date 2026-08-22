#!/bin/bash
#SBATCH --job-name=xvr-train-ljubljana-finetuned
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-10
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr train \
    -v experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    --ckptpath experiments/models/wbct/model.pth \
    --warp experiments/data/ljubljana/$SUBJECT/warp.txt \
    -o experiments/models/ljubljana/finetuned/$SUBJECT \
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
    --lr 0.001 \
    --p-augmentation 0.333 \
    --weight-haus 0.0 \
    --n-warmup-itrs 10 \
    --n-total-itrs 500 \
    --n-grad-accum-itrs 1 \
    --project xvr \
    --name ljubljana-$SUBJECT-finetuned

FINAL=$(ls experiments/models/ljubljana/finetuned/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/ljubljana/finetuned/$SUBJECT.pth
rm -rf experiments/models/ljubljana/finetuned/$SUBJECT
