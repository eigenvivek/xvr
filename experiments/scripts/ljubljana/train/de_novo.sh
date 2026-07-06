#!/bin/bash
#SBATCH --job-name=xvr-train-ljubljana-de-novo
#SBATCH --output=logs/ljubljana_de_novo_%A_%a.out
#SBATCH --error=logs/ljubljana_de_novo_%A_%a.err
#SBATCH --array=1-10
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=24:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

xvr train \
    -v data/ljubljana/$SUBJECT/volume.nii.gz \
    -o experiments/models/ljubljana/de_novo/$SUBJECT \
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
    --p-augmentation 0.5 \
    --lr 0.005 \
    --weight_dice 1.0 \
    --weight_haus 0.1 \
    --n-total-itrs 50000 \
    --n-warmup-itrs 500 \
    --n-grad-accum-itrs 1 \
    --n-save-every-itrs 250 \
    --project xvr-final \
    --group ljubljana \
    --name $SUBJECT-de-novo
FINAL=$(ls experiments/models/ljubljana/de_novo/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/ljubljana/de_novo/$SUBJECT.pth
rm -rf experiments/models/ljubljana/de_novo/$SUBJECT
