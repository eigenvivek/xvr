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
    -o models/ljubljana/de_novo/$SUBJECT \
    --r1 -45.0 105.0 \
    --r2 -5.0 5.0 \
    --r3 -5.0 5.0 \
    --tx -25.0 25.0 \
    --ty 700.0 800.0 \
    --tz -25.0 25.0 \
    --sdd 1250.0 \
    --height 128 \
    --delx 2.31 \
    --model_name resnet34 \
    --lr 0.001 \
    --batch_size 116 \
    --n_total_itrs 30000 \
    --n_save_every_itrs 250 \
    --name ljubljana-$SUBJECT-de-novo \
    --project xvr
