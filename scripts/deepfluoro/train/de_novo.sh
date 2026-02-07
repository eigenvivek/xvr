#!/bin/bash
#SBATCH --job-name=xvr-train-deepfluoro-de-novo
#SBATCH --output=logs/deepfluoro_de_novo_%A_%a.out
#SBATCH --error=logs/deepfluoro_de_novo_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=24:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

xvr train \
    -v data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m data/deepfluoro/$SUBJECT/mask.nii.gz \
    -o models/deepfluoro/de_novo/$SUBJECT \
    --r1 135.0 225.0 \
    --r2 -45.0 45.0 \
    --r3 -15.0 15.0 \
    --tx -150.0 150.0 \
    --ty 450.0 1000.0 \
    --tz -150.0 150.0 \
    --sdd 1020.0 \
    --height 128 \
    --delx 2.1764375 \
    --model_name resnet34 \
    --batch_size 116 \
    --lr 0.001 \
    --n_total_itrs 30000 \
    --n_save_every_itrs 250 \
    --name deepfluoro-$SUBJECT-de-novo \
    --project xvr
