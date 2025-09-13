#!/bin/bash
#SBATCH --job-name=xvr-deepfluoro-patient-agnostic
#SBATCH --output=logs/deepfluoro_pa_%A.out
#SBATCH --error=logs/deepfluoro_pa_%A.err
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=96:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Start training!
uv run xvr train \
    -v data/ctpelvic1k/imgs_registered \
    -o models/deepfluoro/patient_agnostic \
    --r1 -45.0 45.0 \
    --r2 -45.0 45.0 \
    --r3 -15.0 15.0 \
    --tx -150.0 150.0 \
    --ty -1000.0 -450.0 \
    --tz -150.0 150.0 \
    --sdd 1020.0 \
    --height 256 \
    --delx 1.08821875 \
    --reverse_x_axis \
    --batch_size 28 \
    --n_total_itrs 200000 \
    --n_save_every_itrs 1000 \
    --name patient-agnostic \
    --project xvr-deepfluoro
