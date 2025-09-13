#!/bin/bash
#SBATCH --job-name=xvr-deepfluoro-patient-specific
#SBATCH --output=logs/deepfluoro_%A_%a.out
#SBATCH --error=logs/deepfluoro_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Start training!
uv run xvr train \
    -v data/deepfluoro/subject0$SLURM_ARRAY_TASK_ID/volume.nii.gz \
    -m data/deepfluoro/subject0$SLURM_ARRAY_TASK_ID/mask.nii.gz \
    -o models/deepfluoro/patient_specific/subject0$SLURM_ARRAY_TASK_ID \
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
    --n_total_itrs 100000 \
    --n_save_every_itrs 1000 \
    --name patient-specific-0$SLURM_ARRAY_TASK_ID \
    --project xvr-deepfluoro
