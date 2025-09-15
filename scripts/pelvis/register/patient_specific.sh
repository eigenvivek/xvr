#!/bin/bash
#SBATCH --job-name=xvr-deepfluoro-patient-specific
#SBATCH --output=logs/deepfluoro_%A_%a.out
#SBATCH --error=logs/deepfluoro_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

mkdir -p logs

uv run xvr register model \
    data/deepfluoro/subject0$SLURM_ARRAY_TASK_ID/xrays \
    -v data/deepfluoro/subject0$SLURM_ARRAY_TASK_ID/volume.nii.gz \
    -m data/deepfluoro/subject0$SLURM_ARRAY_TASK_ID/mask.nii.gz \
    -c models/pelvis/patient_specific/deepfluoro0$SLURM_ARRAY_TASK_ID.pth \
    -o results/deepfluoro/register/patient_specific/subject0$SLURM_ARRAY_TASK_ID \
    --crop 100 \
    --linearize \
    --labels 1,2,3,4,7 \
    --scales 24,12,6 \
    --n_itrs 250,150,100 \
    --reverse_x_axis
