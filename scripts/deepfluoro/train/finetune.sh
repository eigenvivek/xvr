#!/bin/bash
#SBATCH --job-name=xvr-train-deepfluoro-finetuning
#SBATCH --output=logs/deepfluoro_finetuning_%A_%a.out
#SBATCH --error=logs/deepfluoro_finetuning_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

xvr train \
    -v data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c models/wbct/model.pth \
    -w data/deepfluoro/$SUBJECT/warp2template.txt \
    -o models/deepfluoro/finetuned/$SUBJECT \
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
    --lr 0.001 \
    --p_augmentation 0.333 \
    --batch_size 116 \
    --n_warmup_itrs 10 \
    --n_total_itrs 500 \
    --n_grad_accum_itrs 1 \
    --name deepfluoro-$SUBJECT-finetuned \
    --project xvr
