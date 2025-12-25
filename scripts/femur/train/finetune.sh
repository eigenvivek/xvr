#!/bin/bash
#SBATCH --job-name=xvr-femur-finetuning
#SBATCH --output=logs/femur_finetuning_%A_%a.out
#SBATCH --error=logs/femur_finetuning_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

xvr train \
    -v data/femur/$SUBJECT/volume.nii.gz \
    -m data/femur/$SUBJECT/mask_all.nii.gz \
    -c models/wbct/model.pth \
    -w data/femur/$SUBJECT/warp2template.txt \
    -o models/femur/finetuned/$SUBJECT \
    --r1 75.0 270.0 \
    --r2 -20.0 20.0 \
    --r3 -20.0 20.0 \
    --tx -75.0 75.0 \
    --ty 650.0 950.0 \
    --tz 0.0 100.0 \
    --sdd 1150.0 \
    --height 128 \
    --delx 2.31796875 \
    --model_name resnet34 \
    --lr 0.00025 \
    --p_augmentation 0.333 \
    --batch_size 96 \
    --n_warmup_itrs 10 \
    --n_total_itrs 300 \
    --n_grad_accum_itrs 1 \
    --name femur-$SUBJECT-finetuned \
    --project xvr
