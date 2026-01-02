#!/bin/bash
#SBATCH --job-name=xvr-train-ljubljana-finetuning
#SBATCH --output=logs/ljubljana_finetuning_%A_%a.out
#SBATCH --error=logs/ljubljana_finetuning_%A_%a.err
#SBATCH --array=1-10
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
    -v data/ljubljana/$SUBJECT/volume.nii.gz \
    -c models/wbct/model.pth \
    -w data/ljubljana/$SUBJECT/warp2template.txt \
    -o models/ljubljana/finetuned/$SUBJECT \
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
    --p_augmentation 0.333 \
    --batch_size 116 \
    --n_warmup_itrs 10 \
    --n_total_itrs 500 \
    --n_grad_accum_itrs 1 \
    --name ljubljana-$SUBJECT-finetuned \
    --project xvr
