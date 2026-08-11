#!/bin/bash
#SBATCH --job-name=xvr-train-femur-finetuned
#SBATCH --output=logs/xvr_femur_finetuned_train_%A_%a.out
#SBATCH --error=logs/xvr_femur_finetuned_train_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr train \
    -v experiments/data/femur/$SUBJECT/volume.nii.gz \
    -m experiments/data/femur/$SUBJECT/mask.nii.gz \
    -c experiments/models/wbct.pth \
    -w experiments/data/femur/$SUBJECT/warp.txt \
    -o experiments/models/femur/finetuned/$SUBJECT \
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
    --lr 0.001 \
    --p_augmentation 0.333 \
    --batch_size 116 \
    --n_warmup_itrs 10 \
    --n_total_itrs 500 \
    --n_grad_accum_itrs 1 \
    --name femur-$SUBJECT-finetuned \
    --project xvr
