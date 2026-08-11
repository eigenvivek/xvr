#!/bin/bash
#SBATCH --job-name=xvr-train-deepfluoro-finetuned
#SBATCH --output=logs/xvr_deepfluoro_finetuned_train_%A_%a.out
#SBATCH --error=logs/xvr_deepfluoro_finetuned_train_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr train \
    -v experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c experiments/models/wbct.pth \
    -w experiments/data/deepfluoro/$SUBJECT/warp.txt \
    -o experiments/models/deepfluoro/finetuned/$SUBJECT \
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
