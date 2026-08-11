#!/bin/bash
#SBATCH --job-name=xvr-train-ljubljana-finetuned
#SBATCH --output=logs/xvr_ljubljana_finetuned_train_%A_%a.out
#SBATCH --error=logs/xvr_ljubljana_finetuned_train_%A_%a.err
#SBATCH --array=1-10
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
    -v experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    -c experiments/models/wbct.pth \
    -w experiments/data/ljubljana/$SUBJECT/warp.txt \
    -o experiments/models/ljubljana/finetuned/$SUBJECT \
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
