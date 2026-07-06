#!/bin/bash
#SBATCH --job-name=xvr-train-femur-de-novo
#SBATCH --output=logs/femur_de_novo_%A_%a.out
#SBATCH --error=logs/femur_de_novo_%A_%a.err
#SBATCH --array=1-5
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
    -v data/femur/$SUBJECT/volume.nii.gz \
    -m data/femur/$SUBJECT/mask_all.nii.gz \
    -o experiments/models/femur/de_novo/$SUBJECT \
    --orientation AP \
    --r1 75.0 270.0 \
    --r2 -20.0 20.0 \
    --r3 -20.0 20.0 \
    --tx -75.0 75.0 \
    --ty 650.0 950.0 \
    --tz 0.0 100.0 \
    --sdd 1150.0 \
    --height 128 \
    --delx 2.31796875 \
    --model-name resnet34 \
    --batch-size 116 \
    --p-augmentation 0.5 \
    --lr 0.001 \
    --weight_dice 1.0 \
    --weight_haus 0.1 \
    --n-total-itrs 50000 \
    --n-warmup-itrs 500 \
    --n-grad-accum-itrs 1 \
    --n-save-every-itrs 250 \
    --project xvr-final \
    --group femur \
    --name $SUBJECT-de-novo
FINAL=$(ls experiments/models/femur/de_novo/$SUBJECT/*.pth | sort | tail -n 1)
mv "$FINAL" experiments/models/femur/de_novo/$SUBJECT.pth
rm -rf experiments/models/femur/de_novo/$SUBJECT
