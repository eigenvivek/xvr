#!/bin/bash
#SBATCH --job-name=xvr-register-deepfluoro-finetuned
#SBATCH --output=logs/deepfluoro_register_finetuned_%A_%a.out
#SBATCH --error=logs/deepfluoro_register_finetuned_%A_%a.err
#SBATCH --array=1-6
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

CKPT=experiments/models/deepfluoro/finetuned/$SUBJECT.pth
SAVEPATH=experiments/results/deepfluoro/finetuned/$SUBJECT
mkdir -p "$SAVEPATH"

echo "Subject:  $SUBJECT"
echo "Ckpt:     $CKPT"
echo "Savepath: $SAVEPATH"

xvr register model \
    --files experiments/data/deepfluoro/$SUBJECT/xrays/*.dcm \
    --ckpt "$CKPT" \
    --imagepath experiments/data/deepfluoro/$SUBJECT/volume.nii.gz \
    --labelpath experiments/data/deepfluoro/$SUBJECT/mask.nii.gz \
    --warp experiments/data/deepfluoro/$SUBJECT/warp.txt \
    --labels 1 2 3 4 7 \
    --scales 24 12 6 \
    --n-itrs 500 500 500 \
    --patience 10 10 10 \
    --crop 100 \
    --linearize \
    --savepath "$SAVEPATH"
