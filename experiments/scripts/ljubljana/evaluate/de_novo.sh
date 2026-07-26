#!/bin/bash
#SBATCH --job-name=xvr-eval-ljubljana-de-novo
#SBATCH --output=logs/ljubljana_eval_de_novo_%A_%a.out
#SBATCH --error=logs/ljubljana_eval_de_novo_%A_%a.err
#SBATCH --array=1-10
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

CKPT=experiments/models/ljubljana/de_novo/$SUBJECT.pth
SAVEPATH=experiments/results/ljubljana/de_novo/$SUBJECT
mkdir -p "$SAVEPATH"

echo "Subject:  $SUBJECT"
echo "Ckpt:     $CKPT"
echo "Savepath: $SAVEPATH"

xvr register model \
    --files experiments/data/ljubljana/$SUBJECT/xrays/*[!_max].dcm \
    --ckpt "$CKPT" \
    --imagepath experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    --scales 16 8 4 \
    --n-itrs 500 500 500 \
    --patience 10 10 10 \
    --linearize \
    --subtract-background \
    --savepath "$SAVEPATH"
