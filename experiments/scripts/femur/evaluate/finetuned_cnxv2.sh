#!/bin/bash
#SBATCH --job-name=xvr-eval-femur-finetuned-cnxv2
#SBATCH --output=logs/femur_eval_finetuned_cnxv2_%A_%a.out
#SBATCH --error=logs/femur_eval_finetuned_cnxv2_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia_rtx_a6000|nvidia_rtx_6000_ada_generation"
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

mkdir -p logs

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source .venv/bin/activate

CKPT=experiments/models/femur/finetuned_cnxv2/$SUBJECT.pth
R1=experiments/results/femur/finetuned_cnxv2/$SUBJECT
R2=experiments/results/femur/finetuned_cnxv2_restart/$SUBJECT
mkdir -p "$R1" "$R2"

if [ "$SLURM_ARRAY_TASK_ID" == "4" ]; then
    PP="--no-linearize --no-subtract-background --no-equalize"
else
    PP="--linearize --subtract-background --equalize"
fi

echo "Subject:  $SUBJECT"
echo "Ckpt:     $CKPT"
echo "Preproc:  $PP"

xvr register model \
    --files data/femur/$SUBJECT/xrays/*.dcm \
    --ckpt "$CKPT" \
    --imagepath data/femur/$SUBJECT/volume.nii.gz \
    --labelpath data/femur/$SUBJECT/mask.nii.gz \
    --labels 1 2 3 4 \
    --scales 16 8 4 \
    --n-itrs 500 250 100 \
    --patience 15 10 5 \
    --crop 20 \
    $PP \
    --savepath "$R1"

for FILE in data/femur/$SUBJECT/xrays/*.dcm; do
    STEM=$(basename "$FILE" .dcm)
    xvr register restart \
        --files "$FILE" \
        --ckpt "$R1/$STEM.pth" \
        --imagepath data/femur/$SUBJECT/volume.nii.gz \
        --labelpath data/femur/$SUBJECT/mask.nii.gz \
        --labels 1 2 3 4 \
        --orientation AP \
        --scales 4 2 \
        --n-itrs 250 100 \
        --patience 10 5 \
        --lr-rot 1e-3 \
        --lr-xyz 1e-1 \
        --crop 20 \
        $PP \
        --savepath "$R2"
done
