#!/bin/bash
#SBATCH --job-name=xvr-register-femur-foundation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-5
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=05:00:00

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

CKPT=experiments/models/wbct/model.pth

# subject04's x-rays are already linearized, so they get no intensity preprocessing
if [[ "$SLURM_ARRAY_TASK_ID" == "4" ]]; then
    PP="--no-linearize --no-subtract-background --no-equalize"
    PP_RESTART="--no-linearize --no-subtract-background --equalize"
    CROP_RESTART=0
else
    PP="--linearize --subtract-background --equalize"
    PP_RESTART="$PP"
    CROP_RESTART=1
fi

for PAIR in "foundation:" "foundation_antipodal:--antipodal"; do
    NAME="${PAIR%%:*}"
    ANTIPODAL="${PAIR#*:}"
    OUTDIR=experiments/results/femur/$NAME/$SUBJECT
    RESTART_OUTDIR=experiments/results/femur/${NAME}_restart/$SUBJECT
    rm -rf "$OUTDIR" "$RESTART_OUTDIR"
    mkdir -p "$OUTDIR" "$RESTART_OUTDIR"

    xvr register model \
        --files experiments/data/femur/$SUBJECT/xrays/*.dcm \
        --ckpt "$CKPT" \
        --imagepath experiments/data/femur/$SUBJECT/volume.nii.gz \
        --labelpath experiments/data/femur/$SUBJECT/mask.nii.gz \
        --warp experiments/data/femur/$SUBJECT/warp.txt \
        --labels 1 2 3 4 \
        --scales 24 12 6 \
        --n-itrs 500 500 500 \
        --patience 10 10 10 \
        --crop 20 \
        $PP \
        $ANTIPODAL \
        --savepath "$OUTDIR"

    for FILE in experiments/data/femur/$SUBJECT/xrays/*.dcm; do
        STEM=$(basename "$FILE" .dcm)
        xvr register restart \
            --files "$FILE" \
            --ckpt "$OUTDIR/$STEM.pth" \
            --imagepath experiments/data/femur/$SUBJECT/volume.nii.gz \
            --labelpath experiments/data/femur/$SUBJECT/mask.nii.gz \
            --orientation AP \
            --scales 4 2 \
            --n-itrs 250 100 \
            --patience 10 10 \
            --lr-rot 1e-3 \
            --lr-xyz 1e-1 \
            --crop $CROP_RESTART \
            $PP_RESTART \
            --savepath "$RESTART_OUTDIR"
    done
done
