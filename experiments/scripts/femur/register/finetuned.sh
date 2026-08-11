#!/bin/bash
#SBATCH --job-name=xvr-ttopt-femur-finetuned
#SBATCH --output=logs/xvr_femur_finetuned_ttopt_%A_%a.out
#SBATCH --error=logs/xvr_femur_finetuned_ttopt_%A_%a.err
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

# subject04's x-rays are already linearized, so they get no intensity preprocessing
if [[ "$SLURM_ARRAY_TASK_ID" == "4" ]]; then
    LINEARIZE_FLAG=""
    SUBTRACT_BACKGROUND_FLAG=""
    EQUALIZE_FLAG=""
else
    LINEARIZE_FLAG="--linearize"
    SUBTRACT_BACKGROUND_FLAG="--subtract_background"
    EQUALIZE_FLAG="--equalize"
fi

OUTDIR=experiments/results/femur/finetuned/$SUBJECT
RESTART_OUTDIR=experiments/results/femur/finetuned_restart/$SUBJECT

rm -rf "$OUTDIR" "$RESTART_OUTDIR"
mkdir -p "$OUTDIR" "$RESTART_OUTDIR"

xvr register model \
    experiments/data/femur/$SUBJECT/xrays \
    -v experiments/data/femur/$SUBJECT/volume.nii.gz \
    -m experiments/data/femur/$SUBJECT/mask.nii.gz \
    -c experiments/models/femur/finetuned/$SUBJECT.pth \
    -o $OUTDIR \
    --labels 1,2,3,4 \
    --crop 20 \
    $LINEARIZE_FLAG \
    $SUBTRACT_BACKGROUND_FLAG \
    $EQUALIZE_FLAG \
    --scales 16,8,4 \
    --n_itrs 500,250,100 \
    --warp experiments/data/femur/$SUBJECT/warp.txt

for FILE in experiments/data/femur/$SUBJECT/xrays/*.dcm; do
    XRAY=$(basename "$FILE" .dcm)
    xvr register restart \
        "$FILE" \
        -v experiments/data/femur/$SUBJECT/volume.nii.gz \
        -m experiments/data/femur/$SUBJECT/mask.nii.gz \
        -c $OUTDIR/$XRAY/parameters.pt \
        -o $RESTART_OUTDIR \
        --orientation AP \
        --crop 20 \
        $LINEARIZE_FLAG \
        $SUBTRACT_BACKGROUND_FLAG \
        $EQUALIZE_FLAG \
        --scales 4,2 \
        --n_itrs 250,100 \
        --lr_rot 1e-3 \
        --lr_xyz 1e-1
done
