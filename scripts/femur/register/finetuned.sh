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
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Registration flags
if [[ "$SLURM_ARRAY_TASK_ID" == "4" ]]; then
    LINEARIZE_FLAG=""
    SUBTRACT_BACKGROUND_FLAG=""
    EQUALIZE_FLAG=""
else
    LINEARIZE_FLAG="--linearize"
    SUBTRACT_BACKGROUND_FLAG="--subtract_background"
    EQUALIZE_FLAG="--equalize"
fi

# Run the registration
source /data/vision/polina/users/vivekg/xvr/.venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr register model \
    data/femur/$SUBJECT/xrays \
    -v data/femur/$SUBJECT/volume.nii.gz \
    -m data/femur/$SUBJECT/mask_all.nii.gz \
    -c models/femur/finetuned/$SUBJECT/0001.pth \
    -o results/femur/register/finetuned/$SUBJECT \
    --labels 1,2,3,4 \
    --crop 20 \
    $LINEARIZE_FLAG \
    $SUBTRACT_BACKGROUND_FLAG \
    $EQUALIZE_FLAG \
    --scales 16,8,4 \
    --n_itrs 500,250,100 \
    --warp data/femur/$SUBJECT/warp2template.txt

for FILE in data/femur/$SUBJECT/xrays/*.dcm; do
    XRAY=$(basename "$FILE" .dcm)
    xvr register restart \
        "$FILE" \
        -v data/femur/$SUBJECT/volume.nii.gz \
        -m data/femur/$SUBJECT/mask_all.nii.gz \
        -c results/femur/register/finetuned/$SUBJECT/$XRAY/parameters.pt \
        -o results/femur/register/finetuned_restart/$SUBJECT \
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
