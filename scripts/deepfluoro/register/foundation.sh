#!/bin/bash
#SBATCH --job-name=xvr-ttopt-deepfluoro-foundation
#SBATCH --output=logs/xvr_deepfluoro_foundation_ttopt_%A_%a.out
#SBATCH --error=logs/xvr_deepfluoro_foundation_ttopt_%A_%a.err
#SBATCH --array=1-6
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the registration
source /data/vision/polina/users/vivekg/xvr/.venv/bin/activate

SUBJECT=subject$(printf "%02d" $SLURM_ARRAY_TASK_ID)

xvr register model \
    data/deepfluoro/$SUBJECT/xrays \
    -v data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c models/wbct/model.pth \
    -o results/deepfluoro/register/foundation/$SUBJECT \
    --crop 100 \
    --linearize \
    --labels 1,2,3,4,7 \
    --scales 24,12,6 \
    --n_itrs 500,500,500 \
    --warp data/deepfluoro/$SUBJECT/warp2template.txt

xvr register model \
    data/deepfluoro/$SUBJECT/xrays \
    -v data/deepfluoro/$SUBJECT/volume.nii.gz \
    -m data/deepfluoro/$SUBJECT/mask.nii.gz \
    -c models/wbct/model.pth \
    -o results/deepfluoro/register/foundation_antipodal/$SUBJECT \
    --crop 100 \
    --linearize \
    --labels 1,2,3,4,7 \
    --scales 24,12,6 \
    --n_itrs 500,500,500 \
    --warp data/deepfluoro/$SUBJECT/warp2template.txt \
    --antipodal
