#!/bin/bash
#SBATCH --job-name=xvr-ttopt-ljubljana-de-novo
#SBATCH --output=logs/xvr_ljubljana_de_novo_ttopt_%A_%a.out
#SBATCH --error=logs/xvr_ljubljana_de_novo_ttopt_%A_%a.err
#SBATCH --array=1-10
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
    data/ljubljana/$SUBJECT/xrays \
    -v data/ljubljana/$SUBJECT/volume.nii.gz \
    -c models/ljubljana/de_novo/$SUBJECT/0030.pth \
    -o results/ljubljana/register/de_novo/$SUBJECT \
    --linearize \
    --subtract_background \
    --scales 16,8,4,2 \
    --n_itrs 500,500,500,100 \
    --pattern *[!_max].dcm
