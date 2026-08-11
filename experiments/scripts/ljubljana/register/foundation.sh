#!/bin/bash
#SBATCH --job-name=xvr-ttopt-ljubljana-foundation
#SBATCH --output=logs/xvr_ljubljana_foundation_ttopt_%A_%a.out
#SBATCH --error=logs/xvr_ljubljana_foundation_ttopt_%A_%a.err
#SBATCH --array=1-10
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

OUTDIR=experiments/results/ljubljana/foundation/$SUBJECT
ANTIPODAL_OUTDIR=experiments/results/ljubljana/foundation_antipodal/$SUBJECT

rm -rf "$OUTDIR" "$ANTIPODAL_OUTDIR"
mkdir -p "$OUTDIR" "$ANTIPODAL_OUTDIR"

# --pattern skips the *_max.dcm maximum-intensity projections, which are not registered
xvr register model \
    experiments/data/ljubljana/$SUBJECT/xrays \
    -v experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    -c experiments/models/wbct.pth \
    -o $OUTDIR \
    --linearize \
    --subtract_background \
    --scales 16,8,4 \
    --n_itrs 500,500,500 \
    --pattern '*[!_max].dcm' \
    --warp experiments/data/ljubljana/$SUBJECT/warp.txt

xvr register model \
    experiments/data/ljubljana/$SUBJECT/xrays \
    -v experiments/data/ljubljana/$SUBJECT/volume.nii.gz \
    -c experiments/models/wbct.pth \
    -o $ANTIPODAL_OUTDIR \
    --linearize \
    --subtract_background \
    --scales 16,8,4 \
    --n_itrs 500,500,500 \
    --pattern '*[!_max].dcm' \
    --warp experiments/data/ljubljana/$SUBJECT/warp.txt \
    --antipodal
