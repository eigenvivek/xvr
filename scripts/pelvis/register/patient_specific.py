from pathlib import Path
from subprocess import run

import submitit


def main(model):
    subject_id = str(model.parent).split("/")[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register \
        data/deepfluoro/{subject_id}/xrays \
        -v data/deepfluoro/{subject_id}/volume.nii.gz \
        -m data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id[-2:]}_mask.nii.gz \
        -c {model} \
        -o results/deepfluoro/register/patient_specific/{subject_id}/{epoch} \
        --crop 100 \
        --linearize \
        --labels 1,2,3,4,7 \
        --scales 24,12,6 \
        --reverse_x_axis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    models = list(Path("models/pelvis/patient_specific").glob("**/*1000.pth"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-register-specific",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=12,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, models)
