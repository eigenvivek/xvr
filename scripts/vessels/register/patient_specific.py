from pathlib import Path
from subprocess import run

import submitit


def main(model):
    subject_id = str(model.parent).split("/")[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        data/ljubljana/{subject_id}/xrays \
        -v data/ljubljana/{subject_id}/volume.nii.gz \
        -c {model} \
        -o results/ljubljana/register/patient_specific/{subject_id}/{epoch} \
        --linearize \
        --subtract_background \
        --scales 15,7.5,5 \
        --pattern *[!_max].dcm
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    models = list(Path("models/vessels/patient_specific").glob("**/*1000.pth"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-register-specific",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=10,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, models)
