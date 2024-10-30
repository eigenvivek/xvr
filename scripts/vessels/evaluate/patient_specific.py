from pathlib import Path
from subprocess import run

import submitit


def main(ckptpath):
    *_, subject, epoch = str(ckptpath).split("/")
    epoch = epoch.split("_")[-1].split(".")[0]

    command = f"""
    xvr register model \
        data/ljubljana/{subject}/xrays \
        -v data/ljubljana/{subject}/volume.nii.gz \
        -c {ckptpath} \
        -o results/ljubljana/evaluate/patient_specific/{subject}/{epoch} \
        --linearize \
        --subtract_background \
        --invert \
        --pattern *[!_max].dcm \
        --init_only
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    ckptpath = Path("models/vessels/patient_specific").rglob("*.pth")
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-eval-specific",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=6,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, ckptpath)
