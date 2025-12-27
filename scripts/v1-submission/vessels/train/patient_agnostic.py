from pathlib import Path
from subprocess import run

import submitit


def main():
    dir = Path(__file__).parents[3]

    command = f"""
    xvr train \
        -i {dir}/data/nitrc_mras \
        -o {dir}/models/vessels/patient_agnostic \
        --r1 -45.0 90.0 \
        --r2 -5.0 5.0 \
        --r3 -5.0 5.0 \
        --tx -25.0 25.0 \
        --ty 700 800.0 \
        --tz -25.0 25.0 \
        --sdd 1250.0 \
        --height 128 \
        --delx 2.31 \
        --orientation AP \
        --pretrained \
        --n_epochs 2000 \
        --name nitrc \
        --project xvr-vessels
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-agnostic",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_partition="polina-a6000",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.submit(main)
