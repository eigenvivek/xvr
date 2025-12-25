from pathlib import Path
from subprocess import run

import submitit


def main():
    dir = Path(__file__).parents[3]

    command = f"""
    xvr train \
        -i {dir}/data/totalcta/imgs_registered \
        -o {dir}/models/skull/patient_agnostic \
        --r1 -125.0 125.0 \
        --r2 -30.0 30.0 \
        --r3 -15.0 15.0 \
        --tx -50.0 50.0 \
        --ty -800.0 -700.0 \
        --tz -150.0 150.0 \
        --sdd 1000.0 \
        --height 128 \
        --delx 2.0 \
        --reverse_x_axis \
        --pretrained \
        --n_epochs 2000 \
        --name totalcta \
        --project xvr-skull
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-skull-agnostic",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_partition="polina-a6000",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.submit(main)
