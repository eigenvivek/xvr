from subprocess import run

import submitit


def main():
    command = """
    xvr train \
        -i data/totalcta/imgs_registered \
        -o models/skull/patient_agnostic \
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
        --n_epochs 251 \
        --name totalcta \
        --project xvr-skull
    """
    command = command.strip().split()
    run(command, check=True)

    # Restart training at 2X scale
    command = """
    xvr restart \
        -c models/skull/patient_agnostic/totalcta_0250.pth \
        --rescale 2 \
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
        slurm_partition="A6000",
        slurm_exclude="sumac,fennel",
        timeout_min=10_000,
    )
    jobs = executor.submit(main)
