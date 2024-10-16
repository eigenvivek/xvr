from subprocess import run

import submitit


def main():
    command = """
    xvr train \
        -i data/nitrc_mras \
        -o models/vessels/patient_agnostic \
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
        --n_epochs 251 \
        --name nitrc \
        --project xvr-vessels
    """
    command = command.strip().split()
    run(command, check=True)

    # Restart training at 2X scale
    command = """
    xvr restart \
        -c models/vessels/patient_agnostic/nitrc_0250.pth \
        --rescale 2 \
        --project xvr-nitrc
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-agnostic",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_partition="A6000",
        slurm_exclude="sumac,fennel",
        timeout_min=10_000,
    )
    jobs = executor.submit(main)
