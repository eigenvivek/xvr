from subprocess import run

import submitit


def main():
    command = """
    xvr train \
        -i data/ctpelvic1k/imgs_registered \
        -o models/pelvis/patient_agnostic \
        --r1 -45.0 45.0 \
        --r2 -45.0 45.0 \
        --r3 -15.0 15.0 \
        --tx -150.0 150.0 \
        --ty -1000.0 -450.0 \
        --tz -150.0 150.0 \
        --sdd 1020.0 \
        --height 128 \
        --delx 2.1764375 \
        --reverse_x_axis \
        --pretrained \
        --n_epochs 1000 \
        --name ctpelvic1k \
        --project xvr-pelvis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-agnostic",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_partition="A6000",
        slurm_exclude="sumac,fennel",
        timeout_min=10_000,
    )
    jobs = executor.submit(main)
