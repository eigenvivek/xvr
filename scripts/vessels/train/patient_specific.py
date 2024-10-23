from subprocess import run

import submitit


def main(subject_id):
    command = f"""
    xvr train \
        -i data/ljubljana/subject{subject_id:02d}/volume.nii.gz \
        -o models/vessels/patient_specific/subject{subject_id:02d} \
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
        --n_epochs 1000 \
        --batch_size 16 \
        --name ljubljana{subject_id:02d} \
        --project xvr-vessels
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    subject_id = list(range(1, 11))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-specific",
        gpus_per_node=1,
        mem_gb=12.0,
        slurm_array_parallelism=len(subject_id),
        slurm_partition="2080ti",
        slurm_exclude="",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_id)
