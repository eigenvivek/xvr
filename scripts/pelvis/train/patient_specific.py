from subprocess import run

import submitit


def main(subject_id):
    command = f"""
    xvr train \
        -i data/deepfluoro/subject{subject_id:02d}/volume.nii.gz \
        -o models/pelvis/patient_specific/subject{subject_id:02d} \
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
        --n_epochs 500 \
        --name deepfluoro{subject_id:02d} \
        --project xvr-pelvis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    subject_id = list(range(1, 7))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-specific",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_array_parallelism=len(subject_id),
        slurm_partition="A6000",
        slurm_exclude="sumac,fennel",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_id)
