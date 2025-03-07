from pathlib import Path
from subprocess import run

import submitit


def main(subject_id):
    model = list(Path("models/vessels/base").glob("*1000.pth"))[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    diffpose register \
        data/ljubljana/subject{subject_id:02d}/xrays \
        -v data/ljubljana/subject{subject_id:02d}/volume.nii.gz \
        -c {model} \
        -o results/ljubljana/register/zeroshot/{subject_id}/{epoch} \
        --linearize \
        --subtract_background \
        --scales 15,7.5,5 \
        --pattern *[!_max].dcm
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    subject_ids = list(range(1, 11))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="vessel-eval-diffpose",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=len(subject_ids),
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_ids)
