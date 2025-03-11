from pathlib import Path
from subprocess import run

import submitit


def main(subject_id):
    dir = Path(__file__).parents[3]

    model = sorted(Path(dir / "models/vessels/patient_agnostic").glob("*1100.pth"))[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        {dir}/data/ljubljana/subject{subject_id:02d}/xrays \
        -v {dir}/data/ljubljana/subject{subject_id:02d}/volume.nii.gz \
        -c {model} \
        -o {dir}/results/ljubljana/register/patient_agnostic/subject{subject_id:02d}/{epoch} \
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
        name="xvr-vessels-register-agnostic",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=10,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_ids)
