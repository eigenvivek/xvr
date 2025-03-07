from pathlib import Path
from subprocess import run

import submitit


def main(ckptpath):
    dir = Path(__file__).parents[3]

    subject_id = str(ckptpath.parent).split("/")[-1]
    epoch = ckptpath.stem.split("_")[-1]

    command = f"""
    xvr register model \
        {dir}/data/ljubljana/{subject_id}/xrays \
        -v {dir}/data/ljubljana/{subject_id}/volume.nii.gz \
        -c {dir / ckptpath} \
        -o {dir}/results/ljubljana/evaluate/patient_specific/{subject_id}/{epoch} \
        --linearize \
        --subtract_background \
        --invert \
        --pattern *[!_max].dcm \
        --init_only \
        --verbose 0
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
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, ckptpath)
