from pathlib import Path
from subprocess import run

import submitit


def main(ckptpath):
    subject_id = str(ckptpath.parent).split("/")[-1]
    epoch = ckptpath.stem.split("_")[-1]

    command = f"""
    xvr register model \
        data/ljubljana/{subject_id}/xrays \
        -v data/ljubljana/{subject_id}/volume.nii.gz \
        -c {ckptpath} \
        -o results/ljubljana/evaluate/finetuned/{subject_id}/{epoch} \
        --linearize \
        --subtract_background \
        --pattern *[!_max].dcm \
        --init_only \
        --verbose 0
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    ckptpath = Path("models/vessels/finetuned").rglob("*.pth")

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-eval-finetuned",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=-1,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, ckptpath)
