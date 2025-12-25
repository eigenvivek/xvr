from pathlib import Path
from subprocess import run

import submitit


def main(model):
    dir = Path(__file__).parents[3]

    subject_id = str(model.parent).split("/")[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        {dir}/data/ljubljana/{subject_id}/xrays \
        -v {dir}/data/ljubljana/{subject_id}/volume.nii.gz \
        -c {dir / model} \
        -o {dir}/results/ljubljana/register/finetuned/{subject_id}/{epoch} \
        --linearize \
        --subtract_background \
        --scales 15,7.5,5 \
        --pattern *[!_max].dcm
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    models = list(Path("models/vessels/finetuned").glob("**/*8.pth"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-register-finetuned",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=10,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, models)
