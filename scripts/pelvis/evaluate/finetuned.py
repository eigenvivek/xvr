from pathlib import Path
from subprocess import run

import submitit


def main(model):
    dir = Path(__file__).parents[3]

    subject_id = str(model.parent).split("/")[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        {dir}/data/deepfluoro/{subject_id}/xrays \
        -v {dir}/data/deepfluoro/{subject_id}/volume.nii.gz \
        -c {dir / model} \
        -o {dir}/results/deepfluoro/evaluate/finetuned/{subject_id}/{epoch} \
        --crop 100 \
        --linearize \
        --init_only \
        --verbose 0
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    models = sorted(Path("models/pelvis/finetuned/").glob("**/*.pth"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-eval-finetuned",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=6,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, models)
