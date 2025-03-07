from pathlib import Path
from subprocess import run

import submitit


def main(model):
    subject_id = str(model.parent).split("/")[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        data/deepfluoro/{subject_id}/xrays \
        -v data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}.nii.gz \
        -m data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id[-2:]}_mask.nii.gz \
        -c {model} \
        -o results/deepfluoro/register/finetuned/{subject_id}/{epoch} \
        --crop 100 \
        --linearize \
        --labels 1,2,3,4,7 \
        --scales 24,12,6 \
        --reverse_x_axis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    models = list(Path("models/deepfluoro/finetuned").glob("**/*.pth"))
    
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-register-finetuned",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=12,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, models)
