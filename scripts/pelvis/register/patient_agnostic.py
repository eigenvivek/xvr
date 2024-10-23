from pathlib import Path
from subprocess import run

import submitit


def main(subject_id):
    model = sorted(Path("models/pelvis/patient_agnostic").glob("**/*.pth"))[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register \
        data/deepfluoro/subject{subject_id:02d}/xrays \
        -v data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}.nii.gz \
        -m data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}_mask.nii.gz \
        -c {model} \
        -o results/deepfluoro/register/patient_agnostic/subject{subject_id:02d}/{epoch} \
        --crop 100 \
        --linearize \
        --labels 1,2,3,4,7 \
        --scales 24,12,6 \
        --reverse_x_axis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    subject_ids = range(1, 7)

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-register-agnostic",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=6,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_ids)
