from pathlib import Path
from subprocess import run

import submitit


def main(subject_id):
    dir = Path(__file__).parents[3]

    model = sorted(Path(dir / "models/pelvis/patient_agnostic").glob("**/*905.pth"))[-1]
    epoch = model.stem.split("_")[-1]

    command = f"""
    xvr register model \
        {dir}/data/deepfluoro/subject{subject_id:02d}/xrays \
        -v {dir}/data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}.nii.gz \
        -m {dir}/data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}_mask.nii.gz \
        -c {model} \
        -o {dir}/results/deepfluoro/register/patient_agnostic/subject{subject_id:02d}/{epoch} \
        --warp {dir}/data/ctpelvic1k/combined_subset_registered_deepfluoro/deepfluoro_{subject_id:02d}_reoriented0GenericAffine.mat \
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
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, subject_ids)
