from pathlib import Path
from subprocess import run

import submitit


def main(ckptpath):
    dir = Path(__file__).parents[3]

    for subject_id in range(1, 7):
        command = f"""
        xvr register model \
            {dir}/data/deepfluoro/subject{subject_id:02d}/xrays \
            -v {dir}/data/ctpelvic1k/deepfluoro/deepfluoro_{subject_id:02d}.nii.gz \
            -c {dir / ckptpath} \
            -o {dir}/results/deepfluoro/evaluate/patient_agnostic/subject{subject_id:02d}/{ckptpath.stem.split("_")[-1]} \
            --crop 100 \
            --linearize \
            --warp {dir}/data/ctpelvic1k/combined_subset_registered_deepfluoro/deepfluoro_{subject_id:02d}_reoriented0GenericAffine.mat \
            --init_only \
            --verbose 0
        """
        command = command.strip().split()
        run(command, check=True)


if __name__ == "__main__":
    ckptpath = Path("models/pelvis/patient_agnostic").glob("*.pth")
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-eval-agnostic",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=6,
        slurm_partition="polina-2080ti",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, ckptpath)
