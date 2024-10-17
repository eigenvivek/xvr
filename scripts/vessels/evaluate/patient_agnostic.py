from pathlib import Path
from subprocess import run

import submitit


def main(ckptpath):
    for subject_id in range(1, 11):
        command = f"""
        xvr register \
            data/ljubljana/subject{subject_id:02d}/xrays \
            -v data/ljubljana/subject{subject_id:02d}/volume.nii.gz \
            -c {ckptpath} \
            -o results/ljubljana/evaluate/patient_agnostic/subject{subject_id:02d}/{ckptpath.stem.split('_')[-1]} \
            --linearize \
            --subtract_background \
            --invert \
            --pattern *[!_max].dcm \
            --model_only
        """
        command = command.strip().split()
        run(command, check=True)


if __name__ == "__main__":
    ckptpath = Path("models/vessels/patient_agnostic").glob("*.pth")
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-eval-agnostic",
        gpus_per_node=1,
        mem_gb=10.0,
        slurm_array_parallelism=6,
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, ckptpath)
