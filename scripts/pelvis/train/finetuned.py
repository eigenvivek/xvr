from pathlib import Path
from subprocess import run

import submitit


def main(inpath):
    dir = Path(__file__).parents[3]

    subject = str(Path(inpath).parent).split("/")[-1]
    subject_id = int(subject[-2:])
    ckptpath = sorted(Path(dir / "models/pelvis/patient_agnostic").glob("*.pth"))[-1]

    command = f"""
    xvr finetune \
        -i {dir / inpath} \
        -o {dir}/models/pelvis/finetuned/{subject} \
        -c {ckptpath} \
        --name deepfluoro{subject_id:02d} \
        --project xvr-pelvis
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    inpath = sorted(Path("data/deepfluoro").glob("**/volume.nii.gz"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-pelvis-finetune",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_array_parallelism=len(inpath),
        slurm_partition="polina-a6000",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, inpath)
