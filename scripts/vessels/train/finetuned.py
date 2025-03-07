from pathlib import Path
from subprocess import run

import submitit


def main(inpath):
    subject = str(Path(inpath).parent).split("/")[-1]
    ckptpath = sorted(Path("models/vessels/base").glob("*.pth"))[-1]

    command = f"""
    xvr train \
        -i {inpath} \
        -o models/vessels/finetuned/{subject}
        -c {ckptpath} \
        --project xvr-vessels
    """
    command = command.strip().split()
    run(command, check=True)


if __name__ == "__main__":
    inpath = sorted(Path("data/ljubljana").glob("**/volume.nii.gz"))

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="xvr-vessels-finetune",
        gpus_per_node=1,
        mem_gb=43.5,
        slurm_array_parallelism=len(inpath),
        slurm_partition="polina-a6000",
        slurm_qos="vision-polina-main",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, inpath)
