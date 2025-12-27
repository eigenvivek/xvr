from pathlib import Path

import click
import pandas as pd
import torch
from diffdrr.pose import RigidTransform
from tqdm import tqdm

from xvr.metrics import Evaluator
from xvr.renderer import initialize_drr


def initialize_evaluator(
    dataset: str,
    subject: str,
    intrinsics: dict,
    voxel_shift: float = 0.0,
) -> Evaluator:
    # Load the subject
    volume = Path(f"data/{dataset}/{subject}/volume.nii.gz")
    mask = Path(f"data/{dataset}/{subject}/mask.nii.gz")
    if not mask.exists():
        mask = None

    # Initialize a DRR
    drr = initialize_drr(
        volume,
        mask,
        None,
        "AP",
        *(100, 100, 1000.0, 1.0, 1.0, 0.0, 0.0),
        False,
        "trilinear",
        drr_kwargs=dict(voxel_shift=voxel_shift),
        device="cpu",
    )
    drr.set_intrinsics_(**intrinsics)

    # Return the evaluator
    fiducials = torch.load(f"data/{dataset}/{subject}/fiducials.pt")
    evaluator = Evaluator(drr, fiducials)
    return evaluator


def read_true(dataset: str, subject: str, xray: str) -> tuple[RigidTransform, dict]:
    filename = f"data/{dataset}/{subject}/xrays/{xray}.pt"
    ckpt = torch.load(filename)
    true_pose = RigidTransform(ckpt["pose"])
    if dataset == "deepfluoro":
        mapper = RigidTransform(
            torch.tensor(
                [
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
        )
        true_pose = mapper.compose(true_pose)
    return true_pose, ckpt["intrinsics"]


def read_pred(
    filename: Path,
) -> tuple[RigidTransform, float, RigidTransform, float, float]:
    ckpt = torch.load(filename, weights_only=False)
    pred_pose_init = RigidTransform(ckpt["init_pose"])
    pred_pose_final = RigidTransform(ckpt["final_pose"])
    ncc = ckpt["trajectory"]["ncc"]
    ncc_init = ncc.iloc[0].item()
    ncc_final = ncc.iloc[-1].item()
    runtime = ckpt["runtime"]
    return pred_pose_init, ncc_init, pred_pose_final, ncc_final, runtime


def process_filenames(filenames: tuple[Path, ...]) -> pd.DataFrame:
    metaparams = []
    for filename in filenames:
        *_, dataset, _, partition, subject, xray, _ = str(filename).split("/")
        metaparams.append([filename, dataset, partition, subject, xray])
    df = pd.DataFrame(
        metaparams, columns=["filename", "dataset", "partition", "subject", "xray"]
    )
    df = df.sort_values(by=["dataset", "subject", "xray"], ignore_index=True)
    return df


@click.command()
@click.option("-f", "--filepath", type=click.Path(exists=True))
@click.option("-s", "--savepath", type=click.Path())
def main(filepath, savepath):
    filenames = sorted(Path(filepath).rglob("parameters.pt"))
    df = process_filenames(filenames)

    results = []
    old_key = None
    for _, (filename, dataset, partition, subject, xray) in tqdm(
        df.iterrows(), total=len(df)
    ):
        current_key = (dataset, subject, xray)
        if current_key != old_key:
            true_pose, intrinsics = read_true(dataset, subject, xray)
            evaluator = initialize_evaluator(dataset, subject, intrinsics)
            old_key = (dataset, subject, xray)

        pred_pose_init, ncc_init, pred_pose_final, ncc_final, runtime = read_pred(
            filename
        )
        mpd, mrpe, mtre, dgeo = evaluator(true_pose, pred_pose_init)
        results.append(
            [
                dataset,
                subject,
                xray,
                partition,
                "initial",
                ncc_init,
                0.0,
                mpd,
                mrpe,
                mtre,
                dgeo,
            ]
        )
        mpd, mrpe, mtre, dgeo = evaluator(true_pose, pred_pose_final)
        results.append(
            [
                dataset,
                subject,
                xray,
                partition,
                "final",
                ncc_final,
                runtime,
                mpd,
                mrpe,
                mtre,
                dgeo,
            ]
        )

    results = pd.DataFrame(
        results,
        columns=[
            "dataset",
            "subject",
            "xray",
            "partition",
            "estimate",
            "ncc",
            "runtime",
            "mpd",
            "mrpe",
            "mtre",
            "dgeo",
        ],
    )
    results.to_csv(savepath, index=False)


if __name__ == "__main__":
    main()
