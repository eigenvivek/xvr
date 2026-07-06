from pathlib import Path

import click
import pandas as pd
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import DoubleGeodesicSE3
from diffdrr.pose import RigidTransform
from tqdm import tqdm

MASKS = {"deepfluoro": "mask.nii.gz", "femur": "mask.nii.gz", "ljubljana": None}


class Evaluator:
    """Four 2D/3D registration error metrics (all in mm): mPE, mRPE, mTRE, dGeo."""

    def __init__(self, drr, fiducials):
        self.drr = drr
        self.fiducials = fiducials
        self.geodesic = DoubleGeodesicSE3(drr.detector.sdd, eps=0.0)

    def __call__(self, true_pose, pred_pose):
        x = self.drr.perspective_projection(pred_pose, self.fiducials)
        y = self.drr.perspective_projection(true_pose, self.fiducials)
        mpe = (self.drr.detector.delx * (x - y)).norm(dim=-1).mean(dim=-1)
        mrpe = (
            (self.drr.inverse_projection(pred_pose, x) - self.drr.inverse_projection(true_pose, y))
            .norm(dim=-1)
            .mean(dim=-1)
        )
        mtre = (pred_pose(self.fiducials) - true_pose(self.fiducials)).norm(dim=-1).mean(dim=-1)
        *_, dgeo = self.geodesic(true_pose, pred_pose)
        return torch.stack([mpe, mrpe, mtre, dgeo], dim=-1).squeeze().tolist()


def read_true(dataset, subject, xray, device):
    """Ground-truth pose and stored intrinsics for one x-ray."""
    ckpt = torch.load(f"data/{dataset}/{subject}/xrays/{xray}.pt", weights_only=False)
    pose = RigidTransform(ckpt["pose"].to(torch.float32))
    return pose.to(device), ckpt["intrinsics"]


def build_evaluator(dataset, subject, intrinsics, device):
    data = Path("data") / dataset
    mask = MASKS[dataset]
    subj = read(
        str(data / subject / "volume.nii.gz"),
        str(data / subject / mask) if mask else None,
        None,
        "AP",
    )
    drr = DRR(
        subj, sdd=1000.0, height=100, delx=1.0, renderer="trilinear", reverse_x_axis=False
    ).to(device)
    drr.set_intrinsics_(**intrinsics)
    fiducials = torch.load(data / subject / "fiducials.pt", weights_only=False).to(torch.float32)
    return Evaluator(
        drr.to(device), fiducials[None].to(device) if fiducials.ndim == 2 else fiducials.to(device)
    )


@click.command()
@click.option("--dataset", required=True, type=click.Choice(list(MASKS)))
@click.option("--result", required=True, help="folder under experiments/results/<dataset>/")
@click.option("--main", "path", default="experiments/results/main.csv", type=click.Path())
@click.option("--device", default="cpu")
def main(dataset, result, path, device):
    """Score a registration run's init/final poses into the main metrics CSV."""
    root = Path("experiments/results") / dataset / result
    restart = root.parent / f"{result}_restart"  # femur two-stage: final pose lives here

    rows, evaluator, cached = [], None, None
    for pth in tqdm(sorted(root.glob("subject*/*.pth"))):
        subject, xray = pth.parent.name, pth.stem
        if not Path(f"data/{dataset}/{subject}/xrays/{xray}.pt").exists():
            continue
        true_pose, intrinsics = read_true(dataset, subject, xray, device)
        if cached != subject:
            evaluator, cached = build_evaluator(dataset, subject, intrinsics, device), subject
        final_pth = restart / subject / f"{xray}.pth" if restart.is_dir() else pth
        poses = {
            "init": torch.load(pth, weights_only=False)["init_pose"],
            "final": torch.load(final_pth, weights_only=False)["final_pose"],
        }
        for pose, mat in poses.items():
            mpe, mrpe, mtre, dgeo = evaluator(
                true_pose, RigidTransform(mat.to(torch.float32)).to(device)
            )
            rows.append(
                dict(
                    dataset=dataset,
                    result=result,
                    subject=subject,
                    xray=xray,
                    pose=pose,
                    mPE=mpe,
                    mRPE=mrpe,
                    mTRE=mtre,
                    dGeo=dgeo,
                )
            )

    df = pd.DataFrame(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat(
            [old[~((old.dataset == dataset) & (old.result == result))], df], ignore_index=True
        )
    df.sort_values(["dataset", "result", "subject", "xray", "pose"]).to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
