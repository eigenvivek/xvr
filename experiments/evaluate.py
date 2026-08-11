from pathlib import Path

import click
import pandas as pd
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import DoubleGeodesicSE3
from diffdrr.pose import RigidTransform
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MASKS = {"deepfluoro": "mask.nii.gz", "femur": "mask.nii.gz", "ljubljana": None}

# Exclude four views with questionable ground truth pose parameters:
# https://github.com/rg2/DeepFluoroLabeling-IPCAI2020/issues/8
EXCLUDE = {
    ("deepfluoro", "subject01", "003"),
    ("deepfluoro", "subject01", "050"),
    ("deepfluoro", "subject04", "002"),
    ("deepfluoro", "subject04", "004"),
}


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
    ckpt = torch.load(DATA / dataset / subject / "xrays" / f"{xray}.pt", weights_only=False)
    pose = RigidTransform(ckpt["pose"].to(torch.float32))
    return pose.to(device), ckpt["intrinsics"]


def build_evaluator(dataset, subject, intrinsics, device):
    data = DATA / dataset
    mask = MASKS[dataset]
    subj = read(
        str(data / subject / "volume.nii.gz"),
        str(data / subject / mask) if mask else None,
        None,
        "AP",
    )
    # sdd/delx are placeholders (set_intrinsics_ overwrites them per x-ray); height is fixed.
    drr = DRR(subj, sdd=1000.0, height=100, delx=1.0, renderer="trilinear", reverse_x_axis=False)
    drr = drr.to(device)
    drr.set_intrinsics_(**intrinsics)
    fiducials = torch.load(data / subject / "fiducials.pt", weights_only=False).to(torch.float32)
    return Evaluator(
        drr.to(device), fiducials[None].to(device) if fiducials.ndim == 2 else fiducials.to(device)
    )


def final_stage(root: Path) -> Path:
    """The restart directory of a two-stage run, or the run itself if it has only one stage."""
    restart = root.parent / f"{root.name}_restart"
    return restart if restart.is_dir() else root


def read_pred(run: Path, subject: str, xray: str):
    """Init pose (+ first similarity) from a run's first stage, final pose (+ final similarity)
    from its final stage. The similarity metric is maximized, so higher is better; a run whose
    initial DRR was blank has no optimization log and scores -inf."""
    first_pth = run / subject / f"{xray}.pth"
    final_pth = final_stage(run) / subject / f"{xray}.pth"

    first = torch.load(first_pth, weights_only=False)
    final = first if final_pth == first_pth else torch.load(final_pth, weights_only=False)

    sim_init = first["log"]["losses"][0] if first["log"] is not None else -torch.inf
    sim_final = final["log"]["losses"][-1] if final["log"] is not None else -torch.inf
    return first["init_pose"], sim_init, final["final_pose"], sim_final


@click.command()
@click.option("--dataset", required=True, type=click.Choice(list(MASKS)))
@click.option("--result", required=True, help="folder under experiments/results/<dataset>/")
@click.option("--main", "path", default=str(RESULTS / "registration.csv"), type=click.Path())
@click.option("--device", default="cpu")
def main(dataset, result, path, device):
    """Score a registration run's init/final poses into the main metrics CSV."""
    root = RESULTS / dataset / result

    rows, evaluator, cached = [], None, None
    for pth in tqdm(sorted(root.glob("subject*/*.pth"))):
        subject, xray = pth.parent.name, pth.stem
        if not (DATA / dataset / subject / "xrays" / f"{xray}.pt").exists():
            continue
        if (dataset, subject, xray) in EXCLUDE:
            continue

        init_pose, sim_init, final_pose, sim_final = read_pred(root, subject, xray)

        # Rebuild the DRR when the subject changes
        true_pose, intrinsics = read_true(dataset, subject, xray, device)
        if cached != subject:
            evaluator, cached = build_evaluator(dataset, subject, intrinsics, device), subject

        # Adapt the DRR's intrinsics to the X-ray's intrinsics
        evaluator.drr.set_intrinsics_(**intrinsics)
        evaluator.geodesic = DoubleGeodesicSE3(evaluator.drr.detector.sdd, eps=0.0)

        # Compute errors
        for pose, mat, sim in [("init", init_pose, sim_init), ("final", final_pose, sim_final)]:
            mpe, mrpe, mtre, dgeo = evaluator(
                true_pose, RigidTransform(mat.to(torch.float32)).to(device)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "result": result,
                    "subject": subject,
                    "xray": xray,
                    "pose": pose,
                    "ncc": sim,
                    "mPE": mpe,
                    "mRPE": mrpe,
                    "mTRE": mtre,
                    "dGeo": dgeo,
                }
            )

    df = pd.DataFrame(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out, dtype={"subject": str, "xray": str})
        df = pd.concat(
            [old[~((old.dataset == dataset) & (old.result == result))], df], ignore_index=True
        )
    df.sort_values(["dataset", "result", "subject", "xray", "pose"]).to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
