from pathlib import Path

import click
import pandas as pd
import torch
from diffdrr.pose import RigidTransform
from tqdm import tqdm

from xvr.metrics import Evaluator
from xvr.renderer import initialize_drr

DATA = Path("experiments/data")
RESULTS = Path("experiments/results")

MASKS = {"deepfluoro": "mask.nii.gz", "femur": "mask.nii.gz", "ljubljana": None}

# Exclude four views with questionable ground truth pose parameters:
# https://github.com/rg2/DeepFluoroLabeling-IPCAI2020/issues/8
EXCLUDE = {
    ("deepfluoro", "subject01", "003"),
    ("deepfluoro", "subject01", "050"),
    ("deepfluoro", "subject04", "002"),
    ("deepfluoro", "subject04", "004"),
}


def load_subject(
    dataset: str,
    subject: str,
    device: str,
) -> tuple[object, torch.Tensor]:
    """Load a subject's CT volume and fiducials, reused across all of its x-rays."""
    mask = MASKS[dataset]
    drr = initialize_drr(
        str(DATA / dataset / subject / "volume.nii.gz"),
        str(DATA / dataset / subject / mask) if mask else None,
        None,
        "AP",
        *(100, 100, 1000.0, 1.0, 1.0, 0.0, 0.0),
        False,
        "trilinear",
        device=device,
    )
    fiducials = torch.load(
        DATA / dataset / subject / "fiducials.pt", weights_only=False
    )
    return drr, fiducials.to(device)


def initialize_evaluator(
    drr,
    fiducials: torch.Tensor,
    intrinsics: dict,
) -> Evaluator:
    """Apply one x-ray's intrinsics to the cached DRR and rebuild its Evaluator."""
    drr.set_intrinsics_(**intrinsics)
    return Evaluator(drr, fiducials)


def read_true(
    dataset: str,
    subject: str,
    xray: str,
    device: str,
) -> tuple[RigidTransform, dict]:
    """Read one x-ray's ground-truth pose and stored intrinsics."""
    ckpt = torch.load(
        DATA / dataset / subject / "xrays" / f"{xray}.pt", weights_only=False
    )
    true_pose = RigidTransform(ckpt["pose"].to(torch.float32)).to(device)
    return true_pose, ckpt["intrinsics"]


def read_pred(
    model_path: Path,
    final_path: Path,
    device: str,
) -> tuple[RigidTransform, float, RigidTransform, float, float]:
    """Read the initial pose from the model stage and the final pose from the final stage."""
    model = torch.load(model_path, weights_only=False)
    init_pose = RigidTransform(model["init_pose"].to(torch.float32)).to(device)
    ncc_init = model["trajectory"]["ncc"].iloc[0].item()
    runtime = float(model["runtime"])

    if final_path == model_path:
        final = model
    else:
        final = torch.load(final_path, weights_only=False)
        runtime += float(final["runtime"])
    final_pose = RigidTransform(final["final_pose"].to(torch.float32)).to(device)
    ncc_final = final["trajectory"]["ncc"].iloc[-1].item()

    return init_pose, ncc_init, final_pose, ncc_final, runtime


def final_stage(root: Path) -> Path:
    """The restart directory of a two-stage run, or the run itself if it has only one stage."""
    restart = root.parent / f"{root.name}_restart"
    return restart if restart.is_dir() else root


def hemispheres(root: Path) -> list[Path]:
    """The runs to choose between: a foundation prediction and its antipode, else just the run.

    The foundation model's prediction and its antipode (a ~180 deg C-arm flip) are equally valid
    initializations, so `register/foundation.sh` optimizes both. Ground truth is not available at
    inference, so the winner is whichever reached the higher final image similarity (NCC).
    """
    antipodal = root.parent / f"{root.name}_antipodal"
    return [root, antipodal] if antipodal.is_dir() else [root]


@click.command()
@click.option("--dataset", required=True, type=click.Choice(list(MASKS)))
@click.option(
    "--result", required=True, help="folder under experiments/results/<dataset>/"
)
@click.option(
    "--main", "savepath", default=str(RESULTS / "registration.csv"), type=click.Path()
)
@click.option("--device", default="cpu")
def main(dataset, result, savepath, device):
    """Score a registration run's initial and final poses into the metrics CSV."""
    root = RESULTS / dataset / result
    runs = hemispheres(root)

    results = []
    cached = None
    n_antipode, n_incomplete = 0, 0
    for model_path in tqdm(sorted(root.glob("subject*/*/parameters.pt"))):
        subject, xray = model_path.parent.parent.name, model_path.parent.name
        if (dataset, subject, xray) in EXCLUDE:
            continue

        preds = [
            read_pred(
                model_path, final_stage(root) / subject / xray / "parameters.pt", device
            )
        ]
        for run in runs[1:]:
            counterpart = run / subject / xray / "parameters.pt"
            if not counterpart.exists():
                break
            final_path = final_stage(run) / subject / xray / "parameters.pt"
            preds.append(read_pred(counterpart, final_path, device))
        if len(preds) < len(runs):
            n_incomplete += 1
            continue

        # Ties keep the raw prediction, since max() returns the first maximal element
        init_pose, ncc_init, final_pose, ncc_final, runtime = max(
            preds, key=lambda p: p[3]
        )
        n_antipode += ncc_final != preds[0][3]

        true_pose, intrinsics = read_true(dataset, subject, xray, device)
        if cached != subject:
            drr, fiducials, cached = *load_subject(dataset, subject, device), subject
        evaluator = initialize_evaluator(drr, fiducials, intrinsics)

        for estimate, pred_pose, ncc, elapsed in [
            ("init", init_pose, ncc_init, 0.0),
            ("final", final_pose, ncc_final, runtime),
        ]:
            mpe, mrpe, mtre, dgeo = evaluator(true_pose, pred_pose)
            results.append(
                {
                    "dataset": dataset,
                    "result": result,
                    "subject": subject,
                    "xray": xray,
                    "pose": estimate,
                    "ncc": ncc,
                    "runtime": elapsed,
                    "mPE": mpe,
                    "mRPE": mrpe,
                    "mTRE": mtre,
                    "dGeo": dgeo,
                }
            )

    if len(runs) > 1:
        n = len(results) // 2
        print(
            f"{dataset}/{result}: raw prediction kept for {n - n_antipode}, antipode for {n_antipode}"
        )
    if n_incomplete:
        print(
            f"WARNING: skipped {n_incomplete} x-rays missing a counterpart in {runs[-1]}"
        )

    df = pd.DataFrame(results)
    out = Path(savepath)
    if out.exists():
        old = pd.read_csv(out, dtype={"subject": str, "xray": str})
        df = pd.concat(
            [old[~((old.dataset == dataset) & (old.result == result))], df],
            ignore_index=True,
        )
    df.sort_values(["dataset", "result", "subject", "xray", "pose"]).to_csv(
        out, index=False
    )
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
