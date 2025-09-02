import timm
import torch
from diffdrr.pose import convert
from diffdrr.registration import N_ANGULAR_COMPONENTS


class PoseRegressor(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        height=256,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )
        output = self.backbone(torch.randn(1, 1, height, height)).shape[-1]
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)

    def forward(self, x):
        x = self.backbone(x)
        rot = self.rot_regression(x)
        xyz = 1000 * self.xyz_regression(x)  # Convert from meters to millimeters
        return convert(
            rot,
            xyz,
            convention=self.convention,
            parameterization=self.parameterization,
        )


def load_model(ckptpath, meta=False):
    """Load a pretrained pose regression model"""
    ckpt = torch.load(ckptpath, weights_only=False)
    config = ckpt["config"]

    model_state_dict = ckpt["model_state_dict"]
    model = PoseRegressor(
        model_name=config["model_name"],
        parameterization=config["parameterization"],
        convention=config["convention"],
        norm_layer=config["norm_layer"],
        height=config["height"],
    ).cuda()
    model.load_state_dict(model_state_dict)
    model.eval()

    if meta:
        return model, config, ckpt["date"]
    else:
        return model, config
