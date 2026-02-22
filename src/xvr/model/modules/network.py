import timm
import torch
from jaxtyping import Float
from nanodrr.geometry import Parameterization, convert


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
        unit_conversion_factor=1000.0,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = Parameterization(parameterization).dim

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

        # E.g., if 1000.0, converts output from meters to millimeters
        self.unit_conversion_factor = unit_conversion_factor

    def forward(self, x: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B 4 4"]:
        x = self.backbone(x)
        rot = self.rot_regression(x)
        xyz = self.unit_conversion_factor * self.xyz_regression(x)
        return convert(
            rot,
            xyz,
            parameterization=self.parameterization,
            convention=self.convention,
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
        unit_conversion_factor=config.get("unit_conversion_factor", 1.0),
    ).cuda()
    model.load_state_dict(model_state_dict)
    model.eval()

    if meta:
        return model, config, ckpt["date"]
    else:
        return model, config
