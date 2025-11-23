from collections import OrderedDict
from importlib.metadata import version

import click

from .commands.animate import animate
from .commands.dcm2nii import dcm2nii
from .commands.register import dicom, fixed, model
from .commands.restart import restart
from .commands.train import train


# Taken from https://stackoverflow.com/a/58323807
class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        attrs["context_settings"] = {
            # "max_content_width": 120,
            "help_option_names": ["-h", "--help"],
        }
        super().__init__(name, commands, **attrs)
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=OrderedGroup)
def register():
    """
    Use gradient-based optimization to register XRAY to a CT/MR.

    XRAY can be a space-separated list of DICOM files or a directory.
    """


register.add_command(model)
register.add_command(dicom)
register.add_command(fixed)


@click.group(cls=OrderedGroup)
@click.version_option(version("xvr"), "--version", "-v")
@click.pass_context
def cli(ctx):
    """
    A PyTorch package for 2D/3D XRAY to CT/MR registration.

    Provides functionality for rapidly training pose regression models and
    registering clinical data with gradient-based iterative optimization.
    """


cli.add_command(train)
cli.add_command(restart)
cli.add_command(register)
cli.add_command(animate)
cli.add_command(dcm2nii)
