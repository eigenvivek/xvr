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
        attrs["context_settings"] = {"help_option_names": ["-h", "--help"]}
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
    xvr is a PyTorch package for training, fine-tuning, and performing 2D/3D X-ray to CT/MR registration with pose regression models.
    """


cli.add_command(train)
cli.add_command(restart)
cli.add_command(register)
cli.add_command(animate)
cli.add_command(dcm2nii)
