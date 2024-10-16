from collections import OrderedDict

import click

from .commands import animate, finetune, register, restart, train


# Taken from https://stackoverflow.com/a/58323807
class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super().__init__(name, commands, **attrs)
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=OrderedGroup)
def cli():
    """
    xvr is a command-line interface for training, fine-tuning, and performing 2D/3D X-ray to CT registration with pose regression models.
    """


cli.add_command(train)
cli.add_command(restart)
cli.add_command(finetune)
cli.add_command(register)
cli.add_command(animate)
