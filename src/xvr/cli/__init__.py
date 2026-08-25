from cyclopts import App

from ._help import formatter
from .register import register
from .restart import restart
from .train import train

xvr = App(name="xvr", help_formatter=formatter)


xvr.command(train)
xvr.command(restart)
xvr.command(register)


def main():
    xvr()
