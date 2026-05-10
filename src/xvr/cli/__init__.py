from cyclopts import App

from .register import register
from .restart import restart
from .train import train

xvr = App()


xvr.command(train)
xvr.command(restart)
xvr.command(register)


def main():
    xvr()
