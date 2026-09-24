"""
Configure a torch experiment from the CLI: model, optimizer, and schedule.

Every script also accepts `--config params.yaml` to read the defaults from a
file; anything passed on the command line still wins.

usage: torch_simple.py [-h] [--config path] [--model.in_channels int] [--model.out_channels int]
                       [--opt.dampening float] [--opt.lr float] [--opt.momentum float]
                       [--opt.nesterov bool] [--opt.weight_decay float] [--schedule.gamma float]
"""

from dataclasses import dataclass

import torch.nn as nn
from torch.optim import SGD as SGD_
from torch.optim.lr_scheduler import ExponentialLR as ExponentialLR_

from parsonaut import Parsable


# Subclass Parsable to make a class configurable.
class Model(nn.Module, Parsable):
    def __init__(
        self,
        # str, int, float, bool, Literal, or a tuple of the basic types
        in_channels: int = 4,
        out_channels: int = 2,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)


class SGD(SGD_, Parsable):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


class ExponentialLR(ExponentialLR_, Parsable):
    def __init__(self, optimizer, gamma: float = 0.9):
        super().__init__(optimizer=optimizer, gamma=gamma)


@dataclass
class Params(Parsable):
    # Annotate a nested config with the class itself. `as_lazy` is like a
    # nested partial init: it remembers the arguments without building anything.
    model: Model = Model.as_lazy()
    opt: SGD = SGD.as_lazy(
        lr=1.0,  # we can override some defaults here
    )
    schedule: ExponentialLR = ExponentialLR.as_lazy()


hp = Params.parse_args()  # expose all params on CLI

# The configuration is completely lazy. Think of it as nested partial inits
print("\nConfiguration: \n")
print(hp)

# Here we instantiate the classes
model = hp.model.to_eager()

print("\nModel: \n")
print(model)

opt = hp.opt.to_eager(params=model.parameters())

print("\nOptimizer: \n")
print(opt)

schedule = hp.schedule.to_eager(optimizer=opt)

print("\nSchedule: \n")
print(f"{type(schedule).__name__}(gamma={schedule.gamma})")
