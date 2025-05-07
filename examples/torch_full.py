"""
We may want to dynamically switch between differen optimizers.

In this script, we can select between `--opt ADAM` and `--opt SGD`.
Selecting an optimizer flag along with --help shows corresponding arguments.
By default, help for SGD is shown.

usage: torch_full.py [-h] [--model.in_channels int] [--model.out_channels int] [--opt {SGD,ADAM}] [--opt.dampening float] [--opt.lr float] [--opt.momentum float] [--opt.nesterov bool]
                     [--opt.weight_decay float]

options:
  -h, --help            show this help message and exit
  --model.in_channels int
  --model.out_channels int
  --opt {SGD,ADAM}
  --opt.dampening float
  --opt.lr float
  --opt.momentum float
  --opt.nesterov bool
  --opt.weight_decay float
"""

from dataclasses import dataclass

import torch.nn as nn
from torch.optim import SGD as SGD_
from torch.optim import Adam as Adam_

from parsonaut import CheckpointMixin, Choices, Lazy, Parsable


class Model(nn.Module, CheckpointMixin):
    def __init__(
        self,
        # Use str, int, float, bool, or tuple of those
        in_channels: int = 4,
        out_channels: int = 2,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)


class SGD(SGD_, CheckpointMixin):
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


class Adam(Adam_, CheckpointMixin):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


# We create an ENUM of possible optimizer configurations
class Optimizer(Choices):
    SGD = SGD.as_lazy()
    ADAM = Adam.as_lazy()


@dataclass
class Params(Parsable):
    model: Lazy[Model, ...] = Model.as_lazy()
    opt: Optimizer = Optimizer.SGD  # here we selet a default optimizer option


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

# We can now do model training etc...
# Finally, we can call model.to_checkpoint and opt.to_checkpoint
