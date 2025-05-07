"""
We showcase how to configure a simple experiment:

    1. Define a torch Model
    2. Define an optimizer
    3. Configure both in a single CLI

usage: torch_simple.py [-h] [--model.in_channels int] [--model.out_channels int] [--opt.dampening float] [--opt.lr float] [--opt.momentum float]
                       [--opt.nesterov bool] [--opt.weight_decay float]

options:
  -h, --help            show this help message and exit
  --model.in_channels int
  --model.out_channels int
  --opt.dampening float
  --opt.lr float
  --opt.momentum float
  --opt.nesterov bool
  --opt.weight_decay float
"""

from dataclasses import dataclass

import torch.nn as nn
from torch.optim import SGD as SGD_

from parsonaut import CheckpointMixin, Lazy, Parsable


# Use CheckpointMixin for torch classes that
# expose the state_dict API
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


# Non-torch objects can use Parsable instead
@dataclass
class Params(Parsable):
    # Calling as_lazy is like doing a nested partial of the class __init__ function
    model: Lazy[Model, ...] = Model.as_lazy()
    opt: Lazy[SGD, ...] = SGD.as_lazy(
        lr=1.0,  # we can override some defaults here
    )


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
