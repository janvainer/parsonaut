<div align="center">
<img src="parsonaut.jpg" width="100" height="100" alt="Parsonaut"/>
</div>

# Parsonaut

Auto-configure (not only) torch experiments from the CLI.


Parsonaut makes your experiments
1. **Configurable** - configure any parameter of your experiment from CLI
2. **Reproducible** - easily store your full experiment configuration to disk
3. **Boilerplate-free** - make model checkpointing seampless

## Quickstart

### Installation

To install the library, clone the repository and use `pip`:

```bash
pip install git+https://github.com/janvainer/parsonaut.git
```

### Usage

Let's supercharge a simple torch experiment with automatic CLI configuration
```python
"""
usage: script.py [-h] [--in_channels int] [--out_channels int]

options:
  -h, --help        show this help message and exit
  --in_channels int
  --out_channels int
"""
import torch.nn as nn
from parsonaut import Parsable


class Model(nn.Module, Parsable):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
    ):
        super().__init__()


# Parse user CLI args - a partially initialized model
partial_model = Model.parse_args()

# Serialize model configuration
partial_model.to_file("model_config.yaml")
```
Now we can do some training. We instantiate the model configuration into a torch model.

```python
model = partial_model.to_eager()

# Training code here ...
```

Finally, serialize model configuration AND weights.
```python
model.to_checkpoint("ckpt_dir")
```
We can now load the experiment configuration and model weights later:

```python
model_with_weights = Model.from_checkpoint("ckpt_dir")
just_config = Model.from_file("model_config.yaml")
```

### Config files

Every parser accepts `--config`, so the file you saved above can be fed straight
back in:

```bash
python script.py --config model_config.yaml --in_channels 8
```

The file populates the tree; the command line then overrides individual leaves.
Keys are the command line flags without the leading dashes, so either style works:

```yaml
seed: 7
model.in_channels: 16     # flat, mirroring --model.in_channels
opt:                      # or nested
  lr: 0.5
```

Only the keys you care about have to be present, and a key that does not match
any argument is an error rather than a typo that silently does nothing. A
`_class` tag may name the class already at that node, or a subclass of it. Pass
`ArgumentParser(config_flag="--cfg")` to rename the flag, or `config_flag=None`
to remove it.

### Nested configurations

Annotate a parameter with the class you want to configure. `as_lazy()` is like a
nested `functools.partial`: it records the arguments without building anything.

```python
from dataclasses import dataclass

@dataclass
class Params(Parsable):
    model: Model = Model.as_lazy()
    opt: SGD = SGD.as_lazy(lr=1.0)


hp = Params.parse_args()            # --model.in_channels, --opt.lr, ...
model = hp.model.to_eager()
opt = hp.opt.to_eager(params=model.parameters())
```

A configuration and a real object are interchangeable: every factory is typed as
the class itself, and `to_eager`, `copy`, `to_dict` and `to_file` all work
whichever of the two you are holding.

```python
bigger = hp.model.copy({"in_channels": 8})   # a new config
bigger.to_eager()                            # ... built
model.copy({"in_channels": 8})               # same, from a built object
```

`copy` always returns a configuration, never a duplicate of a live object, so
building the result is a separate step.

`as_lazy()` on an instance hands back the configuration itself, which is how the
rest of the `Lazy` API stays reachable:

```python
hp.opt.as_lazy().cls          # which class the node builds
model.as_lazy().signature
```

Pick the class in Python before building the tree. A complete experiment is in
[examples/torch_simple.py](examples/torch_simple.py).
