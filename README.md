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


# Parse user CLI args - ta partially initialized model
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

Parsonaut allows configuring multiple possibly nested classes.
Moreover, you can dynamically select which classes to use via enums.

To explore more advanced features, please see the following tutorials:
- [zero](examples/torch_simple.py) - simple, but complete experiment
- [hero](examples/torch_full.py) - dynamic and nested configurations
