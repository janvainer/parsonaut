<div align="center">
<img src="parsonaut.jpg" width="100" height="100" alt="Parsonaut"/>
</div>

# Parsonaut

Auto-configure your experiments from the CLI.

## Features

- **Command-Line Integration**: Expose class parameters in a CLI.
- **Partial Configuration**: Partially initialize objects before full instantiation.
- **Nested Configurations**: Support for nested class-based configurations.
- **Dynamic Class Selection**: Choose which class to use from the CLI.
- **Configuration Serialization**: Load and save your configuration to disk.

## Quickstart

### Installation

To install the library, clone the repository and use `pip`:

```bash
pip install git+https://github.com/janvainer/parsonaut.git
```

### Basic usage

Let's create a simple experiment with a parsonaut CLI.
```python
from dataclasses import dataclass
from parsonaut import Lazy, Parsable, ArgumentParser


class Model(Parsable):
    def __init__(
        self,
        # Use str, int, float, bool, or tuple of those
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        pass

    def parameters(self):
        return "dummy_params"


class Optimizer(Parsable):
    def __init__(
        self,
        params,  # untyped or unknown types are ignored
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.9),
    ):
        pass


# Parsable objects can be nested
@dataclass
class Params(Parsable):
    # Calling as_lazy is like doing a nested partial of the class __init__ function
    model: Lazy[Model, ...] = Model.as_lazy()
    opt: Lazy[Optimizer, ...] = Optimizer.as_lazy(
        lr=1.0, # we can override some defaults here
    )

# Expose the configuration in the CLI
parser = ArgumentParser()
parser.add_options(Params.as_lazy())
hp = parser.parse_args()
print(hp)
```

Now calling `python my_script.py --help` gives us argparse-like experience.

```bash
usage: my_script.py [-h] [--model.dropout float] [--model.num_layers int] [--opt.betas float float] [--opt.lr float]

options:
  -h, --help            show this help message and exit
  --model.dropout float
  --model.num_layers int
  --opt.betas float float
  --opt.lr float
```
If we run the script, we can check how the configuration looks like.
Let's try `python my_script.py --opt.lr 20`

```python
# This is just a configuration, not the actual objects
Params(
    model=Model(
        dropout=0.1,
        num_layers=4,
    ),
    opt=Optimizer(
        betas=(0.9, 0.9),
        lr=20.0,  # Yay, our config has changed!
    ),
)
```
And finally, we can add a few lines to instantiate the actual objects from the configs.


```python
# Materialize the configurations into actual objects
model: Model = hp.model.to_eager()

# Fill in required params that were not specified on CLI or in code
opt: Optimizer = hp.opt.to_eager(params=model.parameters())
print(model)
print(opt)
```

And running the script should give us initialized objects.

```bash
<__main__.Model object at 0x1034a1c10>
<__main__.Optimizer object at 0x1034a1e50>
```

### Choices
Parsonaut allows for dynamic class selection.
For example, we can have two models and decide which one should be used.

```python
from dataclasses import dataclass
from parsonaut import Parsable, ArgumentParser, Choices


class Model1(Parsable):
    def __init__(
        self,
        # Use str, int, float, bool, or tuple of those
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        pass


class Model2(Parsable):
    def __init__(
        self,
        # Use str, int, float, bool, or tuple of those
        num_layers2: int = 4,
        dropout2: float = 0.1,
    ):
        pass

class Models(Choices):
    MODEL1 = Model1.as_lazy()
    MODEL2 = Model2.as_lazy()

@dataclass
class Params(Parsable):
    model: Models = Models.MODEL1

parser = ArgumentParser()
parser.add_options(Params.as_lazy())
hp = parser.parse_args()
print(hp)
```

Now if we invoke help, we can see that there is a `--model` flag asking us to select which model type we want to use.

```bash
usage: my_script.py [-h] [--model {MODEL1,MODEL2}] [--model.dropout float] [--model.num_layers int]

options:
  -h, --help            show this help message and exit
  --model {MODEL1,MODEL2}
  --model.dropout float
  --model.num_layers int
```

Notice, that the help shows the params for the first model.
We can check the args for the second model by mentioning it along with help. Calling `python my_script.py --model MODEL2 --help` invokes

```bash
usage: my_script.py [-h] [--model {MODEL1,MODEL2}] [--model.dropout2 float] [--model.num_layers2 int]

options:
  -h, --help            show this help message and exit
  --model {MODEL1,MODEL2}
  --model.dropout2 float
  --model.num_layers2 int
```

Now we can call the script like `python my_script.py --model MODEL2`, and we are going to select the second model instead of the defaul.

```bash
Params(
    model=Model2(
        dropout2=0.1,
        num_layers2=4,
    ),
)
```
