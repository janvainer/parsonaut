from parsonaut.parsable import Lazy, Parsable, parser_from_cli_args


def compute(z: int = 4):
    return z + 1



class Submodel(Parsable):
    def __init__(self, y: int = 5, z: tuple[int, ...] = (1, 2)) -> None:
        super().__init__()


class Model(Parsable):
    def __init__(self, x: int = 1, sub: Lazy[Submodel, ...] = Submodel.as_lazy()) -> None:
        self.sub = sub


x = Model.as_lazy()
print(x.to_dict())

u = {
    "x": 2,
    "sub": {
        "y": 4
    },
}
x = x.update(**u)
print(x.to_dict())
x = x.update(x=3, sub={"y": 3})
print(x.to_dict(include_type=True))
x = Lazy.from_dict(x.to_dict(include_type=True))
print(x.to_dict())
args = parser_from_cli_args(x.to_cli_args()).parse_args()
for arg in vars(args).items():
    print(arg)


# model_lazy = Model.parse_args()
# model = model_lazy.to_eager()
    
# More granular
parser = Model.argument_parser()
parser.add_argument("") # some non-model argument - eg number of gpus or maybe learning rate
args = parser.parse_args()
model = Model.from_args(args)
model.load_state_dict(args.ckpt_path)

# more high-level
model_eager = Model.parse_args()

parser = ArgumentParser()
Model.add_arguments(parser)