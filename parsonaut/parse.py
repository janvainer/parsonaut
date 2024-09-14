import sys
from argparse import SUPPRESS, Action
from argparse import ArgumentParser as _ArgumentParser
from argparse import ArgumentTypeError
from types import SimpleNamespace

from parsonaut.lazy import Choices, Lazy
from parsonaut.typecheck import (
    Missing,
    get_flat_tuple_inner_type,
    is_bool_type,
    is_flat_tuple_type,
    is_float_type,
    is_int_type,
    is_str_type,
)

BOOL_TRUE_FLAGS = ("yes", "true", "t", "y", "1")
BOOL_FALSE_FLAGS = ("no", "false", "f", "n", "0")


class ArgumentParser(_ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_roots = list()

    def add_options(self, lzy: Lazy, dest: str):
        assert "." not in dest
        assert dest not in self.lazy_roots
        self.lazy_roots.append(dest)
        self._add_options(lzy, prefix=f"{dest}.")

    def _add_options(self, lzy: Lazy, prefix: str = ""):
        self.add_argument(f"--{prefix}_class", default=lzy.cls, help=SUPPRESS)
        for k, (typ, value) in sorted(lzy.signature.items()):
            if Lazy.is_lazy_type(typ):
                if isinstance(value, Choices):
                    if f"--{prefix}{k}" in sys.argv:
                        val = sys.argv[sys.argv.index(f"--{prefix}{k}") + 1]
                        value = type(value)[val]

                    self.add_argument(
                        f"--{prefix}{k}",
                        type=str,
                        choices=[e.name for e in type(value)],
                        default=value.name,
                    )
                self._add_options(value, prefix=f"{prefix}{k}.")
            else:
                self.add_option(f"{prefix}{k}", value, typ)

    def add_option(self, name, value, typ):
        assert isinstance(name, str)

        name = f"--{name}"
        check_val = value if value is not Missing else None
        required = False

        # bool
        if is_bool_type(typ):
            self.add_argument(
                name,
                type=str2bool,
                default=value if value is not Missing else None,
                metavar=f"{typ.__name__}",
                help="Parameter descripion.",
                required=required,
            )
        # int | float | str
        elif (
            is_int_type(typ, check_val)
            or is_str_type(typ, check_val)
            or is_float_type(typ, check_val)
        ):
            self.add_argument(
                name,
                type=typ,
                default=value,
                metavar=f"{typ.__name__}",
                help="Parameter descripion.",
                required=required,
            )
        # tuple[bool | int | float |str , ...]
        elif is_flat_tuple_type(typ, check_val):
            subtyp, nitems = get_flat_tuple_inner_type(typ)
            nargs = "*" if nitems == -1 else nitems
            if nargs == "*":
                metavar = f"{subtyp.__name__},"
            else:
                metavar = f"{subtyp.__name__}"

            self.add_argument(
                name,
                nargs=nargs,
                metavar=metavar,
                type=subtyp if subtyp != bool else str2bool,
                default=tuple(value) if value is not Missing else None,
                help="Parameter descripion loong this is a very loong description.",
                required=required,
                action=collect_as(tuple),
            )
        else:
            raise

    def parse_args(self, args=None):
        from collections import defaultdict

        args = super().parse_args(args)
        args_dict = vars(args)
        args_grouped = defaultdict(dict)
        for k, v in args_dict.items():
            if not k.startswith(tuple(self.lazy_roots)):
                args_grouped[k] = v
            else:
                root = [root for root in self.lazy_roots if k.startswith(root)]
                assert len(root) == 1
                root = root[0]
                args_grouped[root][k.split(f"{root}.", 1)[-1]] = v

        args_grouped = dict(args_grouped)
        for root in self.lazy_roots:
            args_grouped[root] = Lazy.from_dict(args_grouped[root])

        return SimpleNamespace(**args_grouped)


def collect_as(coll_type):
    class Collect_as(Action):
        def __call__(self, parser, namespace, values, options_string=None):
            setattr(namespace, self.dest, coll_type(values))

    return Collect_as


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in BOOL_TRUE_FLAGS:
        return True
    elif v.lower() in BOOL_FALSE_FLAGS:
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
