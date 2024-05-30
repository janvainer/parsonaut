from argparse import Action
from argparse import ArgumentParser as _ArgumentParser
from argparse import ArgumentTypeError

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
    def add_option(self, name, value, typ):
        assert isinstance(name, str)

        name = f"--{name}"
        check_val = value if value is not Missing else None
        required = False
        if is_bool_type(typ):
            self.add_argument(
                name,
                type=str2bool,
                default=value if value is not Missing else None,
                metavar=f"{typ.__name__}",
                help="Parameter descripion.",
                required=required,
            )
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
