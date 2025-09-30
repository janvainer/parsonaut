import re
import sys
from argparse import SUPPRESS, Action
from argparse import ArgumentParser as _ArgumentParser
from argparse import ArgumentTypeError
from collections import defaultdict
from types import SimpleNamespace

from parsonaut.lazy import TYPE_NAME, Choices, Lazy
from parsonaut.typecheck import (
    Missing,
    get_flat_tuple_inner_type,
    is_bool_type,
    is_flat_tuple_type,
    is_float_type,
    is_int_type,
    is_optional_single_type,
    is_str_type,
)

BOOL_TRUE_FLAGS = ("yes", "true", "t", "y", "1")
BOOL_FALSE_FLAGS = ("no", "false", "f", "n", "0")


class ArgumentParser(_ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.lazy_roots = list()
        self.args = dict()
        self.aliases = dict()
        self.choices = defaultdict(list)
        self.choices_defaults = dict()
        # We allow add_options without dest if it is the only source of
        # args.
        self._lazy_without_dest = False

        super().__init__(*args, **kwargs)

    def add_options(self, lzy: Lazy, dest: str | None = None):
        assert (
            not self._lazy_without_dest
        ), "Cannot add lazy options without a destination name if other lazy was already added."
        if dest is not None:
            assert "." not in dest
            assert dest not in self.lazy_roots
            self.lazy_roots.append(dest)
        else:
            assert set(self.args.keys()) == {
                "--help"
            }, "Cannot add lazy options without a destination name if other args are present"

        prefix = f"{dest}." if dest is not None else ""
        self._add_options(lzy, prefix=prefix)
        if dest is None:
            self._lazy_without_dest = True

    def _add_options(self, lzy: Lazy, prefix: str = ""):
        self.add_argument(f"--{prefix}_class", default=lzy.cls, help=SUPPRESS)
        for k, (typ, value) in sorted(lzy.signature.items()):
            if Lazy.is_lazy_type(typ):
                if isinstance(value, Choices):
                    self.add_argument(
                        f"--{prefix}{k}",
                        type=str,
                        choices=[e.name for e in type(value)],
                        default=value.name,
                    )
                    self.choices_defaults[f"{prefix}{k}"] = value.name
                    for e in type(value):
                        self.choices[f"{prefix}{k}"].append(e.name)
                        # Add a [] marker to highlight choice values.
                        # We use the marks later to trim the choices.
                        self._add_options(e, prefix=f"{prefix}{k}.[{e.name}].")
                else:
                    self._add_options(value, prefix=f"{prefix}{k}.")
            else:
                self.add_option(f"{prefix}{k}", value, typ)

    def add_option(self, name, value, typ):
        assert isinstance(name, str)
        assert not self._lazy_without_dest

        name = f"--{name}"
        check_val = value if value is not Missing else None
        required = False

        # Optional args not supported for tuples yet
        is_optional, typ = is_optional_single_type(typ, None)

        # bool
        if is_bool_type(typ):
            self.add_argument(
                name,
                type=str2bool,
                default=value if value is not Missing else None,
                metavar=f"{typ.__name__}",
                required=required,
                # For int | None and similar, the user can specify --foo without a value.
                # In such case, the const=None value is used.
                nargs="?" if is_optional else None,
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
                required=required,
                nargs="?" if is_optional else None,
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
                required=required,
                action=collect_as(tuple),
            )
        else:
            raise

    def add_argument(self, *name_or_flags, **kwargs):
        assert not self._lazy_without_dest
        if len(name_or_flags) == 2:
            alias, name = name_or_flags
            assert alias not in self.aliases
            self.aliases[name] = alias
        elif len(name_or_flags) == 1:
            (name,) = name_or_flags

        self.args[name] = kwargs

    def parse_args(self, args=None):  # noqa: C901
        from collections import defaultdict

        args = sys.argv[1:] if args is None else args

        # Choices trimming:
        # Check if user provided a specific value for a choice and trim the other options
        choices = sorted(
            self.choices.items(), key=lambda x: len(x[0].split(".")), reverse=True
        )
        for k, v in choices:
            # If yes, read it from the command line
            prefix = re.sub(r"\[.*?\]\.", "", f"--{k}")
            if prefix in args:
                position = args.index(prefix)
                assert (
                    len(args) > position + 1
                ), f"Expected a value after the choice {prefix}"
                val = args[args.index(prefix) + 1]
                assert (
                    val in v
                ), f"error: argument {prefix}: invalid choice '{val}' (choose from {', '.join(v)})"
            else:
                val = self.choices_defaults[k]

            # We remove choice options:
            # - not selected by the user
            # - not default
            for choice in v:
                if choice != val:
                    remove_prefix = f"--{k}.[{choice}]"
                    for key in self.args.copy():
                        if key.startswith(remove_prefix):
                            del self.args[key]

        for k in self.args.copy():
            if "[" in k:
                kk = re.sub(r"\[.*?\]\.", "", k)
                self.args[kk] = self.args[k]
                del self.args[k]

        for name in self.args:
            if name in self.aliases:
                arg = (self.aliases[name], name)
            else:
                arg = (name,)
            super().add_argument(*arg, **self.args[name])

        args = super().parse_args(args)

        # we can build the Lazy objects from the recursive dicts
        args_dict = vars(args)
        args_grouped = defaultdict(dict)
        for k, v in args_dict.items():
            # Do not add choices names
            if k in self.choices:
                continue
            if not k.startswith(tuple(self.lazy_roots)):
                args_grouped[k] = v
            else:
                root = [root for root in self.lazy_roots if k.startswith(root)]
                assert len(root) == 1
                root = root[0]
                args_grouped[root][k.split(f"{root}.", 1)[-1]] = v

        args_grouped = dict(args_grouped)
        if self.lazy_roots:
            for root in self.lazy_roots:
                args_grouped[root] = Lazy.from_dict(args_grouped[root])

            return SimpleNamespace(**args_grouped)
        elif TYPE_NAME in args_grouped:
            return Lazy.from_dict(args_grouped)
        else:
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
