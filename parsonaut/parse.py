import sys
from argparse import SUPPRESS, Action
from argparse import ArgumentParser as _ArgumentParser
from argparse import ArgumentTypeError
from collections import defaultdict
from dataclasses import dataclass, field
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
        self.lazy_roots = list()
        self.args = dict()
        self.aliases = dict()
        self.choices = defaultdict(list)
        self.choices_defaults = dict()
        super().__init__(*args, **kwargs)

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
                    self.add_argument(
                        f"--{prefix}{k}",
                        type=str,
                        choices=[e.name for e in type(value)],
                        default=value.name,
                    )
                    self.choices_defaults[f"{prefix}{k}"] = value.name
                    for e in type(value):
                        self.choices[f"{prefix}{k}"].append(e.name)
                        self._add_options(e, prefix=f"{prefix}{k}.{e.name}.")
                else:
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
                required=required,
                action=collect_as(tuple),
            )
        else:
            raise

    def add_argument(self, *name_or_flags, **kwargs):
        if len(name_or_flags) == 2:
            alias, name = name_or_flags
            assert alias not in self.aliases
            self.aliases[name] = alias
        elif len(name_or_flags) == 1:
            (name,) = name_or_flags

        self.args[name] = kwargs

    def parse_args(self, args=None):  # noqa: C901
        from collections import defaultdict

        short_to_full = shorten_flags(self.args.keys())
        full_to_short = {v: k for k, v in short_to_full.items()}

        # Choices trimming:
        # Check if user provided a specific value for a choice and trim the other options
        for k, v in self.choices.items():

            # If yes, read it from the command line
            if (prefix := full_to_short[f"--{k}"]) in sys.argv:
                position = sys.argv.index(prefix)
                assert (
                    len(sys.argv) > position + 1
                ), f"Expected a value after the choice {prefix}"
                val = sys.argv[sys.argv.index(prefix) + 1]
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
                    remove_prefix = f"--{k}.{choice}"
                    keys_to_remove = [
                        (key, skey)
                        for key, skey in full_to_short.items()
                        if key.startswith(remove_prefix)
                    ]
                    for key, skey in keys_to_remove:
                        del full_to_short[key]
                        del short_to_full[skey]
                        del self.args[key]

            # And we have to remove the choice name suffix from the related variables
            for key, skey in full_to_short.items():
                if key.startswith(f"--{k}."):
                    new_key = key.replace(f".{val}", "")
                    new_skey = skey.replace(f"{val}.", "")
                    short_to_full[new_skey] = new_key
                    self.args[new_key] = self.args[key]
                    del self.args[key]
                    del short_to_full[skey]

        if short_to_full.pop("--help") is not None:
            super().add_argument("-h", "--help", **self.args["--help"])

        # Now we pass the shortened and trimmedy flags to the parser
        for arg in short_to_full:
            full_name = short_to_full[arg]
            if full_name in self.aliases:
                arg = (self.aliases[full_name], arg)
            else:
                arg = (arg,)

            super().add_argument(*arg, **self.args[full_name])

        args = super().parse_args(args)

        # We need to map the shortened names back to full names so that
        # we can build the Lazy objects from the recursive dicts
        args_dict = vars(args)
        args_grouped = defaultdict(dict)
        for k, v in args_dict.items():
            k = short_to_full["--" + k].replace("--", "")
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


@dataclass
class Trie:
    children: dict[str, "Trie"] = field(default_factory=dict)
    n_suffixes: int = 0

    def insert(self, x: list[str]):
        if not x:
            return self

        prefix, remainder = x[0], x[1:]
        self.n_suffixes += 1
        if prefix in self.children:
            self.children[prefix].insert(remainder)
        else:
            self.children[prefix] = Trie().insert(remainder)

        return self

    def list_sequences(self):
        prefixes = []
        for prefix, child in self.children.items():
            if child.n_suffixes == 0:
                prefixes.append([prefix])
            else:
                suffixes = child.list_sequences()
                for s in suffixes:
                    prefixes.append([prefix] + s)
        return prefixes

    def find_unique_shortest_prefixes(self):
        prefixes = []
        full_sequences = []

        for prefix, child in self.children.items():
            if child.n_suffixes in (0, 1):
                # Prefix is unique if the subtree has a single leaf
                prefixes.append([prefix])
                # Collect the rest of the subtree
                fs = [[prefix] + seq for seq in child.list_sequences()]
                if fs:
                    full_sequences.extend(fs)
                else:
                    full_sequences.append([prefix])
            else:
                for p, f in zip(*child.find_unique_shortest_prefixes()):
                    prefixes.append([prefix] + p)
                    full_sequences.append([prefix] + f)
        return prefixes, full_sequences


def shorten_flags(args: list[str]) -> dict[str, str]:
    sep = "."
    trie = Trie()
    for arg in args:
        trie.insert(arg.split(sep)[::-1])

    prefixes, full_sequences = trie.find_unique_shortest_prefixes()
    prefixes = [
        ("--" if not p[-1].startswith("--") else "") + ".".join(p[::-1])
        for p in prefixes
    ]
    full_sequences = [".".join(s[::-1]) for s in full_sequences]
    res = {prefix: full for prefix, full in zip(prefixes, full_sequences)}
    return dict(sorted(res.items()))


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
