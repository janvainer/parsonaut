import sys
from argparse import Action, ArgumentError, ArgumentTypeError, Namespace
from argparse import ArgumentParser as _ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, NoReturn

from .dicts import flatten_dict
from .lazy import Lazy, apply, is_nested_type
from .serialization import load_dict
from .typecheck import (
    BOOL_FALSE_FLAGS as BOOL_FALSE_FLAGS,
)
from .typecheck import (
    BOOL_TRUE_FLAGS as BOOL_TRUE_FLAGS,
)
from .typecheck import (
    Missing,
    coerce_bool,
    get_flat_tuple_inner_type,
    is_basic_type,
    is_flat_tuple_type,
    optional_inner_type,
)


class ArgumentParser(_ArgumentParser):
    """Turns `Lazy` configurations into command line options.

    A `--config path.yaml` option is always available; its values act as
    defaults, so explicit command line arguments still win. Pass
    `config_flag=None` to leave it out, or another flag name to rename it.

    Flags have to be spelled out in full. Pass `allow_abbrev=True` to get
    argparse's prefix matching back.
    """

    def __init__(self, *, config_flag: str | None = "--config", **kwargs: Any):
        # argparse.__init__ creates the default groups via add_argument_group.
        self._config_flag = config_flag
        self._config_dest: str | None = None
        self._lazy: list[tuple[str | None, Lazy]] = []
        self._roots: list[str] = []
        self._lazy_without_dest = False
        self._tree_dests: set[str] = set()
        self._user_defaults: set[str] = set()
        # Plain-argument defaults from before any config file was applied,
        # so a later parse without a file does not keep the previous one.
        self._plain_baseline: dict[str, Any] | None = None
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(**kwargs)

    def add_options(self, config: Any, dest: str | None = None) -> None:
        """Expose a configuration, e.g. `SomeParsable.as_lazy()`, as options."""
        if not isinstance(config, Lazy):
            raise TypeError(
                "Expected a configuration such as `SomeParsable.as_lazy()`, "
                f"got {type(config).__name__}."
            )
        self._reject_after_root_lazy()

        if dest is None:
            if self._lazy or self._plain_dests() or self._user_defaults:
                raise ValueError(
                    "Cannot add lazy options without a destination name if "
                    "other args are present"
                )
        else:
            if "." in dest:
                raise ValueError(f"Destination name cannot contain a dot: {dest!r}")
            if dest in self._roots:
                raise ValueError(f"Duplicate destination name: {dest!r}")
            self._roots.append(dest)

        self._lazy.append((dest, config))
        prefix = f"{dest}." if dest is not None else ""
        _add_lazy(self, config, prefix, self._tree_dests)
        if dest is None:
            self._lazy_without_dest = True

    def add_argument(self, *name_or_flags: str, **kwargs: Any) -> Action:
        self._reject_after_root_lazy()
        return super().add_argument(*name_or_flags, **kwargs)

    def add_argument_group(self, *args: Any, **kwargs: Any):
        self._reject_after_root_lazy()
        return super().add_argument_group(*args, **kwargs)

    def add_mutually_exclusive_group(self, *args: Any, **kwargs: Any):
        self._reject_after_root_lazy()
        return super().add_mutually_exclusive_group(*args, **kwargs)

    def set_defaults(self, **kwargs: Any) -> None:
        self._reject_after_root_lazy()
        self._user_defaults.update(kwargs)
        super().set_defaults(**kwargs)
        if self._plain_baseline is not None:
            for key, value in kwargs.items():
                if key not in self._tree_dests:
                    self._plain_baseline[key] = value

    def parse_args(  # type: ignore[override]
        self, args: Iterable[str] | None = None, namespace: Any = None
    ) -> Lazy | SimpleNamespace | Namespace:
        # The return is a configuration, or a namespace of them, rather than
        # argparse's Namespace. Callers pass any iterable of arguments.
        argv = self._argv(args)
        trees = self._prepare(argv)
        # Call the base parse_known_args so argparse's parse_args does not
        # re-enter our override and treat a Lazy as the namespace.
        ns, extra = _ArgumentParser.parse_known_args(self, argv, namespace)
        if extra:
            self.error(f"unrecognized arguments: {' '.join(extra)}")
        return self._fold(ns, trees, supplied=namespace is not None)

    def parse_known_args(  # type: ignore[override]
        self, args: Iterable[str] | None = None, namespace: Any = None
    ) -> tuple[Lazy | SimpleNamespace | Namespace, list[str]]:
        argv = self._argv(args)
        trees = self._prepare(argv)
        ns, remaining = _ArgumentParser.parse_known_args(self, argv, namespace)
        return self._fold(ns, trees, supplied=namespace is not None), remaining

    def format_help(self, args: Iterable[str] | None = None) -> str:
        self._prepare(list(args) if args else [])
        return _ArgumentParser.format_help(self)

    def print_help(self, file=None) -> None:
        # `file`, not an argument list: argparse calls this as `print_help(stderr)`.
        if file is None:
            file = sys.stdout
        print(self.format_help(), end="", file=file)

    def format_usage(self, args: Iterable[str] | None = None) -> str:
        self._prepare(list(args) if args else [])
        return _ArgumentParser.format_usage(self)

    def print_usage(self, file=None) -> None:
        # `file`, not an argument list: argparse calls this as `print_usage(stderr)`.
        if file is None:
            file = sys.stdout
        print(self.format_usage(), end="", file=file)

    def _prepare(self, argv: list[str]) -> dict[str | None, Lazy]:
        """Apply a config file, sync argparse defaults, return the trees."""
        self._reject_dest_clashes()
        # Register the declared trees before the config flag is added, so a
        # field named like the flag is reported as a clash, and before plain
        # defaults are snapshotted, so those defaults do not include the tree.
        trees = {dest: lzy for dest, lzy in self._lazy}
        self._install_lazy_options(trees)
        self._ensure_config_flag()
        self._restore_plain_defaults()
        path = _scout_config(self, argv)
        if path is not None:
            try:
                fields = flatten_dict(load_dict(path))
            except ValueError as exc:
                _reraise_with_source(path, exc)
            trees = self._apply_config(path, fields, trees)
        # A `_class` tag may have switched a node to a subclass whose fields
        # are not the ones registered above. Rebuild the flags from the tree
        # that is about to be parsed, so new fields can be overridden and
        # dropped fields are not written back.
        self._install_lazy_options(trees)
        for dest, lzy in trees.items():
            prefix = f"{dest}." if dest is not None else ""
            super().set_defaults(**_leaf_defaults(lzy, prefix))
        return trees

    def _install_lazy_options(self, trees: dict[str | None, Lazy]) -> None:
        self._clear_lazy_options()
        for dest, lzy in trees.items():
            prefix = f"{dest}." if dest is not None else ""
            _add_lazy(self, lzy, prefix, self._tree_dests)

    def _clear_lazy_options(self) -> None:
        for action in list(self._actions):
            if action.dest not in self._tree_dests:
                continue
            for option_string in action.option_strings:
                self._option_string_actions.pop(option_string, None)
            self._defaults.pop(action.dest, None)
            # `container` is the group argparse registered the action on. It is
            # not part of the Action stubs, and it is not always this parser.
            getattr(action, "container")._remove_action(action)
        self._tree_dests.clear()

    def _apply_config(
        self,
        path: str,
        fields: dict[str, Any],
        trees: dict[str | None, Lazy],
    ) -> dict[str | None, Lazy]:
        merged = {}
        remaining = dict(fields)
        for dest, lzy in trees.items():
            prefix = f"{dest}." if dest is not None else ""
            own = {
                key.removeprefix(prefix): value
                for key, value in remaining.items()
                if dest is None or key.startswith(prefix)
            }
            for key in own:
                del remaining[f"{prefix}{key}" if dest is not None else key]
            merged[dest] = apply(lzy, own, source=path) if own else lzy

        unknown = sorted(set(remaining) - self._plain_dests())
        if unknown:
            raise ValueError(
                f"{path}: {unknown[0]!r} does not match any argument. Config "
                "file keys are the command line flags without the leading "
                "dashes."
            )
        if remaining:
            super().set_defaults(**remaining)
        return merged

    def _ensure_config_flag(self) -> None:
        if self._config_flag is None or self._config_dest is not None:
            return
        dest = _dest_of(self._config_flag)
        if dest in self._all_dests():
            raise ValueError(
                f"{self._config_flag} clashes with an argument of the same "
                "name. Build the parser with "
                "`ArgumentParser(config_flag='--some-other-name')`, or with "
                "`config_flag=None` to drop the config file option."
            )
        super().add_argument(
            self._config_flag,
            dest=dest,
            default=None,
            metavar="path",
            help="read defaults from a yaml or json config file",
        )
        self._config_dest = dest

    def _restore_plain_defaults(self) -> None:
        """Put plain arguments back to the defaults from before any config file."""
        if self._plain_baseline is None:
            self._plain_baseline = self._snapshot_plain_defaults()
            return
        if self._plain_baseline:
            super().set_defaults(**self._plain_baseline)

    def _snapshot_plain_defaults(self) -> dict[str, Any]:
        defaults = {}
        for action in self._get_optional_actions():
            if (
                action.dest in (self._config_dest, "help")
                or action.dest in self._tree_dests
            ):
                continue
            defaults[action.dest] = action.default
        for key in self._user_defaults:
            if key not in defaults and key not in self._tree_dests:
                defaults[key] = self._defaults.get(key)
        return defaults

    def _plain_dests(self) -> set[str]:
        dests = set(self._user_defaults)
        for action in self._get_optional_actions():
            if (
                action.dest in (self._config_dest, "help")
                or action.dest in self._tree_dests
            ):
                continue
            dests.add(action.dest)
        return dests

    def _all_dests(self) -> set[str]:
        dests = self._plain_dests() | set(self._tree_dests)
        dests.update(self._roots)
        return dests

    def _reject_dest_clashes(self) -> None:
        clashes = sorted(self._plain_dests() & set(self._roots))
        if clashes:
            raise ValueError(
                f"Duplicate destination name: {clashes[0]!r} names both a "
                "plain argument and a configuration added with `add_options`."
            )

    def _fold(self, ns, trees: dict[str | None, Lazy], *, supplied: bool):
        raw = dict(vars(ns))
        flat, grouped = self._partition(raw)
        if self._roots:
            payload = self._grouped_payload(flat, grouped, trees)
            if not supplied:
                return SimpleNamespace(**payload)
            self._write_supplied(ns, raw, payload)
            return ns
        if None in trees:
            return apply(trees[None], flat)
        if not supplied:
            return SimpleNamespace(**flat)
        self._drop_config_dest(ns)
        return ns

    def _partition(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict]]:
        flat: dict[str, Any] = {}
        grouped: dict[str, dict] = defaultdict(dict)
        for key, value in raw.items():
            if key == self._config_dest:
                continue
            root = self._root_of(key)
            if root is None:
                flat[key] = value
            else:
                grouped[root][key[len(root) + 1 :]] = value
        return flat, grouped

    def _grouped_payload(self, flat, grouped, trees) -> dict[str, Any]:
        namespaces = {
            root: apply(trees[root], values) for root, values in grouped.items()
        }
        for root, tree in trees.items():
            namespaces.setdefault(root, tree)
        for key in set(flat) & set(namespaces):
            del flat[key]
        return {**flat, **namespaces}

    def _write_supplied(self, ns, raw: dict[str, Any], payload: dict[str, Any]) -> None:
        for key in raw:
            if key == self._config_dest or self._root_of(key) is not None:
                delattr(ns, key)
        for key, value in payload.items():
            setattr(ns, key, value)

    def _drop_config_dest(self, ns) -> None:
        if self._config_dest is not None and hasattr(ns, self._config_dest):
            delattr(ns, self._config_dest)

    def _reject_after_root_lazy(self) -> None:
        if self._lazy_without_dest:
            raise ValueError(
                "Cannot add more options next to a lazy that was added "
                "without a destination name."
            )

    def _root_of(self, key: str) -> str | None:
        return next((r for r in self._roots if key.startswith(f"{r}.")), None)

    def _argv(self, args: Iterable[str] | None) -> list[str]:
        return list(sys.argv[1:] if args is None else args)

    def _get_optional_actions(self):
        return [a for a in self._actions if a.option_strings]


def _leaf_defaults(lzy: Lazy, prefix: str) -> dict[str, Any]:
    defaults = {}
    for name, (typ, value) in lzy.signature.items():
        flag = f"{prefix}{name}"
        if is_nested_type(typ) and isinstance(value, Lazy):
            defaults.update(_leaf_defaults(value, f"{flag}."))
        else:
            defaults[flag] = value
    return defaults


def _add_lazy(parser: _ArgumentParser, lzy: Lazy, prefix: str, dests: set[str]) -> None:
    for name, (typ, value) in sorted(lzy.signature.items()):
        flag = f"{prefix}{name}"
        if is_nested_type(typ) and isinstance(value, Lazy):
            _add_lazy(parser, value, f"{flag}.", dests)
        else:
            dests.add(flag)
            _add_option(parser, flag, value, typ)


def _add_argument(parser: _ArgumentParser, *args: Any, **kwargs: Any) -> Action:
    """Add an argument without the guard that blocks a second root lazy.

    Rebuilding flags after a config file switches class goes through here.
    `add_options` has already decided that these flags are allowed.
    """
    if isinstance(parser, ArgumentParser):
        if kwargs.get("dest") == parser._config_dest:
            raise ValueError(
                f"{parser._config_flag} clashes with an argument of the same "
                "name. Build the parser with "
                "`ArgumentParser(config_flag='--some-other-name')`, or with "
                "`config_flag=None` to drop the config file option."
            )
        return super(ArgumentParser, parser).add_argument(*args, **kwargs)
    return parser.add_argument(*args, **kwargs)


def _add_option(parser: _ArgumentParser, name: str, value: Any, typ: Any) -> None:
    flag = f"--{name}"
    is_optional, typ = optional_inner_type(typ)
    optional_kwargs: dict[str, Any] = dict(nargs="?", const=None) if is_optional else {}

    if is_basic_type(typ):
        _add_argument(
            parser,
            flag,
            dest=name,
            type=str2bool if typ is bool else typ,
            default=value,
            metavar=typ.__name__,
            help=_help_for(value),
            **optional_kwargs,
        )
    elif is_flat_tuple_type(typ):
        subtyp, nitems = get_flat_tuple_inner_type(typ)
        # A fixed-length optional tuple still has to accept a bare flag, which
        # means "set this to None". argparse only allows that when nargs can
        # be empty, so the length is checked once values are present.
        unbounded = nitems == -1
        nargs = "*" if unbounded or is_optional else nitems
        metavar = f"{subtyp.__name__}," if unbounded else f"{subtyp.__name__}"
        _add_argument(
            parser,
            flag,
            dest=name,
            nargs=nargs,
            metavar=metavar,
            type=subtyp if subtyp is not bool else str2bool,
            default=value if value is Missing or value is None else tuple(value),
            action=_CollectTuple,
            empty=None if is_optional else (),
            expected=nitems if is_optional and not unbounded else -1,
            help=_help_for(value),
        )
    else:
        raise TypeError(
            f"Cannot expose {name} on the command line: {typ!r} is not a "
            "supported option type."
        )


def _help_for(value: Any) -> str | None:
    return "no default" if value is Missing else None


class _CollectTuple(Action):
    def __init__(self, *args: Any, empty: Any = (), expected: int = -1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.empty = empty
        self.expected = expected

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, self.empty)
            return
        if self.expected >= 0 and len(values) != self.expected:
            noun = "argument" if self.expected == 1 else "arguments"
            raise ArgumentError(self, f"expected {self.expected} {noun}")
        setattr(namespace, self.dest, tuple(values))


class _ScoutFailed(Exception):
    pass


class _Scout(_ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _ScoutFailed(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        raise _ScoutFailed(message or "")


def _scout_config(parser: ArgumentParser, argv: list[str]) -> str | None:
    if parser._config_flag is None or parser._config_dest is None:
        return None
    scout = _Scout(add_help=False, allow_abbrev=parser.allow_abbrev)
    scout.add_argument(parser._config_flag, dest=parser._config_dest, default=None)
    try:
        parsed, _ = scout.parse_known_args(argv)
    except _ScoutFailed:
        return None
    return vars(parsed)[parser._config_dest]


def _dest_of(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def str2bool(v):
    value = coerce_bool(v)
    if isinstance(value, bool):
        return value
    raise ArgumentTypeError("Boolean value expected.")


def _reraise_with_source(path, exc: ValueError) -> NoReturn:
    """Raise `exc` prefixed with `path`, unless the message already has it.

    `load_dict` includes the path in its own errors; wrapping it again reported
    `config.yaml: config.yaml: ...`.
    """
    message = exc.args[0] if exc.args else str(exc)
    prefix = f"{path}: "
    if isinstance(message, str) and message.startswith(prefix):
        raise exc
    raise ValueError(f"{prefix}{message}") from exc
