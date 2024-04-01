import importlib
from builtins import Ellipsis
from types import UnionType
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


T = TypeVar("T")
P = ParamSpec("P")

A = ParamSpec("A")
B = TypeVar("B")


V = TypeVar("V")
BasicType = int | float | bool | str
ParsableType = BasicType | tuple[BasicType] | tuple[tuple[BasicType]]

TYPECHECK_EAGER = True


class MissingType:
    pass


Missing = MissingType()


class Lazy(Generic[T, P]):

    def __init__(self, cls: Callable[P, T], signature: "Signature | partial") -> None:
        self.cls = cls
        self._signature = signature

    @property
    def signature(self):
        if isinstance(self._signature, partial):
            self._signature = self._signature()
        return self._signature

    @staticmethod
    def from_class(
        cls: Type[B] | Callable[A, B], *args: A.args, **kwargs: A.kwargs
    ) -> "Lazy[B, A]":
        if TYPECHECK_EAGER:
            sig = get_typechecked_signature(cls.__init__, *args, **kwargs)
            return Lazy(cls, sig)
        else:
            return Lazy(cls, partial(get_typechecked_signature, cls.__init__, *args, **kwargs))

    @staticmethod
    def from_dict(dct) -> "Lazy":
        # TODO: This is not going to work once we introduce unions
        cls = import_class_from_path(dct.pop("_type"))
        return Lazy.from_class(cls).update(**dct)

    def to_eager(self, **kwargs):
        return self.func(
            **{
                **self.signature,
                **kwargs,
            }
        )

    def to_dict(self, include_type: bool = False):
        if include_type:
            dct = {"_type": get_class_import_path(self.cls)}
        else:
            dct = dict()
        for k, (typ, value) in self.signature.items():
            dct[k] = value.to_dict(include_type=include_type) if isinstance(value, Lazy) else value

        return dct

    def update(self: "Lazy[B, A]", **kwargs: A.kwargs) -> "Lazy[B, A]":

        new_sig = dict()

        kwargs.pop("_type", None)
        assert set(kwargs) <= set(self.signature)

        for k, (typ, value) in self.signature.items():
            if k in kwargs:
                if isinstance(value, Lazy):
                    new_sig[k] = (typ, value.update(**kwargs[k]))
                else:
                    # TODO: deepcopy? Maybe we do not have to as long as we ensure that all is immutable
                    assert _isinstance_of_parsable_type(new_val := kwargs[k], typ)
                    new_sig[k] = (typ, new_val)
            else:
                new_sig[k] = (typ, value)

        return Lazy(self.cls, new_sig)
    
    def to_cli_args(self, prefix: str = "") -> list[tuple[str, Type, ParsableType | MissingType]]:
        args = []
        for k, (typ, value) in self.signature.items():
            if isinstance(value, Lazy):
                args.extend(
                    value.to_cli_args(prefix=f"{k}.")
                )
            else:
                args.append(
                    (f"{prefix}{k}", typ, value)
                )
        return args
    
    def to_argument_parser(self):
        cli_args = self.to_cli_args()
        parser = parser_from_cli_args(cli_args)
        return parser

    def parse_args(self):
        # TODO:
        parser = self.to_argument_parser()
        args = parser.parse_args()

        # TODO: dot dict to nested and then update


class Parsable:

    @classmethod
    def as_lazy(
        cls: Type[B] | Callable[A, B], *args: A.args, **kwargs: A.kwargs
    ) -> Lazy[B, A]:
        return Lazy.from_class(cls, *args, **kwargs)


Signature = dict[
    str, tuple[Type, BasicType | tuple[BasicType, ...], tuple[tuple[BasicType]] | Lazy]
]


def get_signature(
    func: Callable, *args, **kwargs
) -> dict[str, tuple[Type, Lazy | Any]]:
    from inspect import _empty, signature

    sig = signature(func)
    bound = sig.bind_partial(None, *args, **kwargs)
    bound.apply_defaults()

    ret = dict()
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue

        value = bound.arguments.get(
            param_name, param.default if param.default != _empty else Missing
        )
        annotation = param.annotation if param.annotation != _empty else MissingType
        ret[param_name] = (annotation, value)

    return ret


def get_typechecked_signature(func: Callable, *args, **kwargs) -> Signature:

    signature = get_signature(func, *args, **kwargs)

    res = dict()
    for name, (typ, value) in signature.items():

        # enforce default if type is Lazy
        if typ == Lazy:
            assert isinstance(value, Lazy)
            res[name] = (typ, value)
            continue

        # preserve lazy
        if isinstance(value, Lazy):
            if typ == MissingType:
                typ = Lazy

            res[name] = (typ, value)
            continue

        # skip non-parsable
        if not _is_parsable_type(typ):
            # non-parsables cannot be initialized with defaults
            assert (
                value is Missing
            ), f"Cannot initialize Lazy with non-parsable type={typ}."
            continue

        # check parsable type against its value
        assert _isinstance_of_parsable_type(
            value, typ
        ), f"Provided value {name}={value} does not match the provided annotation {name}: {typ}"

        res[name] = (typ, value)

    return res


def parser_from_cli_args(args: list[tuple[str, Type, ParsableType]]):
    from argparse import ArgumentParser, BooleanOptionalAction, ArgumentDefaultsHelpFormatter

    parser =  ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    for (name, typ, value) in args:
        
        if typ in (str, float, int):
            if value is Missing:
                parser.add_argument(f"--{name}", type=typ, required=True)
            else:
                parser.add_argument(f"--{name}", type=typ, default=value, help="%(type)s (default: %(default)s)")
        elif get_origin((nargs:=get_args(typ))[0]) != tuple:
            # TODO: we should not allow tuple of a single element - if so, user does not need tuple
            inner = nargs[0]
            parser.add_argument(
                f"--{name}",
                type=inner,
                default=value,
                help="%(type)s (default: %(default)s)",
                nargs="+" if nargs[1] == Ellipsis else len(nargs)
            )
        else:
            print("Nested tuple <3")
    return parser

def get_class_import_path(cls: Type) -> str:
    pth = cls.__module__ + "." + cls.__name__
    return pth


def import_class_from_path(path: str) -> Type:
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, cls_name)
    except AttributeError:
        raise ImportError(f"Could not import {cls_name} from {module_name}")
    return cls


def _isinstance_of_parsable_type(value: Any, typ: Type) -> bool:
    if isinstance(value, BasicType):
        return isinstance(value, typ)
    elif isinstance(value, tuple):
        outer = get_origin(typ)

        inner = get_args(typ)
        if inner[1] == Ellipsis:
            return all(
                _isinstance_of_parsable_type(item, inner[0])
                for item in value        
            )
        assert len(inner) == len(value)
        return all(
            _isinstance_of_parsable_type(item, inner_typ)
            for item, inner_typ in zip(value, inner)
        )
    else:
        raise


def _is_parsable_type(typ: Type, level=0) -> bool:

    if level > 2:
        return False

    if _is_union(typ):
        return False

    if typ == MissingType:
        return False

    return _is_basic_type(typ) or _is_parsable_container(typ, level)


def _is_union(typ) -> bool:
    outer = get_origin(typ)
    return outer is Union or outer is UnionType


def _is_basic_type(typ) -> bool:
    return typ in (int, float, bool, str)


def _is_parsable_container(typ, level=0) -> bool:
    outer = get_origin(typ)

    # untyped tuple not allowed
    if outer is None:
        return False

    if outer != tuple:
        return False

    inner = get_args(typ)
    return all(_is_parsable_type(i, level + 1) for i in inner if i != Ellipsis)
