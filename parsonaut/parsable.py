import importlib
from builtins import Ellipsis
from copy import deepcopy
from types import UnionType
from typing import (
    Any,
    Callable,
    Generic,
    NamedTuple,
    ParamSpec,
    Type,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

# from typing import Any, Callable, Generic, ParamSpec, Type, TypeVar


T = TypeVar("T")  # , bound="Parsable")
P = ParamSpec("P")

# A = ParamSpec("A")
# B = TypeVar("B")


V = TypeVar("V")
BasicType = int | float | bool | str


class MissingType:
    pass


Missing = MissingType()


class AnnotatedDict(TypedDict):
    _type: Type
    value: "Any | AnnotatedDict"


class Lazy(Generic[T, P]):
    def __init__(
        self, cls: Type[T] | Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ):
        # Best effort to ensure that non-lazy objects are not passed to Lazy as values.
        # If the user passes e.g. an initialized torch model, it would slow down imports and ramp up mem usage.
        # We do not recurse into containers as it could slow down imports for larger projects.
        # Container-related and type-related issues are handled during parsing.
        signature = get_class_init_signature(cls, *args, **kwargs)
        LazyType = int | float | str | bool | list | tuple | Lazy
        for item in signature:
            if item.value is not Missing:
                assert isinstance(item.value, LazyType), (
                    f"Only {LazyType} are allowed as values provided to Lazy, "
                    f"but received type {type(item.value)} for {item.name}"
                )

        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.signature = signature

    def to_annotated_dict(self) -> AnnotatedDict:
        result: dict[str, Any] = {}

        for sig in self.signature:
            match sig:
                case (_, typ, _) if _is_union(typ):
                    raise NotImplementedError()
                case (name, typ, MissingType()):
                    if _is_parsable_type(typ):
                        result[name] = {
                            "_type": typ,
                            "value": Missing,
                        }
                case (name, typ, value) if typ == MissingType:
                    if isinstance(value, BasicType):
                        result[name] = {
                            "_type": type(value),
                            "value": value,
                        }
                case (name, _, Lazy()):
                    result[name] = sig.value.to_annotated_dict()
                case (name, typ, value):
                    assert _is_parsable_type(
                        typ
                    ), f"Provided annotation for {name}: {typ} is not parsable."
                    assert _isinstance_of_parsable_type(
                        value, typ
                    ), f"Provided value {name}={value} does not match the provided annotation {name}: {typ}"
                    result[name] = {
                        "_type": typ,
                        "value": value,
                    }

                case _:
                    raise

        return {
            "_type": self.cls,
            "value": result,
        }

    #     def __eq__(self, other: "Lazy") -> bool:
    #         assert isinstance(other, Lazy)
    #         return self._cfg == other._cfg

    #     def to_eager(self, *args: P.args, **kwargs: P.kwargs) -> T:
    #         assert not args
    #         kwargs.update(self.kwargs)
    #         return self.cls(*self.args, **kwargs)

    # @classmethod
    # def from_dict(cls, dct) -> "Lazy":
    #     dct = deepcopy(dct)
    #     pth: str = dct.pop("_type")
    #     class_ = import_class_from_path(pth)

    #     kwargs = dict()
    #     for k, v in dct.items():
    #         if isinstance(v, dict) and "_type" in v:
    #             kwargs[k] = Lazy.from_dict(v)
    #         else:
    #             kwargs[k] = v

    #     return Lazy(class_, **kwargs)


#     def to_dict(self) -> dict[str, Any]:
#         return self._cfg

#     def parse_from_cli(self): ...


# class ParsableMeta(type):
#     def __call__(cls, *args, **kwargs):

#         cfg = _class_to_dict(cls, allow_eager=True, *args, **kwargs)

#         # https://stackoverflow.com/a/73923070/8378586
#         obj = cls.__new__(cls, *args, **kwargs)
#         obj._cfg = cfg
#         # Initialize the final object
#         obj.__init__(*args, **kwargs)
#         return obj


# # TODO: add a hook to check that defaults in __init__ are allowed?
# Perhaps make it optional with a global flag


# class Parsable(metaclass=ParsableMeta):

#     _cfg: dict[str, Any]

#     @classmethod
#     def as_lazy(
#         cls: Type[B] | Callable[A, B], *args: A.args, **kwargs: A.kwargs
#     ) -> Lazy[B, A]:
#         return Lazy(cls, *args, **kwargs)

#     @classmethod
#     def parse_from_cli(cls: Type[B]) -> B: ...

#     @classmethod
#     def from_dict(cls, dct):
#         # TODO: allow providing eager args?
#         if cls != Parsable:
#             assert _class_name(cls) == dct["_type"]
#         return Lazy.from_dict(dct).to_eager()

#     def to_dict(self):
#         return self._cfg

#     def to_eager(self: B, *args, **kwargs) -> B:
#         assert not (args or kwargs)
#         return self


class Signature(NamedTuple):
    name: str
    annotation: Type | Type[MissingType]
    value: Any | MissingType


def get_class_init_signature(cls, *args, **kwargs) -> tuple[Signature, ...]:
    """Get the signature of a class's __init__ method.

    This function retrieves the signature of a class's __init__ method,
    including the default values and annotations of its parameters.
    If a default value or annotation is not available, it is replaced with `Missing`.

    Args:
        cls: The class for which to retrieve the signature.
        *args: Positional arguments to be passed to the __init__ method.
        **kwargs: Keyword arguments to be passed to the __init__ method.

    Returns:
        A tuple of Signature instances, each containing the name, annotation,
        and value of a parameter.

    Raises:
        AssertionError: If the __init__ method does not have a 'self' parameter.

    """
    from inspect import _empty, signature

    sig = signature(cls.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)
    bound.apply_defaults()

    assert (
        "self" in bound.signature.parameters
    ), f"The {cls=} does not have a __init__ method with a 'self' parameter."

    ret = []
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue

        value = bound.arguments.get(
            param_name, param.default if param.default != _empty else Missing
        )
        annotation = param.annotation if param.annotation != _empty else MissingType

        ret.append(Signature(param_name, annotation, value))

    return tuple(ret)


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
    elif isinstance(value, list | tuple):
        outer = get_origin(typ)

        inner = get_args(typ) if outer == tuple else [get_args(typ)[0]] * len(value)
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

    return _is_basic_type(typ) or _is_parsable_container(typ, level)


def _is_union(typ) -> bool:
    outer = get_origin(typ)
    return outer is Union or outer is UnionType


def _is_basic_type(typ) -> bool:
    return typ in (int, float, bool, str)


def _is_parsable_container(typ, level=0) -> bool:
    outer = get_origin(typ)

    # untyped list or tuple not allowed
    if outer is None:
        return False

    if outer not in (list, tuple):
        return False

    inner = get_args(typ)
    return all(_is_parsable_type(i, level + 1) for i in inner if i != Ellipsis)


# class SomeOther(nn.Module, Parsable):
#     def __init__(self, y: int = 1):
#         super().__init__()
#         self.y = y


# class YetAnother(nn.Module, Parsable):
#     def __init__(self, y: int = 1):
#         super().__init__()
#         self.y = y


# class SomeClass(nn.Module, Parsable):
#     def __init__(
#         self,
#         y: int = 7,
#         x: SomeOther | Lazy[SomeOther, ...] = SomeOther.as_lazy(),
#     ):
#         super().__init__()
#         self.y = y
#         self.x = x.to_eager()


# # here it is lazy. Init has not been called yet
# # x = SomeClass.as_lazy()
# # here all params loaded
# # x1 = x.to_eager()

# # TODO: Only allow missing args to be passed to eager to prevent overriding
# # SomeClass.as_lazy(y=5).to_eager(y=7).to_dict()
# # print(SomeClass(y=5, x=SomeOther(y=2)).to_dict())
# this should work like SomeClass.as_lazy(y=5, x=SomeOther.as_lazy()).to_eager()
# # print(SomeClass.as_lazy(y=5, x=SomeOther.as_lazy(y=2)).to_dict())
# this should work like SomeClass.as_lazy(y=5, x=SomeOther.as_lazy()).to_eager()

# # print(SomeClass.as_lazy(y=5, x=SomeOther(y=2)).to_dict())
# this should work like SomeClass.as_lazy(y=5, x=SomeOther.as_lazy()).to_eager()

# dct = SomeClass.as_lazy(
#     y=5, x=SomeOther.as_lazy(y=2)
# ).to_dict()
# this should work like SomeClass.as_lazy(y=5, x=SomeOther.as_lazy()).to_eager()
# print(dct)
# print(Lazy.from_dict(dct).to_dict())
# print(SomeClass.from_dict(dct))


# SomeClass.parse_from_cli(
#     *args, **kwargs
# )  # SomeClass.as_lazy().parse_from_cli().to_eager(*args, **kwargs)
# What we have:
#   1. pre-init _cfg template for parsing generated from the __init__ init defaults
#   - ok maybe this should happen irrespective of the Lazy object
#       - this is made of (name, default, type) tuples (or nested dicts?)
#       [x] its only needed to parse from CLI and for that we can guarantee that
#           the object is lazified first
#   2. instantion of this config template - the stuff that really goes into the
#       __init__ function
#       - we should go through the args and kwargs and build a separate config instance
#           per instantiation - no global parent-child hooks
#       - use a metaclass for it


# print(_issubtype(type(1), Allowed))
# print(issubtype(int, int | str | Lazy))
# print(issubtype(Lazy, int | str | Lazy))

# print(get_origin(Lazy[SomeClass, ...]) in [int, float, str, Lazy])
# print(get_origin(int) in [int, float, str, Lazy])

# print(SomeClass.as_lazy().to_dict())
# o = SomeClass(*args, **kwargs)  # cfg = SomeClass.as_lazy(*args, **kwargs).to_dict()
# o.to_dict()  # o.cfg

# Add a metaclass so that the Model is always initialized via the as_lazy mechanism.
# So that Model() actually does Model.as_lazy().to_eager() under the hood
