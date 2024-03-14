import importlib
from typing import Any, Type

# from copy import deepcopy
# from typing import Any, Callable, Generic, ParamSpec, Type, TypeVar


# T = TypeVar("T" , bound="Parsable")
# P = ParamSpec("P")

# A = ParamSpec("A")
# B = TypeVar("B")


class MissingType:
    pass


Missing = MissingType()
AllowedTypes = int | float | str


# class Lazy(Generic[T, P]):
#     def __init__(
#         self, cls: Type[T] | Callable[P, T],
#         *args: P.args, **kwargs: P.kwargs
#     ):
#         self.cls = cls
#         self.args = args
#         self.kwargs = kwargs

#         self._cfg = create_dict_from_class_init_args(cls, *args, **kwargs)

#     def __eq__(self, other: "Lazy") -> bool:
#         assert isinstance(other, Lazy)
#         return self._cfg == other._cfg

#     def to_eager(self, *args: P.args, **kwargs: P.kwargs) -> T:
#         assert not args
#         kwargs.update(self.kwargs)
#         return self.cls(*self.args, **kwargs)

#     @classmethod
#     def from_dict(cls, dct) -> "Lazy":
#         dct = deepcopy(dct)
#         pth: str = dct.pop("_type")
#         class_ = import_class_from_path(pth)

#         kwargs = dict()
#         for k, v in dct.items():
#             if isinstance(v, dict) and "_type" in v:
#                 kwargs[k] = Lazy.from_dict(v)
#             else:
#                 kwargs[k] = v

#         return Lazy(class_, **kwargs)

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


Default = Any | MissingType
Annotation = Type | Type[MissingType]


def get_class_init_signature(
    cls, *args, **kwargs
) -> tuple[dict[str, Default], dict[str, Annotation]]:
    """Get the signature of a class, including defaults and annotations.

    This function retrieves the signature of a class's `__init__` method,
    including the default values and annotations of its parameters.
    If a default value or annotation is not available, it is replaced with `Missing`.
    If an argument type does not match the annotation, a `TypeError` is raised.

    Args:
        cls: The class for which to retrieve the signature.
        *args: Positional arguments to be passed to the `__init__` method.
        **kwargs: Keyword arguments to be passed to the `__init__` method.

    Returns:
        A tuple containing two dictionaries:
        - The first dictionary contains the values of the parameters
            passed to the `__init__` method, including any default values.
        - The second dictionary contains the annotations of the parameters
            passed to the `__init__` method, including any missing annotations.

    Raises:
        TypeError: If an argument type does not match the annotation.
        AssertionError: If the `__init__` method does not have a `self` parameter.

    """
    from inspect import _empty, signature

    sig = signature(cls.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)
    bound.apply_defaults()

    assert (
        "self" in bound.signature.parameters
    ), f"The {cls=} does not have a __init__ method with a 'self' parameter."

    values = dict()
    annotations = dict()
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue

        value = bound.arguments.get(
            param_name, param.default if param.default != _empty else Missing
        )
        annotation = param.annotation if param.annotation != _empty else MissingType

        if (
            annotation is not MissingType
            and value is not Missing
            and not isinstance(value, annotation)
        ):
            raise TypeError(
                f"Argument {param_name}={value} is annotated "
                f"with {annotation}, "
                f"but the default value has type {type(value)} "
            )

        values[param_name] = value
        annotations[param_name] = annotation

    return (values, annotations)


def create_dict_from_class_init_args(cls, *args, **kwargs):
    """
    Create a dictionary representation of the class initialization arguments.

    Args:
        cls: The class for which to create the dictionary.
        *args: Positional arguments passed to the class constructor.
            Must match the class signature.
        **kwargs: Keyword arguments passed to the class constructor.
            Must match the class signature.

    The class signature may contain non-allowed types, but the objects
    must implement `to_dict() -> dict[str, AllowedTypes]`.

    Returns:
        A dictionary representation of the class initialization arguments.

    Raises:
        TypeError: If the type of an argument is not an allowed type
            and does not implement `to_dict()`.
        AssertionError: If to_dict returns a dictionary with non-string keys
            or non-allowed values.
    """
    values, _ = get_class_init_signature(cls, *args, **kwargs)

    result: dict[str, Any] = {"_type": get_class_import_path(cls)}

    for name, value in values.items():

        if value is Missing:
            continue

        if isinstance(value, AllowedTypes):
            result[name] = value
        elif hasattr(value, "to_dict"):
            dct = value.to_dict()

            # TODO: We currently support only one level of nesting.
            # To support deeper nesting, we need to recursively check
            # the types of the values.
            assert all(
                isinstance(k, str) for k in dct.keys()
            ), f"Keys of {dct=} are not strings"
            assert all(
                isinstance(v, AllowedTypes) for v in dct.values()
            ), f"Values of {dct=} are not {AllowedTypes}"

            result[name] = dct
        else:
            raise TypeError(
                f"Type is not {AllowedTypes} and does not implement to_dict()"
            )

    return result


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
