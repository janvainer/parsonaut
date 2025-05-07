from enum import Enum
from functools import partial
from typing import Any, Callable, Generic, Mapping, ParamSpec, Type, TypeVar, get_args

from .serialization import Serializable, maybe_import
from .typecheck import Missing, MissingType, is_parsable_type

T = TypeVar("T")
P = ParamSpec("P")

A = ParamSpec("A")
B = TypeVar("B")


TYPECHECK_EAGER = False
TYPE_NAME = "_class"


class Lazy(Generic[T, P], Serializable):
    """
    A mixin class that allows for lazy initialization of a class instance.
    """

    # TODO: we are missing tuple here
    KeyTypes = (
        tuple[type["Lazy"], "Lazy"]
        | tuple[type[bool], bool]
        | tuple[type[int], int]
        | tuple[type[str], str]
        | tuple[type[float], float]
    )

    def __init__(
        self, cls: Type[T] | Callable[P, T], signature: partial | Mapping[str, KeyTypes]
    ) -> None:
        # going around the freezing thingy in __setattr__
        # https://stackoverflow.com/a/4828492
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "_signature", signature)

    def __hash__(self) -> int:
        return hash(
            tuple(
                self.to_dict(
                    with_annotations=True, with_class_tag=True, flatten=True
                ).items()
            )
        )

    def __eq__(self, __value: "object | Lazy") -> bool:
        return hash(self) == hash(__value)

    def __str__(self):
        return lazy_str(self.to_dict(with_class_tag=True))

    def __getattr__(self, x):
        signature = object.__getattribute__(self, "signature")
        if x not in signature:
            return object.__getattribute__(self, x)
        else:
            return signature[x][1]

    def __setattr__(self, *args):
        # This is here for Enum support, otherwise all frozen
        if args[0] in (
            "_value_",
            "_name_",
            "__objclass__",
            "_sort_order_",
        ):
            object.__setattr__(self, args[0], args[1])
        else:
            raise AssertionError("Cannot set attributes of Lazy class")

    @property
    def signature(self) -> Mapping[str, KeyTypes]:
        # This is here to prevent infinite recursion.
        _signature = object.__getattribute__(self, "_signature")
        if isinstance(_signature, partial):
            _signature = _signature()
            object.__setattr__(self, "_signature", _signature)
        return _signature

    @staticmethod
    def is_lazy_type(typ):
        origin = getattr(typ, "__origin__", None)
        if origin is None and issubclass(typ, Lazy):
            return True
        elif origin == Lazy:
            return True
        else:
            return False

    @staticmethod
    def from_class(
        cl: Type[B] | Callable[A, B], *args, skip_non_parsable: bool = False, **kwargs
    ) -> "Lazy[B, A]":

        if should_typecheck_eagerly():
            sig = Lazy.get_signature(
                cl, *args, skip_non_parsable=skip_non_parsable, **kwargs
            )
            return Lazy(cl, sig)
        else:
            return Lazy(
                cl,
                partial(
                    Lazy.get_signature,
                    cl,
                    *args,
                    skip_non_parsable=skip_non_parsable,
                    **kwargs,
                ),
            )

    @staticmethod
    def get_signature(
        cl, *args, skip_non_parsable: bool = False, **kwargs
    ) -> Mapping[str, KeyTypes]:
        func = cl.__init__
        signature = get_signature(func, *args, **kwargs)

        res = dict()
        for name, (typ, value) in signature.items():

            # enforce default values
            if Lazy.is_lazy_type(typ):
                # fill in missing default
                if value is Missing and (subtyp := get_args(typ)):
                    subtyp, *_ = subtyp
                    assert issubclass(subtyp, Parsable)
                    value = Lazy.from_class(subtyp)  # type: ignore
                # or ensure correct type
                else:
                    assert isinstance(
                        value, Lazy
                    ), f"Expected value to be parsable or a Lazy. Got {type(value)}"
                res[name] = (typ, value)
                continue

            # fill in parsable type if value is Parsable
            if isinstance(value, Lazy):
                if typ == MissingType:
                    typ = Lazy
                else:
                    assert Lazy.is_lazy_type(typ)
                res[name] = (typ, value)
                continue

            # Skip variables without type annotation - the user can fill them in with to_eager() call
            if typ == MissingType:
                if not skip_non_parsable:
                    assert value is Missing
                continue

            # skip non-parsable - the user can fill them in with to_eager() call
            if not is_parsable_type(typ):
                if not skip_non_parsable:
                    assert (
                        value is Missing
                    ), f"Cannot initialize Lazy[{cl.__name__}, ...] with variable {name=} and non-parsable type={typ}."
                continue

            # check if the provided value is parsable and matches the annotation
            assert value == Missing or is_parsable_type(typ, value), (
                f"Provided value {name}={value} does not match "
                f"the provided annotation {name}: {typ}"
            )
            res[name] = (typ, value)

        return res

    def copy(self: "Lazy[B, A]", fields: dict | None = None) -> "Lazy[B, A]":
        dct = self.to_dict(with_class_tag=True, flatten=True)
        if fields is None:
            return Lazy.from_dict(dct)

        for field, new_val in fields.items():
            assert (
                "._class" not in field and "_class" not in field
            ), "Cannot change class."
            assert field in dct, f"Attempted to copy with {field=} that is not present."

            dct[field] = new_val

        return Lazy.from_dict(dct)

    def to_dict(
        self,
        recursive: bool = True,
        with_annotations: bool = False,
        with_class_tag: bool = False,
        with_class_tag_as_str: bool = False,
        flatten: bool = False,
    ):
        dct = dict()
        if with_class_tag:
            dct[TYPE_NAME] = self.cls
        elif with_class_tag_as_str:
            dct[TYPE_NAME] = f"{self.cls.__module__}.{self.cls.__name__}"
        for k, (typ, value) in sorted(self.signature.items()):

            if Lazy.is_lazy_type(typ):
                if recursive:
                    assert isinstance(value, Lazy)
                    value = value.to_dict(
                        recursive=recursive,
                        with_annotations=with_annotations,
                        with_class_tag=with_class_tag,
                        with_class_tag_as_str=with_class_tag_as_str,
                    )
                dct[k] = value
            else:
                if with_annotations:
                    dct[k] = (typ, value)
                else:
                    dct[k] = value

        if flatten:
            dct = flatten_dict(dct)

        return dct

    @staticmethod
    def from_dict(dct):
        # For now we assume the dict contains TYPE_NAME
        # In the future, we should be able to infer the TYPE_NAME also for sub-classes from defaults
        if any("." in k for k in dct):
            dct = unflatten_dict(dct)

        signature = dict()

        cls = maybe_import(dct[TYPE_NAME])

        for k, v in dct.items():
            if k == TYPE_NAME:
                continue
            elif isinstance(v, dict):
                signature[k] = Lazy.from_dict(v)
            else:
                signature[k] = v

        return Lazy.from_class(cls, **signature)

    def to_eager(self, *args: P.args, **kwargs: P.kwargs) -> T:
        assert not args

        kwargs2 = self.to_dict(recursive=False)
        kwargs = {**kwargs2, **kwargs}
        kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, MissingType)}

        return self.cls(
            *args,
            **kwargs,
        )


class ParsableMeta(type):
    def __call__(cls, *args, **kwargs):

        cfg = Lazy.from_class(cls, *args, skip_non_parsable=True, **kwargs)

        # https://stackoverflow.com/a/73923070/8378586
        obj = cls.__new__(cls, *args, **kwargs)
        obj._cfg = cfg
        # Initialize the final object
        obj.__init__(*args, **kwargs)
        return obj


class class_or_instance_method(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, instance, owner):
        if instance is not None:
            class_or_instance = instance
        else:
            class_or_instance = owner

        def newfunc(*args, **kwargs):
            return self.f(class_or_instance, *args, **kwargs)

        return newfunc


class Parsable(Serializable, metaclass=ParsableMeta):

    _cfg: Lazy

    @class_or_instance_method
    def as_lazy(cls_or_self, *args, **kwargs) -> Lazy:
        if isinstance(cls_or_self, Parsable):
            assert not args, "Cannot override once configured Parsable."
            assert not kwargs, "Cannot override once configured Parsable."
            return cls_or_self._cfg
        else:
            return Lazy.from_class(cls_or_self, *args, **kwargs)

    def to_dict(
        self,
        with_class_tag: bool = False,
        with_class_tag_as_str: bool = False,
        flatten: bool = False,
    ):
        return self._cfg.to_dict(
            with_class_tag=with_class_tag,
            with_class_tag_as_str=with_class_tag_as_str,
            flatten=flatten,
        )

    @classmethod
    def from_dict(cls, dct) -> Lazy:
        return Lazy.from_class(cls).from_dict(dct)

    @classmethod
    def parse_args(cls, *args, **kwargs) -> Lazy:
        from .parse import ArgumentParser

        parser = ArgumentParser()
        parser.add_options(cls.as_lazy(*args, **kwargs))
        params = parser.parse_args()
        return params


class Choices(Lazy, Enum):
    def __new__(cls, value):
        assert isinstance(
            value, Lazy
        ), f"Choice values must be an instance of Lazy. Got {type(value)}"
        obj = Lazy.__new__(cls)
        return obj

    def __init__(self, *args):
        (orig_lazy,) = args
        object.__setattr__(self, "cls", orig_lazy.cls)
        object.__setattr__(self, "_signature", orig_lazy._signature)


def should_typecheck_eagerly():
    return TYPECHECK_EAGER


class typecheck_eager:
    def __init__(self):
        global TYPECHECK_EAGER
        TYPECHECK_EAGER = True

    def __enter__(self):
        pass

    def __exit__(self, *args, **kws):
        global TYPECHECK_EAGER
        TYPECHECK_EAGER = False


def set_typecheck_eager(eager: bool = True):
    global TYPECHECK_EAGER
    TYPECHECK_EAGER = eager


def get_signature(func: Callable, *args, **kwargs) -> dict[str, tuple[Type, Any]]:
    """Get the signature of a function, including the types of the arguments."""
    from inspect import _empty, signature

    sig = signature(func)
    if "self" in sig.parameters:
        bound = sig.bind_partial(None, *args, **kwargs)
    else:
        bound = sig.bind_partial(*args, **kwargs)
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


def flatten_dict(dct: dict) -> dict:

    def _flatten(dct, prefix: str):
        out = list()
        for k, v in dct.items():
            if prefix:
                k = f"{prefix}.{k}"
            if isinstance(v, dict):
                out.extend(_flatten(v, prefix=k))
            else:
                out.append((k, v))
        return out

    return dict(_flatten(dct, ""))


def unflatten_dict(flat: dict) -> dict:

    base = {}
    for key, value in flat.items():
        root = base

        if "." in key:
            *parts, key = key.split(".")

            for part in parts:
                # This should ignore choice flags such as --encoder ENCODER
                if part in root and not isinstance(root[part], dict):
                    root[part] = {}
                if part not in root:
                    root[part] = {}
                root = root[part]

        root[key] = value

    return base


def lazy_str(dct: dict, level: int = 1):

    def format_attr(k, v):
        if isinstance(v, str):
            return f"{k}='{v}'"
        else:
            return f"{k}={v}"

    header = f'{dct["_class"].__name__}'
    attrs = [
        (
            f"{k}={lazy_str(v, level=level + 1)}"
            if isinstance(v, dict)
            else format_attr(k, v)
        )
        for k, v in dct.items()
        if k != "_class"
    ]
    indent = "    "
    attrs = f",\n{indent * level}".join(attrs)
    out = f"{header}(\n{indent * level}{attrs},\n{indent * (level - 1)})"
    return out
