from functools import partial
from typing import Any, Callable, Generic, Mapping, ParamSpec, Type, TypeVar

from .typecheck import is_parsable_type


class MissingType:
    pass


Missing = MissingType()


T = TypeVar("T")
P = ParamSpec("P")

A = ParamSpec("A")
B = TypeVar("B")


TYPECHECK_EAGER = False
TYPE_NAME = "_class"


class Signature(Generic[T, P]):
    """
    A mixin class that allows for lazy initialization of a class instance.
    """

    KeyTypes = (
        tuple[type["Signature"], "Signature"]
        | tuple[type[bool], bool]
        | tuple[type[int], int]
        | tuple[type[str], str]
        | tuple[type[float], float]
    )

    def __init__(
        self, cls: Type[T] | Callable[P, T], signature: partial | Mapping[str, KeyTypes]
    ) -> None:
        self.cls = cls
        self._signature = signature

    def __hash__(self) -> int:
        return hash(
            tuple(
                self.to_dict(
                    with_annotations=True, with_class_tag=True, flatten=True
                ).items()
            )
        )

    def __eq__(self, __value: "object | Signature") -> bool:
        return hash(self) == hash(__value)

    @property
    def signature(self) -> Mapping[str, KeyTypes]:
        if isinstance(self._signature, partial):
            self._signature = self._signature()
        return self._signature

    @staticmethod
    def is_signature_type(typ):
        return getattr(typ, "__origin__", None) == Signature or issubclass(
            typ, Signature
        )

    @staticmethod
    def from_class(cl: Type[B] | Callable[A, B], *args, **kwargs) -> "Signature[B, A]":

        if should_typecheck_eagerly():
            sig = Signature.get_signature(cl, *args, **kwargs)
            return Signature(cl, sig)
        else:
            return Signature(cl, partial(Signature.get_signature, cl, *args, **kwargs))

    @staticmethod
    def get_signature(cl, *args, **kwargs) -> Mapping[str, KeyTypes]:
        func = cl.__init__
        signature = get_signature(func, *args, **kwargs)

        res = dict()
        for name, (typ, value) in signature.items():

            # enforce default values
            if Signature.is_signature_type(typ):
                # fill in missing default
                if value is Missing:
                    value = Signature.from_class(typ)  # type: ignore
                # or ensure correct type
                else:
                    assert isinstance(
                        value, Signature
                    ), f"Expected value to be parsable or a Signature. Got {type(value)}"
                res[name] = (typ, value)
                continue

            # fill in parsable type if value is Parsable
            if isinstance(value, Signature):
                if typ == MissingType:
                    typ = Signature
                else:
                    assert Signature.is_signature_type(typ)
                res[name] = (typ, value)
                continue

            # Skip variables without type annotation
            if typ == MissingType:
                continue

            # skip non-parsable - the user can fill them in with to_eager() call
            if not is_parsable_type(typ):
                assert (
                    value is Missing
                ), f"Cannot initialize Lazy[{cl.__name__}, ...] with variable {name=} and non-parsable type={typ}."
                continue

            # check if the provided value is parsable and matches the annotation
            assert value is Missing or is_parsable_type(typ, value), (
                f"Provided value {name}={value} does not match "
                f"the provided annotation {name}: {typ}"
            )
            res[name] = (typ, value)

        return res

    def to_dict(
        self,
        recursive: bool = True,
        with_annotations: bool = False,
        with_class_tag: bool = False,
        flatten: bool = False,
    ):
        dct = dict()
        if with_class_tag:
            dct[TYPE_NAME] = self.cls
        for k, (typ, value) in sorted(self.signature.items()):
            if value is Missing and not with_annotations:
                continue

            if Signature.is_signature_type(typ):
                if recursive:
                    assert isinstance(value, Signature)
                    value = value.to_dict(
                        recursive=recursive,
                        with_annotations=with_annotations,
                        with_class_tag=with_class_tag,
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

        cls = dct[TYPE_NAME]
        for k, v in dct.items():
            if k == TYPE_NAME:
                continue
            elif isinstance(v, dict):
                signature[k] = Signature.from_dict(v)
            else:
                signature[k] = v

        return Signature.from_class(cls, **signature)

    def to_eager(self, *args: P.args, **kwargs: P.kwargs) -> T:
        assert not args
        return self.cls(  # type: ignore
            *args,
            **{  # type: ignore
                **self.to_dict(recursive=False),
                **kwargs,
            },
        )


class ParsableMeta(type):
    def __call__(cls, *args, **kwargs):

        cfg = cls.as_lazy(*args, **kwargs).to_dict(
            with_class_tag=True,
        )

        # https://stackoverflow.com/a/73923070/8378586
        obj = cls.__new__(cls, *args, **kwargs)
        obj._cfg = cfg
        # Initialize the final object
        obj.__init__(*args, **kwargs)
        return obj


class Parsable(metaclass=ParsableMeta):

    _cfg: dict

    @classmethod
    def as_lazy(
        cls: Callable[A, B], *args: A.args, **kwargs: A.kwargs
    ) -> "Signature[B, A]":
        return Signature.from_class(cls, *args, **kwargs)

    def to_dict(
        self,
        with_class_tag: bool = False,
        flatten: bool = False,
    ):
        def remove_class_tag(dct):
            return {
                k: (v if not isinstance(v, dict) else remove_class_tag(v))
                for k, v in dct.items()
                if k != TYPE_NAME
            }

        dct = self._cfg
        if not with_class_tag:
            dct = remove_class_tag(self._cfg)

        if flatten:
            dct = flatten_dict(dct)

        return dct

    @classmethod
    def from_dict(cls, dct):
        # For now we assume the dict contains TYPE_NAME
        # In the future, we should be able to infer the TYPE_NAME also for sub-classes from defaults
        return Signature.from_class(cls).from_dict(dct).to_eager()


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
            if isinstance(v, dict):
                out.extend(_flatten(v, prefix=k))
            else:
                if prefix:
                    k = f"{prefix}.{k}"
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
                root.setdefault(part, {})
                root = root[part]

        root[key] = value

    return base
