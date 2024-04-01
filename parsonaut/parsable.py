from functools import partial
from typing import Callable, Generic, ParamSpec, Type, TypeVar

from .signature import Missing, MissingType, get_signature
from .typecheck import is_parsable_type

T = TypeVar("T")
P = ParamSpec("P")

A = ParamSpec("A")
B = TypeVar("B")


TYPECHECK_EAGER = True


class Parsable(Generic[T, P]):
    """
    A mixin class that allows for lazy initialization of a class instance.
    """

    def __init__(self, cls: Callable[P, T], signature) -> None:
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
    ) -> "Parsable[B, A]":
        if should_typecheck_eagerly():
            sig = Parsable.get_signature(cls.__init__, *args, **kwargs)
            return Parsable(cls, sig)
        else:
            return Parsable(
                cls, partial(Parsable.get_signature, cls.__init__, *args, **kwargs)
            )

    @staticmethod
    def get_signature(func: Callable, *args, **kwargs):
        signature = get_signature(func, *args, **kwargs)

        res = dict()
        for name, (typ, value) in signature.items():

            # enforce default if type is Parsable
            if typ == Parsable:
                assert isinstance(value, Parsable)
                res[name] = (typ, value)
                continue

            # fill in parsable type if value is Parsable
            if isinstance(value, Parsable):
                if typ == MissingType:
                    typ = Parsable

                res[name] = (typ, value)
                continue

            # skip non-parsable - the user can fill them in with to_eager() call
            if not is_parsable_type(typ):
                assert (
                    value is Missing
                ), f"Cannot initialize Lazy with non-parsable type={typ}."
                continue

            # check if the provided value is parsable and matches the annotation
            assert value is Missing or is_parsable_type(typ, value), (
                f"Provided value {name}={value} does not match "
                f"the provided annotation {name}: {typ}"
            )
            res[name] = (typ, value)

        return res

    @classmethod
    def as_lazy(
        cls: Type[B] | Callable[A, B], *args: A.args, **kwargs: A.kwargs
    ) -> "Parsable[B, A]":
        return Parsable.from_class(cls, *args, **kwargs)

    def to_dict(self, recursive: bool = True, with_annotations: bool = False):
        dct = dict()
        for k, (typ, value) in self.signature.items():
            val = (
                value.to_dict(recursive=recursive)
                if recursive and isinstance(value, Parsable)
                else value
            )
            if not with_annotations:
                dct[k] = val
            else:
                dct[k] = (typ, val)

        return dct

    def to_eager(self, **kwargs):
        return self.cls(
            **{
                **self.to_dict(recursive=False),
                **kwargs,
            }
        )


def should_typecheck_eagerly():
    return TYPECHECK_EAGER
