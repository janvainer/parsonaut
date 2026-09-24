from abc import ABCMeta
from inspect import signature
from typing import Any, Callable, TypeVar, cast, overload

from .dicts import flatten_dict
from .lazy import Lazy, new_lazy
from .serialization import Serializable, load_dict

T = TypeVar("T", bound="Parsable")


class ParsableMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # An object being built may legitimately be handed arguments that a
        # configuration cannot record, so this path does not reject them.
        cfg = new_lazy(cls, args, kwargs, strict=False)

        # The object is allocated and initialized in two steps so that `_cfg` is
        # already available while `__init__` runs.
        # https://stackoverflow.com/a/73923070/8378586
        obj = _allocate(cls, args, kwargs)
        if isinstance(obj, cls):
            # Not `obj._cfg = cfg`, which a frozen dataclass would refuse.
            object.__setattr__(obj, "_cfg", cfg)
            obj.__init__(*args, **kwargs)
        return obj


def _allocate(cls, args, kwargs) -> Any:
    """Allocate an instance of `cls` without running `__init__` yet."""
    if cls.__new__ is object.__new__:
        # `object.__new__` rejects extra arguments.
        return object.__new__(cls)
    if not _accepts(cls.__new__, cls, *args, **kwargs):
        # A custom `__new__` that does not take the constructor arguments.
        return cls.__new__(cls)
    return cls.__new__(cls, *args, **kwargs)


def _accepts(func, /, *args, **kwargs) -> bool:
    try:
        signature(func).bind(*args, **kwargs)
    except TypeError:
        return False
    except ValueError:  # pragma: no cover - needs a __new__ written in C
        # Not introspectable, so assume it takes them and let the call decide.
        return True
    return True


class _AsLazy:
    """Exposes `as_lazy` on both a `Parsable` class and its instances.

    On the class it configures a new object and is typed as returning that
    class, so it can be used as a default: `model: Model = Model.as_lazy()`.
    On an instance it hands back the configuration the object was built from,
    typed as the `Lazy` it really is, so `copy` and the signature stay
    reachable.
    """

    @overload
    def __get__(self, obj: None, owner: type[T]) -> Callable[..., T]: ...

    @overload
    def __get__(self, obj: T, owner: type) -> Callable[[], Lazy[T]]: ...

    def __get__(self, obj: Any, owner: Any) -> Any:
        if obj is None:
            return lambda *args, **kwargs: Lazy.from_class(owner, *args, **kwargs)

        def already_configured(*args, **kwargs):
            if args or kwargs:
                raise TypeError(
                    f"{type(obj).__name__} is already configured, so `as_lazy` "
                    "cannot take any arguments."
                )
            return _config_of(obj)

        return already_configured


def _config_of(obj) -> Lazy:
    cfg = getattr(obj, "_cfg", None)
    if cfg is None:
        raise AttributeError(
            f"{type(obj).__name__} was not created through Parsable (e.g. it "
            "was built with __new__, copied or unpickled), so its "
            "configuration is unavailable."
        )
    return cfg


class Parsable(Serializable, metaclass=ParsableMeta):
    """Makes a class configurable from files and from the command line.

    Every factory below is typed as returning the class itself, even the ones
    that return a :class:`Lazy` configuration. That is what lets a nested
    config be annotated with its target class::

        class Params(Parsable):
            model: Model = Model.as_lazy()

    Call :meth:`to_eager` to turn a configuration into a real object; on an
    object that is already built it is a no-op. The same goes for :meth:`copy`,
    :meth:`to_dict` and :meth:`to_file`, which work whichever of the two you
    are holding. :meth:`as_lazy` on an instance returns the configuration
    itself, for the rest of the :class:`Lazy` API.
    """

    _cfg: Lazy

    as_lazy = _AsLazy()

    def to_eager(self: T, *args, **kwargs) -> T:
        if args or kwargs:
            raise TypeError(
                f"{type(self).__name__} is already built, so `to_eager` cannot "
                "take any arguments."
            )
        return self

    def copy(self: T, fields: dict | None = None) -> Lazy[T]:
        """This configuration with `fields` changed.

        Always returns a configuration, whether it is called on one or on a
        built object - nothing is copied out of a live object except the
        arguments it was built with. Call `to_eager` to build the result.
        """
        return cast(Lazy[T], _config_of(self).copy(fields))

    def to_dict(
        self,
        *,
        flatten: bool = False,
        skip_missing: bool = False,
    ):
        return _config_of(self).to_dict(
            flatten=flatten,
            skip_missing=skip_missing,
        )

    @classmethod
    def from_dict(cls: type[T], dct: dict, *, key: str | None = None) -> T:  # type: ignore[override]
        """A configuration of this class from `dct`.

        `key` selects a nested node, dotted the same way as a command-line
        flag (`model.encoder`). The node is read as a configuration of this
        class; the file does not name one.
        """
        if key is not None:
            dct = _node(dct, key)
        return cast(T, Lazy.from_class(cls).copy(flatten_dict(dct)))

    @classmethod
    def from_file(cls: type[T], path, *, key: str | None = None) -> T:
        """A configuration of this class from a yaml or json file.

        `key` selects a nested node, as in :meth:`from_dict`.
        """
        return cast(T, cls.from_dict(load_dict(path), key=key))

    @classmethod
    def parse_args(cls: type[T], *args: Any, **kwargs: Any) -> T:
        from .parse import ArgumentParser

        parser = ArgumentParser()
        parser.add_options(cls.as_lazy(*args, **kwargs))
        return cast(T, parser.parse_args())


def _node(dct: dict, key: str) -> dict:
    """The mapping at the dotted `key`.

    A file may nest the node or spell it with dots; both name the same path.
    """
    if not key or key.startswith(".") or key.endswith(".") or ".." in key:
        raise ValueError(
            f"Invalid config key {key!r}. Use a dotted path such as 'model.encoder'."
        )

    flat = flatten_dict(dct)
    prefix = f"{key}."
    fields = {
        name.removeprefix(prefix): value
        for name, value in flat.items()
        if name.startswith(prefix)
    }
    if key in flat:
        value = flat[key]
        if fields:
            raise ValueError(
                f"{key!r} is used both as a value and as a group of values."
            )
        if isinstance(value, dict):
            return value
        raise ValueError(f"{key} is a value ({value!r}), not a nested configuration.")
    if not fields:
        raise ValueError(f"Config has no node {key!r}.")
    return fields
