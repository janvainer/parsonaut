from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from .dicts import flatten_dict, unflatten_dict
from .serialization import Serializable
from .typecheck import (
    Missing,
    MissingType,
    Node,
    classify,
    is_union_type,
)

T = TypeVar("T")
B = TypeVar("B")

#: A bound signature: parameter name -> (annotation, value).
Signature = Mapping[str, tuple[Type, Any]]

_TYPECHECK_EAGER: ContextVar[bool] = ContextVar(
    "parsonaut_typecheck_eager", default=False
)

# Filled in on first use to avoid a circular import with `parsonaut.parsable`.
_PARSABLE_BASE: type | None = None
_SCHEMA: dict[Any, dict[str, "Field"]] = {}
# Classes whose schema is currently being built, so a cycle can be reported
# instead of recursing until the stack overflows.
_SCHEMA_STACK: list = []


def _parsable_base() -> type:
    global _PARSABLE_BASE
    if _PARSABLE_BASE is None:
        from .parsable import Parsable

        _PARSABLE_BASE = Parsable
    return _PARSABLE_BASE


@dataclass(frozen=True)
class Field:
    """One recorded parameter of a class: its annotation and default."""

    typ: Any
    default: Any


def schema_of(cl) -> dict[str, Field]:
    """The configurable parameters of `cl`, cached after the first successful build."""
    if cl in _SCHEMA:
        return _SCHEMA[cl]
    if cl in _SCHEMA_STACK:
        start = _SCHEMA_STACK.index(cl)
        names = [_type_name(item) for item in (*_SCHEMA_STACK[start:], cl)]
        raise TypeError(
            f"{' -> '.join(names)} refers to itself, so a default "
            "configuration cannot be inferred. Pass an explicit default "
            "instead."
        )
    _SCHEMA_STACK.append(cl)
    try:
        schema = _build_schema(cl)
    finally:
        _SCHEMA_STACK.pop()
    _SCHEMA[cl] = schema
    return schema


def _build_schema(cl) -> dict[str, Field]:
    func = cl.__init__ if isinstance(cl, type) else cl
    bound, _, _ = _bind(func, (), {})
    fields = {}
    for name, (typ, value) in bound.items():
        field = _try_field(cl, name, typ, value, provided=False)
        if field is not None:
            fields[name] = field
    return fields


def _try_field(cl, name: str, typ, value, *, provided: bool) -> Field | None:
    try:
        form = classify(typ)
    except TypeError as exc:
        raise TypeError(f"{_where(cl)}: {exc} ({name}: {_type_name(typ)}).") from exc

    if (isinstance(form, Node) and form.cls is not None) or isinstance(value, Lazy):
        nested_cls = form.cls if isinstance(form, Node) else None
        typ, value = _resolve_nested(cl, name, typ, nested_cls, value)
        _reject_reserved_name(cl, name)
        return Field(typ, value)

    if typ is MissingType or form is None:
        reason = (
            "it has no type annotation"
            if typ is MissingType
            else f"{_type_name(typ)} is not a configurable type"
        )
        if provided:
            raise TypeError(
                f"{_where(cl)}: cannot record a value for {name!r} in a "
                f"configuration, because {reason}. Pass it to `to_eager()` "
                "instead."
            )
        return None

    if not form.accepts(value):
        raise TypeError(
            f"Provided value {name}={value!r} does not match "
            f"the provided annotation {name}: {_type_name(typ)}"
        )
    _reject_reserved_name(cl, name)
    return Field(typ, value)


class Lazy(Generic[T], Serializable):
    """A frozen, serializable description of how to build a `T`.

    Think of it as a nested `functools.partial`: it remembers the target class
    together with the (possibly incomplete) arguments for its `__init__`, and
    builds the real object only when :meth:`to_eager` is called.
    """

    def __init__(self, cls: Type[T] | Callable[..., T], signature: partial | Signature):
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "_signature", signature)

    def _identity(self) -> tuple:
        return (self.cls,) + tuple(
            (name, typ, value._identity() if isinstance(value, Lazy) else value)
            for name, (typ, value) in sorted(self.signature.items())
        )

    def __hash__(self) -> int:
        return hash(self._identity())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lazy):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        return _lazy_str(self)

    def __repr__(self):
        try:
            return self.__str__()
        except Exception as exc:  # noqa: BLE001 - reporting beats raising here
            return f"<invalid {_type_name(self.cls)} configuration: {exc}>"

    def __getattr__(self, x):
        if x.startswith("_"):
            raise AttributeError(x)

        signature = object.__getattribute__(self, "signature")
        if x not in signature:
            raise AttributeError(
                f"{_type_name(self.cls)} has no parameter {x!r}. It takes "
                f"{', '.join(sorted(signature)) or 'no parameters'}."
            )
        return signature[x][1]

    def __setattr__(self, name, value):
        raise AttributeError(
            f"{_type_name(type(self))} is frozen; build the object with "
            "`to_eager()` or make a changed copy with `copy()`."
        )

    @property
    def signature(self) -> Signature:
        _signature = object.__getattribute__(self, "_signature")
        if isinstance(_signature, partial):
            _signature = _signature()
            object.__setattr__(self, "_signature", _signature)
        return _signature

    def as_lazy(self: "Lazy[B]") -> "Lazy[B]":
        return self

    @staticmethod
    def from_class(cl: Type[B] | Callable[..., B], /, *args, **kwargs) -> "Lazy[B]":
        return new_lazy(cl, args, kwargs)

    @staticmethod
    def get_signature(cl, /, *args, **kwargs) -> Signature:
        return _build_signature(cl, args, kwargs)

    def copy(self: "Lazy[B]", fields: dict | None = None) -> "Lazy[B]":
        """This configuration with `fields` changed.

        Keys may be nested or dotted. Unknown keys are errors.
        """
        return apply(self, fields or {})

    def to_dict(
        self,
        *,
        flatten: bool = False,
        skip_missing: bool = False,
    ):
        dct = dict()
        for k, (typ, value) in sorted(self.signature.items()):
            if value is Missing and skip_missing:
                continue

            if isinstance(classify(typ), Node):
                dct[k] = cast(Lazy, value).to_dict(skip_missing=skip_missing)
                continue

            dct[k] = value

        return flatten_dict(dct) if flatten else dct

    def _own_values(self) -> dict[str, Any]:
        return {name: value for name, (_, value) in self.signature.items()}

    def to_eager(self, *args, **kwargs) -> T:
        if args:
            raise TypeError(
                f"{_type_name(self.cls)} takes named parameters only here; "
                f"got {len(args)} positional one(s)."
            )
        kwargs = {**self._own_values(), **kwargs}
        unset = sorted(k for k, v in kwargs.items() if v is Missing)
        if unset:
            raise TypeError(
                f"{_type_name(self.cls)} has no value for "
                f"{', '.join(repr(k) for k in unset)}. Set "
                f"{'them' if len(unset) > 1 else 'it'} on the command line "
                f"({' '.join(f'--{k}' for k in unset)}), in a config file, or "
                f"pass {'them' if len(unset) > 1 else 'it'} to `to_eager()`."
            )
        return self.cls(**kwargs)


def new_lazy(cl, args=(), kwargs=None, *, strict: bool = True) -> Lazy:
    # The signature is built on first use, so take a copy now. A later
    # mutation of a list or dict the caller still holds stays outside.
    args = tuple(_snapshot(arg) for arg in args)
    kwargs = _snapshot(kwargs or {})
    if _TYPECHECK_EAGER.get():
        return Lazy(cl, _build_signature(cl, args, kwargs, strict=strict))
    return Lazy(cl, partial(_build_signature, cl, args, kwargs, strict=strict))


def _snapshot(value):
    """A shallow copy of lists and dicts, so later mutation stays outside."""
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


def _build_signature(cl, args=(), kwargs=None, *, strict: bool = True) -> Signature:
    schema = schema_of(cl)
    func = cl.__init__ if isinstance(cl, type) else cl
    bound, provided, extras = _bind(func, args, kwargs or {})

    # *args / **kwargs have no field to store. A live object still receives
    # them; a configuration can only keep them when the caller passes them to
    # `to_eager()` later.
    if extras and strict:
        raise TypeError(
            f"{_where(cl)}: cannot record {_unrecordable_extras(extras)} in a "
            "configuration. Pass them to `to_eager()` instead."
        )

    for name in provided:
        if name not in schema:
            typ, value = bound[name]
            _try_field(cl, name, typ, value, provided=strict)

    res = dict()
    for name, field in schema.items():
        if name in provided:
            value = _updated_value(name, field.typ, field.default, bound[name][1])
        else:
            value = field.default
            if isinstance(value, Lazy):
                value = value.copy()
        res[name] = (field.typ, value)
    return res


def apply(node: Lazy, updates: dict, source: str | None = None) -> Lazy:
    """Return `node` with `updates` applied.

    Keys may be nested or dotted. Unknown keys are errors.
    """
    try:
        return _apply(node, unflatten_dict(flatten_dict(updates)), "")
    except (TypeError, ValueError) as exc:
        if source is None:
            raise
        message = exc.args[0] if exc.args else exc
        raise type(exc)(f"{source}: {message}") from exc


def _apply(node: Lazy, updates: dict, path: str) -> Lazy:
    signature = node.signature
    for key in updates:
        if key not in signature:
            raise ValueError(
                f"Attempted to copy with field='{path}{key}' that is not present."
            )

    rebuilt = {}
    for key, (typ, value) in signature.items():
        if key in updates:
            value = _updated_value(f"{path}{key}", typ, value, updates[key])
        elif isinstance(value, Lazy):
            value = _apply(value, {}, f"{path}{key}.")
        rebuilt[key] = (typ, value)

    return Lazy(node.cls, rebuilt)


def _updated_value(where: str, typ, current, update):
    if isinstance(update, dict):
        if not isinstance(current, Lazy):
            raise TypeError(
                f"Attempted to copy {where} with a dict of fields, but it is "
                "not a nested configuration."
            )
        return _apply(current, update, f"{where}.")

    form = classify(typ)
    if isinstance(form, Node):
        config = _as_config(update)
        if not isinstance(config, Lazy):
            raise TypeError(
                f"Attempted to copy {where} with {update!r}, which is not a "
                "configuration."
            )
        expected = form.cls
        if (
            expected is not None
            and config.cls is not expected
            and not _is_subclass(config.cls, expected)
        ):
            raise TypeError(
                f"Attempted to copy {where} with a {_type_name(config.cls)} "
                f"configuration, but {where} is annotated as {_type_name(typ)}."
            )
        return config

    if form is None or not form.accepts(update):
        raise TypeError(
            f"Provided value {where}={update!r} does not match "
            f"the provided annotation {where}: {_type_name(typ)}"
        )
    return update


def _as_config(value):
    if isinstance(value, Lazy):
        return value
    if isinstance(value, _parsable_base()):
        return value.as_lazy()
    return value


def is_nested_type(typ) -> bool:
    """Is `typ` an annotation for a nested (lazily built) config?"""
    return isinstance(classify(typ), Node)


def _is_class(typ) -> bool:
    return isinstance(typ, type) and get_origin(typ) is None


def _is_subclass(typ, base: type) -> bool:
    return _is_class(typ) and issubclass(typ, base)


def _resolve_nested(cl, name: str, typ, nested_cls: type | None, value):
    where = _where(cl)

    if value is Missing:
        return typ, _inferred_default(where, name, typ, nested_cls)

    value = _as_config(value)
    if not isinstance(value, Lazy):
        raise TypeError(
            f"{where}: expected {name} to default to a Parsable config "
            f"(e.g. `SomeParsable.as_lazy()`), got {type(value).__name__}."
        )

    if nested_cls is None:
        if not is_nested_type(typ):
            annotation = (
                "has no annotation"
                if typ is MissingType
                else f"is annotated with {_type_name(typ)}"
            )
            raise TypeError(
                f"{where}: {name} defaults to a config but {annotation}. Annotate "
                "it with the Parsable subclass it builds, or with "
                f"`Lazy[SomeClass]` for a class that is not one."
            )
        return typ, value

    if value.cls is not nested_cls and not _is_subclass(value.cls, nested_cls):
        raise TypeError(
            f"{where}: {name} defaults to a {_type_name(value.cls)} config, "
            f"but {name} is annotated as {_type_name(typ)}."
        )
    return typ, value


def _inferred_default(where: str, name: str, typ, nested_cls: type | None) -> Lazy:
    if nested_cls is None:
        raise TypeError(
            f"{where}: cannot infer a default for {name}: {_type_name(typ)}. "
            "Annotate it with a Parsable subclass, or give it a default such "
            "as `SomeParsable.as_lazy()`."
        )

    if _is_subclass(nested_cls, _parsable_base()):
        # Resolve it now, while `cl` is still on the schema stack, so a class
        # that refers back to one being built fails here instead of recursing
        # when the default is later copied.
        schema_of(nested_cls)
        return Lazy.from_class(nested_cls)

    raise TypeError(
        f"{where}: cannot create a default for {name}: "
        f"{_type_name(typ)} because {_type_name(nested_cls)} does not "
        "subclass Parsable."
    )


def _where(cl) -> str:
    return f"{_type_name(cl)}.__init__" if isinstance(cl, type) else _type_name(cl)


def _type_name(typ) -> str:
    if is_union_type(typ) or get_origin(typ) is not None:
        return str(typ)
    return getattr(typ, "__name__", None) or str(typ)


_RESERVED_NAMES = frozenset(
    {
        "as_lazy",
        "cls",
        "copy",
        "from_class",
        "from_dict",
        "from_file",
        "get_signature",
        "signature",
        "to_dict",
        "to_eager",
        "to_file",
    }
)


def _reject_reserved_name(cl, name: str) -> None:
    if name.startswith("_"):
        raise TypeError(
            f"{_where(cl)} has a parameter named {name!r}. A configurable "
            "parameter cannot start with an underscore, because it would be "
            "unreachable on the configuration. Please rename it."
        )
    if name in _RESERVED_NAMES:
        raise TypeError(
            f"{_where(cl)} has a parameter named {name!r}, which clashes with "
            "the Lazy API and would be unreachable. Please rename it."
        )


@contextmanager
def typecheck_eager(eager: bool = True):
    """Validate signatures immediately instead of on first use."""
    token = _TYPECHECK_EAGER.set(eager)
    try:
        yield
    finally:
        _TYPECHECK_EAGER.reset(token)


def _bind(
    func: Callable, args: tuple, kwargs: dict
) -> tuple[dict[str, tuple[Type, Any]], set[str], dict[str, Any]]:
    sig = signature(func)
    first, *_ = [*sig.parameters, None]
    if first == "self":
        bound = sig.bind_partial(None, *args, **kwargs)
    else:
        bound = sig.bind_partial(*args, **kwargs)

    explicit = set(bound.arguments) - {"self"}
    bound.apply_defaults()

    ret = dict()
    var_names = set()
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            var_names.add(param_name)
            continue

        value = bound.arguments.get(
            param_name,
            param.default if param.default is not Parameter.empty else Missing,
        )
        annotation = (
            param.annotation if param.annotation is not Parameter.empty else MissingType
        )
        ret[param_name] = (annotation, value)

    extras = {name: bound.arguments[name] for name in explicit & var_names}
    return ret, explicit - var_names, extras


def _unrecordable_extras(extras: dict[str, Any]) -> str:
    """The extra positional and keyword arguments a config cannot store."""
    parts: list[str] = []
    for value in extras.values():
        if isinstance(value, dict):
            parts.extend(repr(key) for key in sorted(value))
        elif value:
            count = len(value)
            noun = "argument" if count == 1 else "arguments"
            parts.append(f"{count} positional {noun}")
    return ", ".join(parts)


def _lazy_str(node: Lazy, level: int = 1):

    def format_attr(k, v):
        return f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"

    header = _type_name(node.cls)
    attrs = []
    for k, (_, value) in sorted(node.signature.items()):
        if isinstance(value, Lazy):
            attrs.append(f"{k}={_lazy_str(value, level=level + 1)}")
        else:
            attrs.append(format_attr(k, value))
    if not attrs:
        return f"{header}()"
    indent = "    "
    attrs = f",\n{indent * level}".join(attrs)
    out = f"{header}(\n{indent * level}{attrs},\n{indent * (level - 1)})"
    return out
