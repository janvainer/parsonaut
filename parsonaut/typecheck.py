from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal as TypingLiteral, Union, get_args, get_origin

BASIC_TYPES = (int, float, bool, str)


class MissingType:
    """Sentinel for "no value was provided".

    The class itself, rather than the ``Missing`` instance, is the sentinel
    for "no type annotation was provided".
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "???"

    def __reduce__(self):
        # Keeps `value is Missing` true across pickling, copying and
        # deep-copying, all of which go through __reduce__.
        return (MissingType, ())


Missing = MissingType()


def _is_instance_of_basic(value: Any, basic_typ) -> bool:
    """`isinstance`, except that a bool is not accepted as an int.

    `isinstance(True, int)` is True, which would let `n: int = True` through
    while `b: bool = 1` is correctly rejected.
    """
    if basic_typ is not bool and isinstance(value, bool):
        return False
    return isinstance(value, basic_typ)


def is_union_type(typ: Any) -> bool:
    """Check whether `typ` is a union, written either as `A | B` or `Union[A, B]`."""
    return isinstance(typ, UnionType) or get_origin(typ) is Union


def get_union_members(typ: Any) -> tuple[Any, ...]:
    """Return the non-`None` members of a union type, or `()` for a non-union."""
    if not is_union_type(typ):
        return ()
    return tuple(a for a in get_args(typ) if a is not type(None))


def union_allows_none(typ: Any) -> bool:
    return is_union_type(typ) and type(None) in get_args(typ)


def optional_inner_type(typ: Any) -> tuple[bool, Any]:
    """Unwrap `Optional[T]`.

    Returns `(True, T)` if `typ` is `T | None`. `T` may itself be a union of
    `Literal`s of one basic type, as in `Literal["a"] | Literal["b"] | None`.
    Otherwise returns `(False, typ)`. Whether a value fits is a separate
    question, answered by :func:`classify`.
    """
    members = get_union_members(typ)
    if not (union_allows_none(typ) and members):
        return False, typ
    if len(members) == 1:
        return True, members[0]
    inner = _union_of(members)
    if _literal_choices(inner) is not None:
        return True, inner
    return False, typ


def _is_flat_tuple_type(typ: Any, args):
    return (
        # Container is a tuple and contains inner annotation
        get_origin(typ) is tuple
        # A bare `tuple[()]` carries no inner annotation
        and len(args) > 0
        # The inner annotation is a BasicType
        and args[0] in BASIC_TYPES
        # the follow-up annotations are of the same type, or an Ellipsis
        and all(subt in (args[0], Ellipsis) for subt in args)
    )


def _flatten_literal(typ: Any) -> tuple[Any, ...]:
    flat: list[Any] = []
    for arg in get_args(typ):
        if get_origin(arg) is TypingLiteral:
            flat.extend(_flatten_literal(arg))
        else:
            flat.append(arg)
    return tuple(flat)


def _basic_kind(value: Any) -> type | None:
    """The basic type of `value`, with a bool kept distinct from an int."""
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, str):
        return str
    return None


def _union_of(members: tuple[Any, ...]) -> Any:
    out = members[0]
    for member in members[1:]:
        out = out | member
    return out


def _literal_parts(typ: Any) -> tuple[Any, ...] | None:
    """The `Literal`s in `typ` when it is one literal, or a union of them."""
    if get_origin(typ) is TypingLiteral:
        return (typ,)
    if is_union_type(typ) and not union_allows_none(typ):
        members = get_union_members(typ)
        if members and all(get_origin(member) is TypingLiteral for member in members):
            return members
    return None


def _literal_choices(typ: Any) -> tuple[Any, ...] | None:
    """Values of a `Literal` whose members are one basic type, or `None`.

    `Literal["a"] | Literal["b"]` is the same set of choices as
    `Literal["a", "b"]`. A mix of basic types is not.
    """
    parts = _literal_parts(typ)
    if parts is None:
        return None
    flat: list[Any] = []
    for part in parts:
        flat.extend(_flatten_literal(part))
    choices: list[Any] = []
    seen: set[tuple[type, Any]] = set()
    for item in flat:
        key = (type(item), item)
        if key not in seen:
            seen.add(key)
            choices.append(item)
    if not choices:
        return None
    kind = _basic_kind(choices[0])
    if kind is None or any(_basic_kind(item) is not kind for item in choices):
        return None
    return tuple(choices)


@dataclass(frozen=True)
class Basic:
    """`int`, `float`, `bool`, or `str`."""

    typ: type

    def accepts(self, value: Any) -> bool:
        return value is Missing or _is_instance_of_basic(value, self.typ)


@dataclass(frozen=True)
class Tuple:
    """A flat tuple of one basic type. `length` is `...` when the tuple is open."""

    typ: type
    length: Any

    def accepts(self, value: Any) -> bool:
        if value is Missing:
            return True
        return (
            isinstance(value, tuple)
            and (self.length is Ellipsis or len(value) == self.length)
            and all(_is_instance_of_basic(item, self.typ) for item in value)
        )


@dataclass(frozen=True)
class Literal:
    """One basic type, and the values a parameter may take."""

    typ: type
    choices: tuple

    def accepts(self, value: Any) -> bool:
        if value is Missing:
            return True
        return any(
            type(value) is type(choice) and value == choice for choice in self.choices
        )


@dataclass(frozen=True)
class Optional:
    """A leaf that may also be `None`. A node is never optional."""

    inner: Basic | Tuple | Literal

    def accepts(self, value: Any) -> bool:
        return value is None or value is Missing or self.inner.accepts(value)


@dataclass(frozen=True)
class Node:
    """A nested configuration. `cls` is the class the node builds."""

    cls: type | None


def _node(typ: Any) -> Node | None:
    from .lazy import Lazy
    from .parsable import Parsable

    if get_origin(typ) is Lazy:
        args = get_args(typ)
        cls = None
        if args and isinstance(args[0], type) and get_origin(args[0]) is None:
            cls = args[0]
        return Node(cls)
    if (
        isinstance(typ, type)
        and get_origin(typ) is None
        and issubclass(typ, (Lazy, Parsable))
    ):
        return Node(typ if issubclass(typ, Parsable) else None)
    return None


def classify(typ: Any) -> Basic | Tuple | Literal | Optional | Node | None:
    """The grammar production `typ` belongs to, or `None` when it is not one.

    `Model | None` and `Model | Other` raise. A node is always present.
    """
    if is_union_type(typ):
        members = get_union_members(typ)
        if any(_node(member) is not None for member in members):
            kind = (
                "optional nested configs"
                if union_allows_none(typ)
                else "unions of nested configs"
            )
            raise TypeError(f"{kind} are not supported")

    is_optional, inner = optional_inner_type(typ)
    if is_optional:
        leaf = classify(inner)
        if isinstance(leaf, (Basic, Tuple, Literal)):
            return Optional(leaf)
        return None

    if typ in BASIC_TYPES:
        return Basic(typ)

    args = get_args(typ)
    if _is_flat_tuple_type(typ, args):
        length = Ellipsis if Ellipsis in args else len(args)
        return Tuple(args[0], length)

    choices = _literal_choices(typ)
    if choices is not None:
        return Literal(type(choices[0]), choices)

    return _node(typ)
