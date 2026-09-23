from types import UnionType
from typing import Any, Union, get_args, get_origin

BASIC_TYPES = (int, float, bool, str)

#: Spellings the command line accepts for a bool. YAML leaves these as text;
#: a bool field accepts them when the value is checked against its annotation.
BOOL_TRUE_FLAGS = ("yes", "true", "t", "y", "1")
BOOL_FALSE_FLAGS = ("no", "false", "f", "n", "0")


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


def coerce_bool(value: Any) -> Any:
    """Return a bool when `value` is one, or a bool written as text.

    Anything else is returned unchanged, so the caller's own type check can
    reject it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.lower()
        if token in BOOL_TRUE_FLAGS:
            return True
        if token in BOOL_FALSE_FLAGS:
            return False
    return value


def _is_instance_of_basic(value: Any, basic_typ) -> bool:
    """`isinstance`, except that a bool is not accepted as an int.

    `isinstance(True, int)` is True, which would let `n: int = True` through
    while `b: bool = 1` is correctly rejected.
    """
    if basic_typ is not bool and isinstance(value, bool):
        return False
    return isinstance(value, basic_typ)


def is_basic_type(typ: Any, value: Any = Missing) -> bool:
    """Is `typ` one of int, float, bool or str, and does `value` fit it?

    Args:
        typ: The type to check.
        value: The value to check against the type. Defaults to `Missing`, in
            which case only the type is checked.
    """
    if typ not in BASIC_TYPES:
        return False
    return value is Missing or _is_instance_of_basic(value, typ)


def is_flat_tuple_type(typ: Any, value: Any = Missing) -> bool:
    """Check if `typ` is a flat tuple type, optionally validate a value against it.

    The inner type must be one of the basic types - int, float, bool or str.
    If more inner type values are provided, they must all be of the same type.

    Examples of passing inputs:

            - tuple[int, int], (1, 2)
            - tuple[int, ...], (1, 2, 3)

    Examples of failing inputs:

            - tuple[int, str], (1, 2)
            - tuple[int, int], (1, 2, 3)

    Args:
        typ: The type to check.
        value (Any, optional): The value to compare against the type.
            Defaults to `Missing`, in which case only the type is checked.

    Returns:
        bool: True if the type is a flat tuple type, False otherwise.
    """
    args = get_args(typ)

    if value is Missing:
        return _is_flat_tuple_type(typ, args)
    else:
        return (
            isinstance(value, tuple)
            and _is_flat_tuple_type(typ, args)
            and (len(args) == len(value) or Ellipsis in args)
            and all(_is_instance_of_basic(item, args[0]) for item in value)
        )


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

    Returns `(True, T)` if `typ` is `Union[T, None]` with a single non-`None`
    member, and `(False, typ)` otherwise. Whether a value fits is a separate
    question, answered by :func:`is_parsable_type`.
    """
    members = get_union_members(typ)
    if union_allows_none(typ) and len(members) == 1:
        return True, members[0]
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


def is_parsable_type(typ: Any, value: Any = Missing) -> bool:
    """Can a configuration record a `typ` parameter, and does `value` fit it?

    Args:
        typ: The type to check.
        value (Any, optional): The value to check against.
            Defaults to `Missing`, in which case only the type is checked.

    Returns:
        bool: True if the type is parsable, False otherwise.
    """
    is_optional, inner = optional_inner_type(typ)
    if is_optional and value is None:
        # `None` is a legal value for `Optional[T]`, so only the type is checked.
        value = Missing
    return is_basic_type(inner, value) or is_flat_tuple_type(inner, value)


def get_flat_tuple_inner_type(typ: Any) -> tuple[Any, int]:
    """
    Get the inner type and length of a flat tuple.

    The length is -1 if the tuple has an ellipsis,
    indicating that it can have any number of elements.

    Args:
        typ: The type of the tuple.

    Returns:
        tuple[Any, int]: The inner type and length of the flat tuple.

    Raises:
        TypeError: If the type is not a valid flat tuple type.

    """
    args = get_args(typ)
    if not args:
        raise TypeError(f"{typ} must have at least one inner type.")

    basetype = args[0]
    if basetype not in BASIC_TYPES:
        raise TypeError(f"The inner type of {typ} must be one of {BASIC_TYPES}.")
    if Ellipsis in args:
        if len(args) != 2:
            raise TypeError(f"{typ}: an ellipsis must be the second argument.")
        return basetype, -1

    if any(subt != basetype for subt in args):
        raise TypeError(f"{typ}: all inner types must be the same.")
    return basetype, len(args)
