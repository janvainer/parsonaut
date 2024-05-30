from builtins import Ellipsis
from functools import lru_cache
from typing import Any, Type, get_args, get_origin

BASIC_TYPES = (int, float, bool, str)


class MissingType:
    pass


Missing = MissingType()


def _is_basic_type(typ: Type, basic_typ, value: Any | None = None) -> bool:
    assert basic_typ in BASIC_TYPES
    typ_ok = typ == basic_typ
    if value is None:
        return typ_ok
    else:
        return typ_ok and isinstance(value, basic_typ)


def is_float_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, float, value)


def is_int_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, int, value)


def is_bool_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, bool, value)


def is_str_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, str, value)


def is_flat_tuple_type(typ: Type, value: Any | None = None) -> bool:
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
        typ (Type): The type to check.
        value (Any | None, optional): The value to compare against the type. Defaults to None.

    Returns:
        bool: True if the type is a flat tuple type, False otherwise.
    """
    args = get_args(typ)

    if value is None:
        return _is_flat_tuple_type(typ, args)
    else:
        return (
            _is_flat_tuple_type(typ, args)
            and (len(args) == len(value) or Ellipsis in args)
            and all(isinstance(item, args[0]) for item in value)
        )


@lru_cache(maxsize=1)
def _is_flat_tuple_type(typ: Type, args):
    return (
        # Container is a tuple and contains inner annotation
        get_origin(typ) == tuple
        # The inner annotation is a BasicType
        and args[0] in BASIC_TYPES
        # the follow-up annotations are of the same type, or an Ellipsis
        and all(subt in (args[0], Ellipsis) for subt in args)
    )


def is_parsable_type(typ: Type, value: Any | None = None) -> bool:
    """Check if the given type is parsable.

    Args:
        typ (Type): The type to check.
        value (Any | None, optional): The value to check against. Defaults to None.

    Returns:
        bool: True if the type is parsable, False otherwise.
    """
    return any(
        (
            is_int_type(typ, value),
            is_float_type(typ, value),
            is_bool_type(typ, value),
            is_str_type(typ, value),
            is_flat_tuple_type(typ, value),
        )
    )


def get_flat_tuple_inner_type(typ: Type[tuple]) -> tuple[Type, int]:
    """
    Get the inner type and length of a flat tuple.

    The length is -1 if the tuple has an ellipsis,
    indicating that it can have any number of elements.

    Args:
        typ (Type[tuple]): The type of the tuple.

    Returns:
        tuple[Type, int]: A tuple containing the inner type and length of the flat tuple.

    Raises:
        AssertionError: If the type is not a valid flat tuple type.

    """
    args = get_args(typ)
    assert len(args) > 0, "Tuple type must have at least one argument."
    basetype = get_args(typ)[0]
    assert basetype in BASIC_TYPES or is_flat_tuple_type(basetype), (
        "The inner type must be one of the basic types: "
        f"{BASIC_TYPES} or a flat tuple type."
    )
    if Ellipsis in args:
        assert len(args) == 2, "Ellipsis must be the second argument."
        return basetype, -1
    else:
        assert all(
            subt == basetype for subt in args
        ), "All inner types must be the same."
        return basetype, len(args)
