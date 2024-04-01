from builtins import Ellipsis
from typing import Type, get_args, get_origin

BASIC_TYPES = (int, float, bool, str)


def is_float_type(typ: Type) -> bool:
    return typ == float


def is_int_type(typ: Type) -> bool:
    return typ == int


def is_bool_type(typ: Type) -> bool:
    return typ == bool


def is_str_type(typ: Type) -> bool:
    return typ == str


def is_flat_tuple_type(typ: Type) -> bool:
    args = get_args(typ)
    return (
        # Container is a tuple and contains inner annotation
        get_origin(typ) == tuple
        # The inner annotation is a BasicType
        and args[0] in BASIC_TYPES
        # and the follow-up annotations are of the same type, or an Ellipsis
        and all(subt in (args[0], Ellipsis) for subt in args)
    )


def is_nested_tuple_type(typ: Type) -> bool:
    args = get_args(typ)
    return (
        # Container is a tuple and contains inner annotation
        get_origin(typ) == tuple
        # The inner annotation is a flat tuple type
        and is_flat_tuple_type(args[0])
        # and if there is more annotations, the second one is an Ellipsis
        and (len(args) == 1 or args[1] == Ellipsis)
    )


def get_flat_tuple_inner_type(typ: Type[tuple]) -> tuple[Type, int]:
    args = get_args(typ)
    assert len(args) > 0
    basetype = get_args(typ)[0]
    assert basetype in BASIC_TYPES or is_flat_tuple_type(basetype)
    if Ellipsis in args:
        assert len(args) == 2
        return basetype, -1
    else:
        assert all(subt == basetype for subt in args)
        return basetype, len(args)
