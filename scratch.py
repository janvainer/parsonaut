# Allowed: int, bool, str, ...
# Containers: list[int|bool|str|...], list[list[int|bool|str|...]]

# No dicts, no more nested than 2-level list

import json
from argparse import Action, ArgumentParser
from types import UnionType
from typing import (
    Any,
    NamedTuple,
    Type,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


def _is_parsable_type(typ: Type, level=0) -> bool:

    if level > 2:
        return False

    if _is_union(typ):
        return False

    return _is_basic_type(typ) or _is_parsable_container(typ, level)


def _is_union(typ) -> bool:
    outer = get_origin(typ)
    return outer is Union or outer is UnionType


def _is_basic_type(typ) -> bool:
    return typ in (int, float, bool, str)


def _is_parsable_container(typ, level=0) -> bool:
    outer = get_origin(typ)

    # untyped list or tuple not allowed
    if outer is None:
        return False

    if outer not in (list, tuple):
        return False

    inner = get_args(typ)
    return all(_is_parsable_type(i, level + 1) for i in inner)


def _isinstance_of_parsable_type(value: Any, typ: Type) -> bool:
    BasicTypes = int | float | bool | str

    if isinstance(value, BasicTypes):
        return isinstance(value, typ)
    elif isinstance(value, list | tuple):
        outer = get_origin(typ)

        inner = get_args(typ) if outer == tuple else [get_args(typ)[0]] * len(value)
        assert len(inner) == len(value)
        return all(
            _isinstance_of_parsable_type(item, inner_typ)
            for item, inner_typ in zip(value, inner)
        )
    else:
        raise


print(_is_parsable_type(int))
print(_is_parsable_type(list))
print(_is_parsable_type(list[int]))
print(_is_parsable_type(list[list]))
print(_is_parsable_type(list[list[int]]))
exit()


# a: Union = Any -> fail for now
# a: ParsableType = Missing -> required
# b: ParsableType = matching_value_type -> ok
# (_, MissingType, value) -> check parsable type, but do not use in parser unless we are able to infer the type
# (_, MissingType, Missing) -> ignore, we cannot parse this, it must be provided to to_eager
# (_, Type1, Type2) -> fail, the value does not match annotation


ParsableType = int | float | str | bool | list | tuple
x = ("hello", str | int, "1")

Annotation = list
print(get_origin(Annotation) or Annotation)
print(issubclass(list[str], ParsableType))
# print(set() is ParsableType())
# match x:
#     case (_, typ, _) if _is_union(typ):
#         raise
#     case (
#         _,
#         typ,
#         int() | float() | str() | bool(),
#     ) if type(str) is str:
#         print("yes")
#     case _:
#         print("no")


Signature = dict[str,]

ParsableSignature = dict[
    str, tuple[Type[ParsableType], ParsableType | "ParsableSignature"]
]

parsable_signature = {
    name: (
        int,
        1,
    ),
    name: (
        list[int],
        Missing,
    ),
    name: (
        Lazy,
        {
            name: (
                int,
                1,
            ),
        },
    ),
}

[(name, typ, value), (name, Lazy, Lazy())]


exit()

BasicTypes = int | float | bool | str
ContainerTypes = (
    list[BasicTypes]
    | list[list[BasicTypes]]
    | list[tuple[BasicTypes]]
    | tuple[BasicTypes]
    | tuple[tuple[BasicTypes]]
)
Allowed = BasicTypes | ContainerTypes
MAX_LEVEL = 2

from types import UnionType
from typing import Any, Type, TypeGuard, TypeVar, Union, get_args, get_origin

V = TypeVar("V")


def _isinstance(value: Any, typ: Type[V], level: int = 0) -> TypeGuard[V]:
    if level > MAX_LEVEL:
        raise

    assert not _is_union(typ)

    if isinstance(value, BasicTypes):
        return isinstance(value, typ)

    elif isinstance(value, list | tuple):
        outer = get_origin(typ)

        # untyped list or tuple not allowed
        if outer is None:
            raise

        elif outer not in (list, tuple):
            raise

        inner = get_args(typ) if outer == tuple else [get_args(typ)[0]] * len(value)
        assert len(inner) == len(value)
        return all(
            _isinstance(item, inner_typ, level + 1)
            for item, inner_typ in zip(value, inner)
        )
    else:
        raise


def _is_union(typ) -> bool:
    outer = get_origin(typ)
    return outer is Union or outer is UnionType


print(Allowed)
print(_isinstance(["str"], list[str]))


exit()

# def _isintance(value, typ):
#     # Check int, str, float, bool, ...
#     if origin:=get_origin(typ) is None:
#         return isinstance(value, typ)

#     nested_typ = get_args(typ)
#     match value:
#         case tuple():
#             return all(
#                 _isinstance(v, t)
#                 for v, t in zip(value, nested_typ)
#             )
#         case list():
#             return all(
#                 _isinstance(v, nested_typ)
#                 for v in value
#             )


#     else:


#         return isinstance(value, origin) and all(
#             _isinstance(v, nested_typ) for v in value
#         )


print(Allowed)

parser = ArgumentParser()
parser.add_argument("-l", type=json.loads)  #'[[1,2],["foo","bar"],[3.14,"baz",20]]'
parser.add_argument("-m", type=int, nargs=3)  #'[[1,2],["foo","bar"],[3.14,"baz",20]]'

args = parser.parse_args()
print(args)
