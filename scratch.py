# Allowed: int, bool, str, ...
# Containers: list[int|bool|str|...], list[list[int|bool|str|...]]

# No dicts, no more nested than 2-level list    

from argparse import ArgumentParser, Action
import json



BasicTypes = int | float | bool | str
ContainerTypes = list[BasicTypes] | list[list[BasicTypes]] | list[tuple[BasicTypes]] | tuple[BasicTypes] | tuple[tuple[BasicTypes]]
Allowed = BasicTypes | ContainerTypes
MAX_LEVEL = 2

from types import UnionType
from typing import get_origin, get_args, Any, Type, TypeVar, TypeGuard, Union


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

        inner = get_args(typ) if outer == tuple else [get_args(typ)[0]]*len(value)
        assert len(inner) == len(value)
        return all(
            _isinstance(item, inner_typ, level+1)
            for item, inner_typ in zip(value, inner)
        )
    else:
        raise


def _is_union(typ) -> bool:
    outer = get_origin(typ)
    return outer is Union or outer is UnionType

print(Allowed)
print(
    _isinstance(
        ["str"], list[str]
    )
)


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
parser.add_argument('-l', type=json.loads)  #'[[1,2],["foo","bar"],[3.14,"baz",20]]'
parser.add_argument('-m', type=int, nargs=3)  #'[[1,2],["foo","bar"],[3.14,"baz",20]]'   

args = parser.parse_args()
print(args)