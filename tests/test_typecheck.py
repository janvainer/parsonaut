from itertools import combinations
from typing import Optional, Union

import pytest

from parsonaut.typecheck import (
    BASIC_TYPES,
    get_flat_tuple_inner_type,
    is_basic_type,
    is_flat_tuple_type,
    is_parsable_type,
    optional_inner_type,
)


@pytest.mark.parametrize("typ", BASIC_TYPES)
def test_is_basic_type_accepts_its_own_values(typ):
    assert is_basic_type(typ)
    assert is_basic_type(typ, typ())


def test_is_basic_type_rejects_other_types_and_values():
    assert not is_basic_type(list)
    assert not is_basic_type(tuple[int, ...])
    assert not is_basic_type(float, 1)
    assert not is_basic_type(str, 1)
    assert not is_basic_type(int, 1.0)


def test_is_basic_type_does_not_take_a_bool_for_an_int():
    # A bool is not accepted as an int, even though isinstance(True, int) is.
    assert is_basic_type(bool, True)
    assert not is_basic_type(int, True)
    assert not is_basic_type(bool, 1)


@pytest.mark.parametrize(
    "inner_typ",
    BASIC_TYPES,
)
def test_is_flat_tuple_type_accepts_base_types(
    inner_typ,
):
    assert is_flat_tuple_type(tuple[inner_typ])
    assert is_flat_tuple_type(tuple[inner_typ, inner_typ])
    assert is_flat_tuple_type(tuple[inner_typ, ...])

    val = inner_typ()
    assert is_flat_tuple_type(tuple[inner_typ], (val,))
    assert is_flat_tuple_type(tuple[inner_typ, inner_typ], (val, val))
    assert is_flat_tuple_type(tuple[inner_typ, ...], (val,))
    assert is_flat_tuple_type(tuple[inner_typ, ...], (val, val, val))


@pytest.mark.parametrize(
    ("typ1", "typ2"),
    list(combinations(BASIC_TYPES, 2)),
)
def test_is_flat_tuple_type_rejects_mixed_types(typ1, typ2):
    assert not is_flat_tuple_type(tuple[typ1, typ2])


def test_is_flat_tuple_type_rejects_empty_tuple():
    assert not is_flat_tuple_type(tuple)
    # `tuple[()]` has a tuple origin but no arguments to inspect.
    assert not is_flat_tuple_type(tuple[()])
    assert not is_flat_tuple_type(tuple[()], ())


def test_is_flat_tuple_type_rejects_non_tuple():
    assert not is_flat_tuple_type(list[int])
    assert not is_flat_tuple_type(int)


def test_is_flat_tuple_type_rejects_nested_tuple():
    assert not is_flat_tuple_type(tuple[tuple[int, int, int]])


def test_is_flat_tuple_type_rejects_bools_for_int():
    # Same asymmetry as is_basic_type: a bool must not pass as an int.
    assert not is_flat_tuple_type(tuple[int, ...], (True, False))
    assert is_flat_tuple_type(tuple[bool, ...], (True, False))


def test_is_flat_tuple_type_rejects_mismatched_values():
    assert not is_flat_tuple_type(tuple[int], (1.0,))
    assert not is_flat_tuple_type(tuple[int], tuple())
    assert not is_flat_tuple_type(tuple[int], (1, 1))
    assert not is_flat_tuple_type(tuple[int, int], (1, 1.0))
    assert not is_flat_tuple_type(tuple[int, ...], (1, 1.0))


def test_get_flat_tuple_inner_type_accepted_cases():
    assert get_flat_tuple_inner_type(tuple[int]) == (int, 1)
    assert get_flat_tuple_inner_type(tuple[int, int]) == (int, 2)
    assert get_flat_tuple_inner_type(tuple[int, int, int]) == (int, 3)
    assert get_flat_tuple_inner_type(tuple[int, ...]) == (int, -1)


def test_get_flat_tuple_inner_type_raises_on_invalid_cases():
    with pytest.raises(TypeError, match="at least one inner type"):
        get_flat_tuple_inner_type(tuple)

    # A tuple of tuples is not flat, so `is_flat_tuple_type` never offers one.
    with pytest.raises(TypeError, match="inner type"):
        get_flat_tuple_inner_type(tuple[tuple[int, int], ...])

    with pytest.raises(TypeError, match="all inner types must be the same"):
        get_flat_tuple_inner_type(tuple[str, int])


@pytest.mark.parametrize(
    "typ",
    BASIC_TYPES,
)
def test_is_parsable_type_accepts(typ):
    assert is_parsable_type(typ)
    assert is_parsable_type(tuple[typ])
    assert is_parsable_type(tuple[typ, typ])
    assert is_parsable_type(tuple[typ, ...])


@pytest.mark.parametrize(
    "typ, expected",
    [
        (int | None, (True, int)),
        (Optional[str], (True, str)),
        (tuple[int, ...] | None, (True, tuple[int, ...])),
        (int, (False, int)),
        # More than one member left over, so there is no single inner type.
        (Union[int, str], (False, Union[int, str])),
        (Union[int, None, str], (False, Union[int, None, str])),
    ],
)
def test_optional_inner_type(typ, expected):
    assert optional_inner_type(typ) == expected


@pytest.mark.parametrize(
    "typ, value, expected",
    [
        (int | None, None, True),
        (int | None, 5, True),
        (int | None, "hi", False),
        (Optional[str], None, True),
        (Optional[str], 123, False),
        (int, 42, True),
        # None is only a value for an optional annotation.
        (int, None, False),
        (tuple[int, ...] | None, (1, 2), True),
        (tuple[int, ...] | None, (1, "x"), False),
        (Union[int, str], "hello", False),
    ],
)
def test_is_parsable_type_checks_values_against_optionals(typ, value, expected):
    assert is_parsable_type(typ, value) is expected
