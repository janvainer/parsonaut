from itertools import combinations

import pytest

from parsonaut.typecheck import (
    BASIC_TYPES,
    get_flat_tuple_inner_type,
    is_bool_type,
    is_flat_tuple_type,
    is_float_type,
    is_int_type,
    is_parsable_type,
    is_str_type,
)


def test_is_float_type():
    assert is_float_type(float)
    assert is_float_type(float, 1.0)

    assert not is_float_type(int)
    assert not is_float_type(float, 1)


def test_is_int_type():
    assert is_int_type(int)
    assert is_int_type(int, 1)
    assert is_int_type(int, True)  # isinstance(True, int) == True

    assert not is_int_type(float)
    assert not is_int_type(int, 1.0)


def test_is_bool_type():
    assert is_bool_type(bool)
    assert is_bool_type(bool, True)
    assert not is_bool_type(bool, 0)
    assert not is_bool_type(bool, 1)
    assert not is_bool_type(int)


def test_is_str_type():
    assert is_str_type(str)
    assert is_str_type(str, "hello")
    assert not is_str_type(int)
    assert not is_str_type(str, 1)


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
    combinations(BASIC_TYPES, 2),
)
def test_is_flat_tuple_type_rejects_mixed_types(typ1, typ2):
    assert not is_flat_tuple_type(tuple[typ1, typ2])


def test_is_flat_tuple_type_rejects_empty_tuple():
    assert not is_flat_tuple_type(tuple)


def test_is_flat_tuple_type_rejects_non_tuple():
    assert not is_flat_tuple_type(list[int])
    assert not is_flat_tuple_type(int)


def test_is_flat_tuple_type_rejects_nested_tuple():
    assert not is_flat_tuple_type(tuple[tuple[int, int, int]])


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

    assert get_flat_tuple_inner_type(tuple[tuple[int], ...]) == (tuple[int], -1)
    assert get_flat_tuple_inner_type(tuple[tuple[int, ...], ...]) == (
        tuple[int, ...],
        -1,
    )
    assert get_flat_tuple_inner_type(tuple[tuple[int, int], ...]) == (
        tuple[int, int],
        -1,
    )


def test_get_flat_tuple_inner_type_raises_on_invalid_cases():
    with pytest.raises(AssertionError):
        get_flat_tuple_inner_type(tuple)

    with pytest.raises(AssertionError):
        get_flat_tuple_inner_type(tuple[tuple])

    with pytest.raises(AssertionError):
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
