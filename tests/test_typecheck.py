from itertools import combinations

import pytest

from parsonaut.typecheck import (
    BASIC_TYPES,
    get_flat_tuple_inner_type,
    is_bool_type,
    is_flat_tuple_type,
    is_float_type,
    is_int_type,
    is_nested_tuple_type,
    is_str_type,
)


def test_is_float_type():
    assert is_float_type(float)
    assert not is_float_type(int)


def test_is_int_type():
    assert is_int_type(int)
    assert not is_int_type(float)


def test_is_bool_type():
    assert is_bool_type(bool)
    assert not is_bool_type(int)


def test_is_str_type():
    assert is_str_type(str)
    assert not is_str_type(int)


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


@pytest.mark.parametrize(
    "inner_typ",
    BASIC_TYPES,
)
def test_is_nested_tuple_type_accepts_base_types(
    inner_typ,
):
    assert is_nested_tuple_type(tuple[tuple[inner_typ]])
    assert is_nested_tuple_type(tuple[tuple[inner_typ, inner_typ]])
    assert is_nested_tuple_type(tuple[tuple[inner_typ, ...]])
    assert is_nested_tuple_type(tuple[tuple[inner_typ, ...], ...])


@pytest.mark.parametrize(
    ("typ1", "typ2"),
    combinations(BASIC_TYPES, 2),
)
def test_is_nested_tuple_type_rejects_mixed_types(typ1, typ2):
    assert not is_nested_tuple_type(tuple[tuple[typ1, typ2]])


def test_is_nested_tuple_type_rejects_empty_tuple():
    assert not is_nested_tuple_type(tuple)
    assert not is_nested_tuple_type(tuple[tuple])


def test_is_nested_tuple_type_rejects_non_tuple():
    assert not is_nested_tuple_type(int)
    assert not is_nested_tuple_type(tuple[list[int]])
    assert not is_nested_tuple_type(list[tuple[int]])


def test_is_nested_tuple_type_rejects_too_deeply_nested_tuple():
    assert not is_flat_tuple_type(tuple[tuple[tuple[int]]])


def test_is_nested_tuple_type_rejects_non_consistent_subtypes():
    assert not is_flat_tuple_type(tuple[tuple[int], int])
    assert not is_flat_tuple_type(tuple[tuple[int], tuple[int]])


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
