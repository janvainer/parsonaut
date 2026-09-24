from itertools import combinations
from typing import Literal, Optional, Union

import pytest

from parsonaut.typecheck import (
    BASIC_TYPES,
    Basic,
    Literal as LiteralForm,
    Missing,
    Optional as OptionalForm,
    Tuple,
    classify,
    optional_inner_type,
)


def _fits(typ, value=Missing) -> bool:
    form = classify(typ)
    return isinstance(form, (Basic, Tuple, LiteralForm, OptionalForm)) and form.accepts(
        value
    )


@pytest.mark.parametrize("typ", BASIC_TYPES)
def test_classify_accepts_a_basic_type(typ):
    assert classify(typ) == Basic(typ)
    assert _fits(typ, typ())


def test_classify_rejects_a_basic_value_of_the_wrong_type():
    assert classify(list) is None
    assert classify(tuple[int, ...]) == Tuple(int, ...)
    assert not _fits(float, 1)
    assert not _fits(str, 1)
    assert not _fits(int, 1.0)


def test_classify_does_not_take_a_bool_for_an_int():
    # A bool is not accepted as an int, even though isinstance(True, int) is.
    assert _fits(bool, True)
    assert not _fits(int, True)
    assert not _fits(bool, 1)


@pytest.mark.parametrize("inner_typ", BASIC_TYPES)
def test_classify_accepts_a_flat_tuple(inner_typ):
    assert classify(tuple[inner_typ]) == Tuple(inner_typ, 1)
    assert classify(tuple[inner_typ, inner_typ]) == Tuple(inner_typ, 2)
    assert classify(tuple[inner_typ, ...]) == Tuple(inner_typ, ...)

    val = inner_typ()
    assert _fits(tuple[inner_typ], (val,))
    assert _fits(tuple[inner_typ, inner_typ], (val, val))
    assert _fits(tuple[inner_typ, ...], (val,))
    assert _fits(tuple[inner_typ, ...], (val, val, val))


@pytest.mark.parametrize(("typ1", "typ2"), list(combinations(BASIC_TYPES, 2)))
def test_classify_rejects_a_mixed_tuple(typ1, typ2):
    assert classify(tuple[typ1, typ2]) is None


def test_classify_rejects_an_empty_tuple():
    assert classify(tuple) is None
    assert classify(tuple[()]) is None


def test_classify_rejects_a_nested_tuple():
    assert classify(tuple[tuple[int, int, int]]) is None
    assert classify(list[int]) is None


def test_classify_rejects_bools_for_an_int_tuple():
    assert not _fits(tuple[int, ...], (True, False))
    assert _fits(tuple[bool, ...], (True, False))


def test_classify_rejects_a_tuple_value_of_the_wrong_shape():
    assert not _fits(tuple[int], (1.0,))
    assert not _fits(tuple[int], tuple())
    assert not _fits(tuple[int], (1, 1))
    assert not _fits(tuple[int, int], (1, 1.0))
    assert not _fits(tuple[int, ...], (1, 1.0))


@pytest.mark.parametrize("typ", BASIC_TYPES)
def test_classify_accepts_the_leaves(typ):
    assert isinstance(classify(typ), Basic)
    assert isinstance(classify(tuple[typ]), Tuple)
    assert isinstance(classify(tuple[typ, typ]), Tuple)
    assert isinstance(classify(tuple[typ, ...]), Tuple)


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
def test_classify_checks_values_against_optionals(typ, value, expected):
    assert _fits(typ, value) is expected


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (Literal["train", "eval"], "train"),
        (Literal[1, 2, 3], 2),
        (Literal[1.0, 2.5], 2.5),
        (Literal[False, True], False),
        (Literal[Literal["a"], "b"], "b"),
        (Literal["a"] | Literal["b"], "b"),
    ],
)
def test_classify_accepts_a_literal_of_one_basic_type(typ, value):
    form = classify(typ)
    assert isinstance(form, LiteralForm)
    assert form.accepts(value)
    assert form.typ is type(value)
    assert value in form.choices


def test_classify_rejects_a_literal_value_of_a_different_type():
    # `1 == 1.0` and `True == 1`, so the type has to be part of the check.
    assert not _fits(Literal[1, 2], 1.0)
    assert not _fits(Literal[1.0, 2.0], 1)
    assert not _fits(Literal[1, 2], True)
    assert not _fits(Literal[True], 1)
    assert not _fits(Literal["a", "b"], "c")


def test_classify_rejects_a_mixed_literal():
    assert classify(Literal["a", 1]) is None
    assert classify(Literal[b"a"]) is None
    assert classify(int) == Basic(int)


@pytest.mark.parametrize(
    "typ, value, expected",
    [
        (Literal["a", "b"] | None, None, True),
        (Literal["a", "b"] | None, "a", True),
        (Literal["a", "b"] | None, "c", False),
        (Literal["a"] | Literal["b"] | None, None, True),
        (Literal["a"] | Literal["b"] | None, "b", True),
        (Literal["a"] | Literal["b"] | None, "c", False),
        (Literal[1, 2], 1, True),
    ],
)
def test_classify_accepts_an_optional_literal(typ, value, expected):
    assert _fits(typ, value) is expected
