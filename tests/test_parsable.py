import pytest

from parsonaut.parsable import (
    Lazy,
    Missing,
    MissingType,
    Signature,
    _is_parsable_type,
    _isinstance_of_parsable_type,
    get_class_import_path,
    get_class_init_signature,
    import_class_from_path,
)
from parsonaut.testing import DummyFlat, DummyNested


def test_get_class_init_signature_flat():
    signature = get_class_init_signature(DummyFlat)
    assert signature == (
        Signature("a", MissingType, Missing),
        Signature("b", str, Missing),
        Signature("c", float, 3.14),
    )

    signature = get_class_init_signature(DummyFlat, 1, c=2.71)
    assert signature == (
        Signature("a", MissingType, 1),
        Signature("b", str, Missing),
        Signature("c", float, 2.71),
    )


def test_get_class_init_signature_nested():
    signature = get_class_init_signature(DummyNested)
    assert signature == (
        Signature("a", Lazy[DummyFlat, ...], Missing),
        Signature("b", str, Missing),
        Signature("c", float, 3.14),
    )

    a = DummyFlat(1, "2")
    signature = get_class_init_signature(DummyNested, a, c=2.71)
    assert signature == (
        Signature("a", Lazy[DummyFlat, ...], a),
        Signature("b", str, Missing),
        Signature("c", float, 2.71),
    )


def test_get_class_init_signature_raises_if_init_does_not_have_self():
    class A:
        def __init__(a, b: str, c: float = 3.14):
            pass

    with pytest.raises(AssertionError):
        get_class_init_signature(A)


def test_get_class_import_path():
    assert get_class_import_path(DummyFlat) == "parsonaut.testing.DummyFlat"
    assert get_class_import_path(DummyNested) == "parsonaut.testing.DummyNested"


def test_import_class_from_path():
    assert import_class_from_path("parsonaut.testing.DummyNested") == DummyNested

    with pytest.raises(ImportError):
        import_class_from_path("parsonaut.testing.DummyNestedNonExistent")


def test_Lazy_fails_if_provided_with_non_lazy_value():
    Lazy(DummyNested, Lazy(DummyFlat, 1, "1"))
    with pytest.raises(AssertionError):
        Lazy(DummyNested, DummyFlat(1, "1"))


def test_Lazy_to_annotated_dict_fails_for_union_type():
    class Dummy:
        def __init__(self, a: int | str):
            pass

    with pytest.raises(NotImplementedError):
        Lazy(Dummy, 1).to_annotated_dict()


def test_Lazy_to_annotated_dict_fails_for_inconsistent_annotation():
    with pytest.raises(AssertionError):
        Lazy(DummyFlat, b=1).to_annotated_dict()


def test_Lazy_to_annotated_dict_fails_for_nonparsable_annotation():

    class Dummy:
        def __init__(self, a: list):
            pass

    with pytest.raises(AssertionError):
        Lazy(Dummy, [1]).to_annotated_dict()


def test_Lazy_flat_to_annotated_dict():
    lazy = Lazy(DummyFlat, 1, "1")
    assert lazy.to_annotated_dict() == {
        "_type": DummyFlat,
        "value": {
            "a": {
                "_type": int,
                "value": 1,
            },
            "b": {
                "_type": str,
                "value": "1",
            },
            "c": {
                "_type": float,
                "value": 3.14,
            },
        },
    }


def test_Lazy_flat_to_annotated_dict_skips_unannotated_and_preserves_annotated_without_default():
    lazy = Lazy(DummyFlat)
    assert lazy.to_annotated_dict() == {
        "_type": DummyFlat,
        "value": {
            "b": {
                "_type": str,
                "value": Missing,
            },
            "c": {
                "_type": float,
                "value": 3.14,
            },
        },
    }


def test_Lazy_nested_to_annotated_dict():
    lazy = Lazy(DummyNested, Lazy(DummyFlat, 1, "1"), "hello")
    assert lazy.to_annotated_dict() == {
        "_type": DummyNested,
        "value": {
            "a": {
                "_type": DummyFlat,
                "value": {
                    "a": {
                        "_type": int,
                        "value": 1,
                    },
                    "b": {
                        "_type": str,
                        "value": "1",
                    },
                    "c": {
                        "_type": float,
                        "value": 3.14,
                    },
                },
            },
            "b": {
                "_type": str,
                "value": "hello",
            },
            "c": {
                "_type": float,
                "value": 3.14,
            },
        },
    }


def test_Lazy_nested_to_annotated_dict_skips_unannotated_and_preserves_annotated_without_default():
    lazy = Lazy(DummyNested)
    assert lazy.to_annotated_dict() == {
        "_type": DummyNested,
        "value": {
            "b": {
                "_type": str,
                "value": Missing,
            },
            "c": {
                "_type": float,
                "value": 3.14,
            },
        },
    }


def test_is_parsable_type():
    assert _is_parsable_type(int)
    assert _is_parsable_type(float)
    assert _is_parsable_type(str)
    assert _is_parsable_type(bool)

    assert _is_parsable_type(list[int])
    assert _is_parsable_type(list[float])
    assert _is_parsable_type(list[str])
    assert _is_parsable_type(list[bool])

    assert _is_parsable_type(list[list[int]])
    assert _is_parsable_type(list[list[float]])
    assert _is_parsable_type(list[list[str]])
    assert _is_parsable_type(list[list[bool]])

    assert _is_parsable_type(tuple[int, ...])
    assert _is_parsable_type(tuple[int])
    assert _is_parsable_type(tuple[float])
    assert _is_parsable_type(tuple[str])
    assert _is_parsable_type(tuple[bool])

    assert _is_parsable_type(tuple[tuple[int, ...], ...])
    assert _is_parsable_type(tuple[tuple[int]])
    assert _is_parsable_type(tuple[tuple[float]])
    assert _is_parsable_type(tuple[tuple[str]])
    assert _is_parsable_type(tuple[tuple[bool]])

    assert _is_parsable_type(list[tuple[int]])
    assert _is_parsable_type(tuple[list[int]])

    assert not _is_parsable_type(str | int)
    assert not _is_parsable_type(list)
    assert not _is_parsable_type(tuple)
    assert not _is_parsable_type(dict)
    assert not _is_parsable_type(set)
    assert not _is_parsable_type(list[list])
    assert not _is_parsable_type(list[list[list[int]]])
    assert not _is_parsable_type(DummyFlat)


def test_isinstance_of_parsable_type():
    assert _isinstance_of_parsable_type(1, int)
    assert _isinstance_of_parsable_type(1.0, float)
    assert _isinstance_of_parsable_type("hello", str)
    assert _isinstance_of_parsable_type(True, bool)

    assert _isinstance_of_parsable_type([1], list[int])
    assert _isinstance_of_parsable_type([1.0], list[float])
    assert _isinstance_of_parsable_type(["hello"], list[str])
    assert _isinstance_of_parsable_type([True], list[bool])

    assert _isinstance_of_parsable_type([[1]], list[list[int]])
    assert _isinstance_of_parsable_type([[1.0]], list[list[float]])
    assert _isinstance_of_parsable_type([["hello"]], list[list[str]])
    assert _isinstance_of_parsable_type([[True]], list[list[bool]])

    assert _isinstance_of_parsable_type((1,), tuple[int])
    assert _isinstance_of_parsable_type((1.0,), tuple[float])
    assert _isinstance_of_parsable_type(("hello",), tuple[str])
    assert _isinstance_of_parsable_type((True,), tuple[bool])

    assert _isinstance_of_parsable_type(((1,),), tuple[tuple[int]])
    assert _isinstance_of_parsable_type(((1.0,),), tuple[tuple[float]])
    assert _isinstance_of_parsable_type((("hello",),), tuple[tuple[str]])
    assert _isinstance_of_parsable_type(((True,),), tuple[tuple[bool]])

    assert _isinstance_of_parsable_type([(1,)], list[tuple[int]])
    assert _isinstance_of_parsable_type(([1],), tuple[list[int]])

    assert not _isinstance_of_parsable_type(1, float)
    assert not _isinstance_of_parsable_type(1.0, int)
    assert not _isinstance_of_parsable_type("hello", float)
    assert not _isinstance_of_parsable_type(True, str)

    assert not _isinstance_of_parsable_type([1], list[float])
    assert not _isinstance_of_parsable_type([1.0], list[int])
    assert not _isinstance_of_parsable_type(["hello"], list[float])
    assert not _isinstance_of_parsable_type([True], list[str])

    assert not _isinstance_of_parsable_type([[1]], list[list[float]])
    assert not _isinstance_of_parsable_type([[1.0]], list[list[int]])
    assert not _isinstance_of_parsable_type([["hello"]], list[list[float]])
    assert not _isinstance_of_parsable_type([[True]], list[list[str]])

    assert not _isinstance_of_parsable_type((1,), tuple[float])
    assert not _isinstance_of_parsable_type((1.0,), tuple[int])
    assert not _isinstance_of_parsable_type(("hello",), tuple[float])
    assert not _isinstance_of_parsable_type((True,), tuple[str])

    assert not _isinstance_of_parsable_type(((1,),), tuple[tuple[float]])
    assert not _isinstance_of_parsable_type(((1.0,),), tuple[tuple[int]])
    assert not _isinstance_of_parsable_type((("hello",),), tuple[tuple[float]])
    assert not _isinstance_of_parsable_type(((True,),), tuple[tuple[str]])
