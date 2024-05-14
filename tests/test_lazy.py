from functools import partial

import pytest

from parsonaut.lazy import (
    Lazy,
    Missing,
    MissingType,
    Parsable,
    flatten_dict,
    get_signature,
    set_typecheck_eager,
    should_typecheck_eagerly,
    typecheck_eager,
    unflatten_dict,
)


class DummyFlat(Parsable):
    def __init__(self, a, b: str, c: float = 3.14):
        pass


class DummyNested(Parsable):
    def __init__(self, a: str, b: Lazy[DummyFlat, ...], c: float = 3.14):
        pass


def test_get_class_init_signature_flat():

    signature = get_signature(DummyFlat.__init__)
    assert signature == {
        "a": (MissingType, Missing),
        "b": (str, Missing),
        "c": (float, 3.14),
    }

    signature = get_signature(DummyFlat.__init__, 1, c=2.71)
    assert signature == {
        "a": (MissingType, 1),
        "b": (str, Missing),
        "c": (float, 2.71),
    }

    def dummy_func(a, b: str, c: float = 3.14):
        pass

    signature = get_signature(dummy_func, 1, c=2.71)
    assert signature == {
        "a": (MissingType, 1),
        "b": (str, Missing),
        "c": (float, 2.71),
    }


def test_get_class_init_signature_nested():

    signature = get_signature(DummyNested.__init__)
    assert signature == {
        "a": (str, Missing),
        "b": (Lazy[DummyFlat, ...], Missing),
        "c": (float, 3.14),
    }

    b = DummyFlat(1, "2")
    signature = get_signature(DummyNested.__init__, b=b, c=2.71)
    assert signature == {
        "a": (str, Missing),
        "b": (Lazy[DummyFlat, ...], b),
        "c": (float, 2.71),
    }


def test_set_typecheck_eagerly():
    set_typecheck_eager(True)
    assert should_typecheck_eagerly() is True

    set_typecheck_eager(False)
    assert should_typecheck_eagerly() is False

    # Set to default so that other tests work fine
    set_typecheck_eager()


def test_typecheck_eager_context():

    set_typecheck_eager(True)
    with typecheck_eager():
        assert should_typecheck_eagerly()
    assert not should_typecheck_eagerly()

    # Set to default so that other tests work fine
    set_typecheck_eager(False)


def test_eager_signature_check():
    lazy_dummy = Lazy.from_class(DummyFlat)
    assert isinstance(lazy_dummy._signature, partial)

    with typecheck_eager():
        lazy_dummy = Lazy.from_class(DummyFlat)
        assert isinstance(lazy_dummy._signature, dict)


def test_Lazy__eq__():
    s1 = Lazy(DummyNested, Lazy.get_signature(DummyNested))
    s2 = Lazy(DummyNested, Lazy.get_signature(DummyNested))
    s3 = Lazy(DummyNested, Lazy.get_signature(DummyNested, a="hello"))
    assert s1 == s2
    assert s1 != s3


def test_Lazy_get_signature():
    assert Lazy.get_signature(DummyFlat) == {
        "b": (str, Missing),
        "c": (float, 3.14),
    }

    assert Lazy.get_signature(DummyNested) == {
        "a": (str, Missing),
        "b": (Lazy[DummyFlat, ...], Lazy.from_class(DummyFlat)),
        "c": (float, 3.14),
    }

    assert Lazy.get_signature(DummyNested, a="hello") != {
        "a": (str, Missing),
        "b": (DummyFlat, Lazy.from_class(DummyFlat)),
        "c": (float, 3.14),
    }


def test_Lazy_get_signature_raises_for_invalid_type():

    class GoodDummy:
        def __init__(self, a: Lazy[DummyFlat, ...]) -> None:
            pass

    Lazy.get_signature(GoodDummy)

    class BadDummy:
        # GoodDummy is not Parsable so it should raise
        def __init__(self, a: Lazy[GoodDummy, ...]) -> None:
            pass

    with pytest.raises(AssertionError):
        Lazy.get_signature(BadDummy)


def test_Lazy_get_signature_fills_in_type_for_lazy_default():

    class Dummy:
        def __init__(self, a=Lazy.from_class(DummyFlat)) -> None:
            pass

    lazy_dummy = Lazy.get_signature(Dummy)
    assert lazy_dummy["a"][0] == Lazy


def test_Lazy_get_signature_fails_if_lazy_default_has_wrong_annotation():

    class Dummy:
        def __init__(self, a: DummyFlat = Lazy.from_class(DummyFlat)) -> None:
            pass

    with pytest.raises(AssertionError):
        Lazy.get_signature(Dummy)


def test_Lazy_from_class():
    s1 = Lazy(DummyNested, Lazy.get_signature(DummyNested))
    s2 = Lazy.from_class(DummyNested)
    assert s1 == s2

    s1 = Lazy(DummyNested, Lazy.get_signature(DummyNested, a="hello"))
    s2 = Lazy.from_class(DummyNested, a="hello")
    assert s1 == s2


def test_Lazy_to_dict():
    assert Lazy.from_class(DummyNested).to_dict() == {
        "b": {"c": 3.14},
        "c": 3.14,
    }
    assert Lazy.from_class(DummyNested).to_dict(with_class_tag=True) == {
        "_class": DummyNested,
        "b": {"_class": DummyFlat, "c": 3.14},
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(with_annotations=True) == {
        "a": (str, Missing),
        "b": {
            "b": (str, Missing),
            "c": (float, 3.14),
        },
        "c": (float, 3.14),
    }

    assert Lazy.from_class(DummyNested).to_dict(flatten=True) == {
        "b.c": 3.14,
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(recursive=False) == {
        "b": Lazy.from_class(DummyFlat),
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(
        with_annotations=True, with_class_tag=True
    ) == {
        "_class": DummyNested,
        "a": (str, Missing),
        "b": {
            "_class": DummyFlat,
            "b": (str, Missing),
            "c": (float, 3.14),
        },
        "c": (float, 3.14),
    }

    assert Lazy.from_class(DummyNested).to_dict(
        with_annotations=True, with_class_tag=True, flatten=True
    ) == {
        "_class": DummyNested,
        "a": (str, Missing),
        "b._class": DummyFlat,
        "b.b": (str, Missing),
        "b.c": (float, 3.14),
        "c": (float, 3.14),
    }


def test_Lazy_from_dict_nested():
    assert Lazy.from_dict(
        {
            "_class": DummyNested,
            "b": {"_class": DummyFlat, "c": 3.14},
            "c": 3.14,
        }
    ) == Lazy.from_class(DummyNested)


def test_Lazy_from_dict_flat():
    assert Lazy.from_dict(
        {
            "_class": DummyNested,
            "b._class": DummyFlat,
            "b.c": 3.14,
            "c": 3.14,
        }
    ) == Lazy.from_class(DummyNested)


def test_flatten_dict():
    assert flatten_dict({"1": "2", "3": {"4": "5"}}) == {"1": "2", "3.4": "5"}


def test_unflatten_dict():
    assert unflatten_dict({"1": "2", "3.4": "5"}) == {"1": "2", "3": {"4": "5"}}


def test_Lazy_skips_nonparsable_without_defaults():
    class DummyFlat(Lazy):
        def __init__(self, a: list[str]):
            pass

    s = Lazy.from_class(DummyFlat)
    assert "a" not in s.signature


def test_Lazy_fails_if_provided_with_non_parsable_default():
    class DummyFlat(Lazy):
        def __init__(self, a: list[str] = [1]):  # type: ignore
            pass

    with pytest.raises(AssertionError):
        with typecheck_eager():
            Lazy.from_class(DummyFlat)


def test_Lazy_fails_if_provided_with_inconsistent_annotation():
    class DummyFlat(Lazy):
        def __init__(self, a: str = 1):  # type: ignore
            pass

    with pytest.raises(AssertionError):
        with typecheck_eager():
            Lazy.from_class(DummyFlat)
