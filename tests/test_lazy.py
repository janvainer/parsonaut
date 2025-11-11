from dataclasses import dataclass
from functools import partial

import pytest

from parsonaut import Parsable
from parsonaut.lazy import (
    Lazy,
    Missing,
    MissingType,
    flatten_dict,
    get_signature,
    set_typecheck_eager,
    should_typecheck_eagerly,
    typecheck_eager,
    unflatten_dict,
)


class DummyFlat(Parsable):
    def __init__(self, a, b: str, c: float = 3.14):
        self.a = a
        self.b = b
        self.c = c


class DummyNested(Parsable):
    def __init__(self, a: str, b: Lazy[DummyFlat, ...], c: float = 3.14):
        self.a = a
        self.b = b.to_eager(a=a)
        self.c = c


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
        "a": Missing,
        "b": {"c": 3.14, "b": Missing},
        "c": 3.14,
    }
    assert Lazy.from_class(DummyNested).to_dict(with_class_tag=True) == {
        "_class": DummyNested,
        "a": Missing,
        "b": {"_class": DummyFlat, "c": 3.14, "b": Missing},
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
        "a": Missing,
        "b.b": Missing,
        "b.c": 3.14,
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(recursive=False) == {
        "a": Missing,
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


def test_Lazy_fails_if_provided_with_inconsistent_annotation():
    class DummyFlat(Lazy):
        def __init__(self, a: str = 1):  # type: ignore
            pass

    with pytest.raises(AssertionError):
        with typecheck_eager():
            Lazy.from_class(DummyFlat)


def test_Lazy_on_dataclasses():

    @dataclass
    class InnerDummy:
        a: str = "hello"
        b: int = 1

    lazy_dummy = Lazy.from_class(InnerDummy)
    assert lazy_dummy.signature == {"a": (str, "hello"), "b": (int, 1)}

    @dataclass
    class OuterDummy:
        c: Lazy[InnerDummy, ...] = Lazy.from_class(InnerDummy)
        d: tuple[int, ...] = (1, 2, 3)

    lazy_dummy = Lazy.from_class(OuterDummy)
    assert lazy_dummy.signature == {
        "c": (Lazy[InnerDummy, ...], Lazy.from_class(InnerDummy)),
        "d": (tuple[int, ...], (1, 2, 3)),
    }


def test_Lazy_to_eager():
    lazy_dummy = Lazy.from_class(DummyFlat)

    # Missing kwargs
    with pytest.raises(TypeError):
        lazy_dummy.to_eager()

    # only kwargs are allowed, not args
    with pytest.raises(AssertionError):
        lazy_dummy.to_eager(1, "hello", 1.0)

    dummy = lazy_dummy.to_eager(a=1, b="hello", c=1.0)
    assert isinstance(dummy, DummyFlat)
    assert dummy.a == 1
    assert dummy.b == "hello"
    assert dummy.c == 1.0


def test_Parsable_as_lazy():
    assert DummyFlat.as_lazy() == Lazy.from_class(DummyFlat)
    assert DummyFlat.as_lazy(b="hello") == Lazy.from_class(DummyFlat, b="hello")

    assert DummyNested.as_lazy() == Lazy.from_class(DummyNested)
    assert DummyNested.as_lazy(
        a="hello", b=DummyFlat.as_lazy(b="there")
    ) == Lazy.from_class(
        DummyNested, a="hello", b=Lazy.from_class(DummyFlat, b="there")
    )


@pytest.mark.parametrize(
    ("obj",),
    [
        (DummyFlat(5, "hello"),),
        (DummyFlat.as_lazy().to_eager(a=5, b="hello"),),
        (DummyFlat.as_lazy(b="hello").to_eager(a=5),),
    ],
)
def test_Parsable_init_options(obj):
    assert hasattr(obj, "_cfg")

    assert obj.a == 5
    assert obj.b == "hello" == obj._cfg.b
    assert obj.c == 3.14 == obj._cfg.c
    assert obj._cfg.cls == DummyFlat


def test_Parsable_to_dict():
    assert DummyFlat(a=5, b="hello").to_dict() == {"b": "hello", "c": 3.14}

    assert DummyFlat(a=5, b="hello").to_dict(with_class_tag=True) == {
        "_class": DummyFlat,
        "b": "hello",
        "c": 3.14,
    }

    assert DummyNested(a="hello", b=DummyFlat.as_lazy(b="hello")).to_dict() == {
        "a": "hello",
        "b": {
            "b": "hello",
            "c": 3.14,
        },
        "c": 3.14,
    }

    assert DummyNested(a="hello", b=DummyFlat.as_lazy(b="hello")).to_dict(
        with_class_tag=True
    ) == {
        "_class": DummyNested,
        "a": "hello",
        "b": {
            "_class": DummyFlat,
            "b": "hello",
            "c": 3.14,
        },
        "c": 3.14,
    }

    assert DummyNested(a="hello", b=DummyFlat.as_lazy(b="hello")).to_dict(
        with_class_tag=True, flatten=True
    ) == {
        "_class": DummyNested,
        "a": "hello",
        "b._class": DummyFlat,
        "b.b": "hello",
        "b.c": 3.14,
        "c": 3.14,
    }


def test_Parsable_from_dict_flat():
    x = DummyNested(a="hello", b=DummyFlat.as_lazy(b="hello"))
    y = DummyNested.from_dict(
        {
            "_class": DummyNested,
            "a": "hello",
            "b._class": DummyFlat,
            "b.b": "hello",
            "b.c": 3.14,
            "c": 3.14,
        }
    ).to_eager()
    assert x._cfg == y._cfg
    assert x.a == y.a
    assert x.c == y.c

    assert x.b.a == y.b.a
    assert x.b.b == y.b.b
    assert x.b.c == y.b.c


def test_Parsable_from_dict_nested():
    x = DummyNested(a="hello", b=DummyFlat.as_lazy(b="hello"))
    y = DummyNested.from_dict(
        {
            "_class": DummyNested,
            "a": "hello",
            "b": {
                "_class": DummyFlat,
                "b": "hello",
                "c": 3.14,
            },
            "c": 3.14,
        }
    ).to_eager()
    assert x._cfg == y._cfg
    assert x.a == y.a
    assert x.c == y.c

    assert x.b.a == y.b.a
    assert x.b.b == y.b.b
    assert x.b.c == y.b.c


def test_Lazy_blank_copies_are_identical():
    a = DummyNested.as_lazy()
    b = a.copy()
    assert a == b


def test_Lazy_copies_do_not_share_flat_data():
    a = DummyNested.as_lazy()
    b = a.copy()

    # changing the original does not impact the copy
    a._signature["a"] = (str, "hello")
    assert b == DummyNested.as_lazy()
    assert a != b

    # changing the copy does not change original
    a = DummyNested.as_lazy()
    b._signature["a"] = (str, "hello")
    assert a == DummyNested.as_lazy()
    assert a != b


def test_Lazy_copies_do_not_share_nested_data():
    a = DummyNested.as_lazy()
    b = a.copy()

    # changing the original does not impact the copy
    a._signature["b"][1]._signature["b"] = (str, "hello")
    assert b == DummyNested.as_lazy()
    assert a != b

    # changing the copy does not change original
    a = DummyNested.as_lazy()
    b._signature["b"][1]._signature["b"] = (str, "hello")
    assert a == DummyNested.as_lazy()
    assert a != b


def test_Lazy_copy_changes_field():
    a = DummyNested.as_lazy()
    b = a.copy({"a": "hello", "b.b": "there"})
    assert b.to_dict() == {
        "a": "hello",
        "b": {
            "b": "there",
            "c": 3.14,
        },
        "c": 3.14,
    }
    assert a.to_dict() == {
        "a": Missing,
        "b": {
            "b": Missing,
            "c": 3.14,
        },
        "c": 3.14,
    }


def test_Lazy_copy_raises_for_unknown_field():
    with pytest.raises(AssertionError):
        DummyNested.as_lazy().copy({"x": 1})


def test_Lazy_copy_raises_for_wrong_type():
    with pytest.raises(AssertionError):
        DummyNested.as_lazy().copy({"b": 1})


def test_Lazy__str__():
    x = DummyNested.as_lazy().__str__()
    assert (
        x
        == "DummyNested(\n    a=???,\n    b=DummyFlat(\n        b=???,\n        c=3.14,\n    ),\n    c=3.14,\n)"
    )


def test_Lazy_frozen():
    x = DummyNested.as_lazy()

    with pytest.raises(AssertionError):
        x.b = "hello"


def test_Lazy__getattr__():
    x = DummyNested.as_lazy(c=0.0)
    assert x.c == 0.0
