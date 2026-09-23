import copy
import pickle
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from parsonaut import Lazy, Parsable
from parsonaut.lazy import Missing, MissingType, _bind, _type_name, typecheck_eager


def bound(func, *args, **kwargs):
    """Every parameter of `func` as (annotation, value), arguments applied."""
    return _bind(func, args, kwargs)[0]


def dummy_factory(width: int = 8, name: str = "m"):
    return f"{name}/{width}"


class DummyFlat(Parsable):
    def __init__(self, a, b: str, c: float = 3.14):
        self.a = a
        self.b = b
        self.c = c


class DummyNested(Parsable):
    def __init__(self, a: str, b: DummyFlat, c: float = 3.14):
        self.a = a
        self.b = b.to_eager(a=a)
        self.c = c


def test_binding_flat():

    signature = bound(DummyFlat.__init__)
    assert signature == {
        "a": (MissingType, Missing),
        "b": (str, Missing),
        "c": (float, 3.14),
    }

    signature = bound(DummyFlat.__init__, 1, c=2.71)
    assert signature == {
        "a": (MissingType, 1),
        "b": (str, Missing),
        "c": (float, 2.71),
    }

    def dummy_func(a, b: str, c: float = 3.14):
        pass

    signature = bound(dummy_func, 1, c=2.71)
    assert signature == {
        "a": (MissingType, 1),
        "b": (str, Missing),
        "c": (float, 2.71),
    }


def test_binding_nested():

    signature = bound(DummyNested.__init__)
    assert signature == {
        "a": (str, Missing),
        "b": (DummyFlat, Missing),
        "c": (float, 3.14),
    }

    b = DummyFlat(1, "2")
    signature = bound(DummyNested.__init__, b=b, c=2.71)
    assert signature == {
        "a": (str, Missing),
        "b": (DummyFlat, b),
        "c": (float, 2.71),
    }


def test_binding_resolves_postponed_annotations():
    # `from __future__ import annotations` turns every annotation into a string.
    module = ModuleType("postponed")
    exec(
        "from __future__ import annotations\n"
        "def dummy_func(a: int = 1, b: tuple[str, ...] = ('x',)): pass\n",
        module.__dict__,
    )

    assert bound(module.dummy_func) == {
        "a": (int, 1),
        "b": (tuple[str, ...], ("x",)),
    }


def test_binding_skips_var_args():
    def dummy_func(a: int = 1, *args: int, **kwargs: str):
        pass

    assert bound(dummy_func) == {"a": (int, 1)}


def _is_eager():
    return isinstance(Lazy.from_class(DummyFlat)._signature, dict)


def test_signatures_are_resolved_lazily_by_default():
    assert not _is_eager()

    with typecheck_eager():
        assert _is_eager()
        # Nested blocks restore the outer setting instead of clearing it.
        with typecheck_eager(False):
            assert not _is_eager()
        assert _is_eager()

    assert not _is_eager()


def test_typecheck_eager_does_nothing_until_entered():
    typecheck_eager()
    assert not _is_eager()


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
        "b": (DummyFlat, Lazy.from_class(DummyFlat)),
        "c": (float, 3.14),
    }

    assert Lazy.get_signature(DummyNested, a="hello") != {
        "a": (str, Missing),
        "b": (DummyFlat, Lazy.from_class(DummyFlat)),
        "c": (float, 3.14),
    }


def test_Lazy_get_signature_raises_for_invalid_type():

    class GoodDummy:
        def __init__(self, a: Lazy[DummyFlat]) -> None:
            pass

    Lazy.get_signature(GoodDummy)

    class BadDummy:
        def __init__(self, a: Lazy[GoodDummy]) -> None:
            pass

    with pytest.raises(TypeError, match="does not subclass Parsable"):
        Lazy.get_signature(BadDummy)


def test_Lazy_get_signature_raises_for_optional_nested():
    class Dummy:
        def __init__(self, a: DummyFlat | None = None) -> None:
            pass

    with pytest.raises(TypeError, match="optional nested configs"):
        Lazy.get_signature(Dummy)


@pytest.mark.parametrize("name", ["copy", "cls", "signature", "to_eager"])
def test_Lazy_get_signature_raises_for_reserved_parameter_name(name):
    namespace: dict = {}
    exec(f"def __init__(self, {name}: int = 1) -> None: pass", namespace)
    Dummy = type("Dummy", (), {"__init__": namespace["__init__"]})

    with pytest.raises(TypeError, match="clashes with the Lazy API"):
        Lazy.get_signature(Dummy)


@pytest.mark.parametrize("name", ["_class", "_hidden"])
def test_Lazy_get_signature_raises_for_an_underscore_parameter_name(name):
    namespace: dict = {}
    exec(f"def __init__(self, {name}: int = 1) -> None: pass", namespace)
    Dummy = type("Dummy", (), {"__init__": namespace["__init__"]})

    with pytest.raises(TypeError, match="cannot start with an underscore"):
        Lazy.get_signature(Dummy)


def test_an_unconfigurable_underscore_parameter_is_still_allowed():
    class Dummy:
        def __init__(self, _handle=None, a: int = 1) -> None:
            pass

    assert Lazy.get_signature(Dummy) == {"a": (int, 1)}


def test_Lazy_subscript_is_a_nested_annotation():
    class Dummy:
        def __init__(self, a: Lazy[DummyFlat]) -> None:
            pass

    assert Lazy.get_signature(Dummy) == {
        "a": (Lazy[DummyFlat], Lazy.from_class(DummyFlat))
    }


def test_Lazy_get_signature_needs_an_annotation_for_a_lazy_default():
    class Dummy:
        def __init__(self, a=Lazy.from_class(DummyFlat)) -> None:
            pass

    with pytest.raises(TypeError, match="defaults to a config but has no annotation"):
        Lazy.get_signature(Dummy)


def test_Lazy_get_signature_accepts_parsable_annotation_for_lazy_default():

    class Dummy:
        def __init__(self, a: DummyFlat = DummyFlat.as_lazy()) -> None:
            pass

    assert Lazy.get_signature(Dummy) == {"a": (DummyFlat, Lazy.from_class(DummyFlat))}


def test_Lazy_get_signature_fails_if_lazy_default_has_wrong_annotation():

    class NotParsable:
        pass

    class Dummy:
        def __init__(self, a: NotParsable = Lazy.from_class(DummyFlat)) -> None:  # type: ignore[assignment]
            pass

    with pytest.raises(TypeError, match="is annotated with NotParsable"):
        Lazy.get_signature(Dummy)


def test_Lazy_get_signature_rejects_a_default_of_the_wrong_class():
    class Other(Parsable):
        def __init__(self, z: int = 1) -> None:
            pass

    class Sub(DummyFlat):
        pass

    class Ok:
        def __init__(self, a: DummyFlat = Sub.as_lazy()) -> None:
            pass

    assert Lazy.get_signature(Ok)["a"][1].cls is Sub

    class Bad:
        def __init__(self, a: DummyFlat = Other.as_lazy()) -> None:  # type: ignore[assignment]
            pass

    with pytest.raises(TypeError, match="annotated as DummyFlat"):
        Lazy.get_signature(Bad)


def test_Lazy_copy_rejects_a_nested_config_of_the_wrong_class():
    class Other(Parsable):
        def __init__(self, z: int = 1) -> None:
            pass

    class Sub(DummyFlat):
        pass

    accepted = Lazy.from_class(DummyNested).copy({"b": Sub.as_lazy()})
    assert accepted.b.cls is Sub

    with pytest.raises(TypeError, match="annotated as DummyFlat"):
        Lazy.from_class(DummyNested).copy({"b": Other.as_lazy()})


def test_Lazy_get_signature_fails_for_non_lazy_default_of_nested_param():
    class Dummy:
        def __init__(self, a: DummyFlat = "nonsense") -> None:  # type: ignore[assignment]
            pass

    with pytest.raises(TypeError, match="expected a to default to"):
        Lazy.get_signature(Dummy)


def test_Lazy_names_the_parameters_it_has_when_asked_for_one_it_does_not():
    with pytest.raises(AttributeError, match=r"no parameter 'bb'.*takes a, b, c"):
        Lazy.from_class(DummyNested).bb


def test_Lazy_from_dict_leaves_a_list_to_the_annotation():
    dct = {"_class": f"{__name__}.DummyFlat", "b": "x", "c": [1.0, 2.0]}

    with pytest.raises(TypeError, match="does not match the provided annotation"):
        Lazy.from_dict(dct).signature


def test_Lazy_configures_a_factory_function():
    cfg = Lazy.from_class(dummy_factory, width=16)

    assert cfg.to_eager() == "m/16"
    assert Lazy.from_dict(cfg.to_dict(class_tag="str")).to_eager() == "m/16"


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
    assert Lazy.from_class(DummyNested).to_dict(class_tag=True) == {
        "_class": DummyNested,
        "a": Missing,
        "b": {"_class": DummyFlat, "c": 3.14, "b": Missing},
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(class_tag="str") == {
        "_class": f"{__name__}.DummyNested",
        "a": Missing,
        "b": {"_class": f"{__name__}.DummyFlat", "c": 3.14, "b": Missing},
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(flatten=True) == {
        "a": Missing,
        "b.b": Missing,
        "b.c": 3.14,
        "c": 3.14,
    }

    assert Lazy.from_class(DummyNested).to_dict(class_tag=True, flatten=True) == {
        "_class": DummyNested,
        "a": Missing,
        "b._class": DummyFlat,
        "b.b": Missing,
        "b.c": 3.14,
        "c": 3.14,
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

    with pytest.raises(TypeError, match="does not match"):
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
        c: Lazy[InnerDummy] = Lazy.from_class(InnerDummy)
        d: tuple[int, ...] = (1, 2, 3)

    lazy_dummy = Lazy.from_class(OuterDummy)
    assert lazy_dummy.signature == {
        "c": (Lazy[InnerDummy], Lazy.from_class(InnerDummy)),
        "d": (tuple[int, ...], (1, 2, 3)),
    }


def test_Lazy_to_eager():
    lazy_dummy = Lazy.from_class(DummyFlat)

    with pytest.raises(TypeError):
        lazy_dummy.to_eager()

    with pytest.raises(TypeError, match="named parameters only"):
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

    assert DummyFlat(a=5, b="hello").to_dict(class_tag=True) == {
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
        class_tag=True
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
        class_tag=True, flatten=True
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
    a = Lazy.from_class(DummyNested)
    b = a.copy()
    assert a == b


def test_Lazy_copies_do_not_share_flat_data():
    a = Lazy.from_class(DummyNested)
    b = a.copy()

    a._signature["a"] = (str, "hello")
    assert b == DummyNested.as_lazy()
    assert a != b

    a = Lazy.from_class(DummyNested)
    b._signature["a"] = (str, "hello")
    assert a == DummyNested.as_lazy()
    assert a != b


def test_Lazy_copies_do_not_share_nested_data():
    a = Lazy.from_class(DummyNested)
    b = a.copy()

    a._signature["b"][1]._signature["b"] = (str, "hello")
    assert b == DummyNested.as_lazy()
    assert a != b

    a = Lazy.from_class(DummyNested)
    b._signature["b"][1]._signature["b"] = (str, "hello")
    assert a == DummyNested.as_lazy()
    assert a != b


def test_Lazy_copy_changes_field():
    a = Lazy.from_class(DummyNested)
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
    with pytest.raises(ValueError, match="is not present"):
        Lazy.from_class(DummyNested).copy({"x": 1})


def test_Lazy_copy_raises_for_wrong_type():
    with pytest.raises(TypeError, match="not a configuration"):
        Lazy.from_class(DummyNested).copy({"b": 1})


def test_Lazy__str__():
    x = DummyNested.as_lazy().__str__()
    assert (
        x
        == "DummyNested(\n    a=???,\n    b=DummyFlat(\n        b=???,\n        c=3.14,\n    ),\n    c=3.14,\n)"
    )


def test_Lazy__str__of_a_config_without_parsable_parameters():
    class NoParams(Parsable):
        def __init__(self):
            pass

    class HoldsNoParams(Parsable):
        def __init__(self, inner: NoParams = NoParams.as_lazy(), a: int = 1):
            pass

    assert str(NoParams.as_lazy()) == "NoParams()"
    assert str(HoldsNoParams.as_lazy()) == (
        "HoldsNoParams(\n    a=1,\n    inner=NoParams(),\n)"
    )


def test_Lazy_error_messages_spell_out_unions_and_generics():
    assert _type_name(DummyFlat | None) == str(DummyFlat | None)
    assert "DummyFlat" in _type_name(DummyFlat | None)
    assert _type_name(tuple[int, ...]) == "tuple[int, ...]"
    assert _type_name(DummyFlat) == "DummyFlat"

    def some_func():
        pass

    assert _type_name(some_func) == "some_func"

    class Dummy:
        def __init__(self, a: DummyFlat | None = None) -> None:
            pass

    with pytest.raises(TypeError, match=r"a: .*DummyFlat \| None"):
        Lazy.get_signature(Dummy)


def test_Lazy_frozen():
    x = Lazy.from_class(DummyNested)

    with pytest.raises(AttributeError, match="frozen"):
        x.b = "hello"


def test_Lazy__getattr__():
    x = DummyNested.as_lazy(c=0.0)
    assert x.c == 0.0


def test_Lazy__repr__():
    assert repr(DummyNested.as_lazy()) == str(DummyNested.as_lazy())


def test_Lazy__eq__with_other_types():
    assert DummyNested.as_lazy() != "not a config"
    assert DummyNested.as_lazy().__eq__("not a config") is NotImplemented


def test_Lazy_copy_handles_fields_ending_in_the_class_tag():
    class Weird(Parsable):
        def __init__(self, my_class: str = "x"):
            pass

    assert Lazy.from_class(Weird).copy({"my_class": "y"}).my_class == "y"


def test_Lazy_to_dict_converts_nested_tuples_to_lists():
    class Inner(Parsable):
        def __init__(self, t: tuple[int, int] = (1, 2)):
            pass

    class Outer(Parsable):
        def __init__(self, inner: Inner = Inner.as_lazy(), t: tuple[int, int] = (3, 4)):
            pass

    assert Outer.as_lazy().to_dict(tuples_as_lists=True) == {
        "inner": {"t": [1, 2]},
        "t": [3, 4],
    }


def test_Lazy_to_dict_converts_optional_tuples_to_lists():
    class Opt(Parsable):
        def __init__(
            self, xs: tuple[int, ...] | None = (1, 2), ys: tuple[int, ...] | None = None
        ):
            pass

    assert Opt.as_lazy().to_dict(tuples_as_lists=True) == {"xs": [1, 2], "ys": None}


def test_Lazy_to_dict_can_skip_missing_values():
    assert DummyNested.as_lazy().to_dict(skip_missing=True) == {
        "b": {"c": 3.14},
        "c": 3.14,
    }

    dct = DummyNested.as_lazy().to_dict(class_tag=True, skip_missing=True)
    assert Lazy.from_dict(dct) == DummyNested.as_lazy()


def test_Lazy_to_dict_can_tag_the_class_as_an_import_path():
    dct = DummyNested.as_lazy().to_dict(class_tag="str", skip_missing=True)
    assert dct["_class"] == f"{__name__}.DummyNested"
    assert Lazy.from_dict(dct) == DummyNested.as_lazy()


def test_Lazy_from_dict_without_a_class_tag():
    with pytest.raises(ValueError, match="without a '_class' key"):
        Lazy.from_dict({"a": 1})


def test_Lazy_from_dict_fills_in_nested_classes_from_the_defaults():
    config = Lazy.from_dict({"_class": DummyNested, "b": {"b": "set"}})
    assert config == DummyNested.as_lazy(b=DummyFlat.as_lazy(b="set"))

    config = Lazy.from_dict({"_class": DummyNested, "b.b": "set"})
    assert config == DummyNested.as_lazy(b=DummyFlat.as_lazy(b="set"))


def test_Lazy_from_dict_accepts_nested_and_dotted_keys_together():
    config = Lazy.from_dict(
        {"_class": DummyNested, "b.b": "set", "b": {"_class": DummyFlat}}
    )
    assert config == DummyNested.as_lazy(b=DummyFlat.as_lazy(b="set"))


def test_Lazy_from_dict_keeps_a_nested_class_within_its_annotation():
    class Other(Parsable):
        def __init__(self, z: int = 1):
            pass

    with pytest.raises(TypeError, match="Cannot switch b to Other"):
        Lazy.from_dict({"_class": DummyNested, "b": {"_class": Other}})


def test_Lazy_from_dict_reports_an_unknown_key():
    with pytest.raises(ValueError, match="field='nope' that is not present"):
        Lazy.from_dict({"_class": DummyFlat, "nope": 1})


def test_Lazy_from_class_accepts_a_parameter_named_cl():
    class HasCl(Parsable):
        def __init__(self, cl: int = 1):
            pass

    assert HasCl.as_lazy(cl=2).as_lazy().cl == 2
    assert Lazy.from_class(HasCl, cl=3).cl == 3
    assert Lazy.get_signature(HasCl, cl=4) == {"cl": (int, 4)}


def test_Missing_is_a_singleton():
    assert MissingType() is Missing
    assert copy.deepcopy(Missing) is Missing
    assert pickle.loads(pickle.dumps(Missing)) is Missing


def test_Lazy_copy_rejects_a_dict_for_a_flat_field():
    with pytest.raises(TypeError, match="not a nested configuration"):
        Lazy.from_class(DummyNested).copy({"c": {"nope": 1}})


def test_Lazy_copy_accepts_a_config_for_a_nested_field():
    replaced = Lazy.from_class(DummyNested).copy({"b": DummyFlat.as_lazy(b="set")})
    assert replaced.b.b == "set"


def test_Lazy_accepts_a_built_object_where_a_config_is_expected():
    built = DummyFlat(1, "hello")
    config = Lazy.from_class(DummyNested, b=built)

    assert config.b == DummyFlat.as_lazy(b="hello")
    assert config.to_dict()["b"] == {"b": "hello", "c": 3.14}


def test_Lazy_rejects_values_it_cannot_record():
    class Dummy(Parsable):
        def __init__(self, unannotated, lr: float = 0.1):
            pass

    with pytest.raises(TypeError, match="no type annotation"):
        Lazy.from_class(Dummy, unannotated=[1, 2]).signature

    class Unsupported(Parsable):
        def __init__(self, items: list[int] = [], lr: float = 0.1):
            pass

    with pytest.raises(TypeError, match="not a configurable type"):
        Lazy.from_class(Unsupported, items=[1]).signature

    assert Dummy([1, 2], lr=0.5).to_dict() == {"lr": 0.5}


def test_Lazy_widens_int_defaults_for_float_parameters():
    class Dummy(Parsable):
        def __init__(self, lr: float = 1, betas: tuple[float, float] = (0, 1)):
            pass

    assert Lazy.from_class(Dummy).signature == {
        "lr": (float, 1.0),
        "betas": (tuple[float, float], (0.0, 1.0)),
    }
    assert Lazy.from_class(Dummy, lr=2).signature["lr"] == (float, 2.0)
    with pytest.raises(TypeError, match="does not match"):
        Lazy.from_class(Dummy, lr=True).signature


def test_a_list_is_recorded_as_it_was_when_it_was_passed():
    class WithTuple(Parsable):
        def __init__(self, xs: tuple[int, ...] = ()):
            self.xs = tuple(xs)

    data = [1, 2]
    built = WithTuple(data)  # type: ignore[arg-type]
    data.append(3)
    assert built.xs == (1, 2)
    assert built.to_dict()["xs"] == (1, 2)

    data = [4, 5]
    config = WithTuple.as_lazy(xs=data)
    data.append(6)
    assert config.xs == (4, 5)


def test_Lazy_accepts_lists_where_a_tuple_is_annotated():
    class Dummy(Parsable):
        def __init__(self, t: tuple[int, int] = (1, 2)):
            pass

    assert Lazy.from_class(Dummy, t=[3, 4]).t == (3, 4)
    assert Lazy.from_class(Dummy).copy({"t": [5, 6]}).t == (5, 6)


def test_Lazy_type_annotations_are_validated_against_values():
    class Dummy(Parsable):
        def __init__(self, a: str = 1):  # type: ignore[assignment]
            pass

    with pytest.raises(TypeError, match="does not match"):
        Lazy.from_class(Dummy).signature


def test_Lazy_rejects_none_default_for_a_non_optional_annotation():
    class Dummy(Parsable):
        def __init__(self, a: str = None):  # type: ignore[assignment]
            pass

    with pytest.raises(TypeError, match="does not match"):
        Lazy.from_class(Dummy).signature


def test_Lazy_accepts_none_default_for_an_optional_annotation():
    class Dummy(Parsable):
        def __init__(self, a: str | None = None):
            pass

    assert Lazy.from_class(Dummy).signature == {"a": (str | None, None)}


def test_validation_survives_python_O():
    script = (
        "from parsonaut import Parsable\n"
        "class A(Parsable):\n"
        "    def __init__(self, x: int = 1): pass\n"
        "for fn in (lambda: A.as_lazy().copy({'nope': 1}),\n"
        "           lambda: A.as_lazy(x='nope').to_dict()):\n"
        "    try:\n"
        "        fn()\n"
        "        print('NOT RAISED')\n"
        "    except (TypeError, ValueError) as exc:\n"
        "        print(type(exc).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["ValueError", "TypeError"]


def test_repr_of_an_invalid_config_reports_instead_of_raising():
    class Dummy(Parsable):
        def __init__(self, items: list[int] = [1]):
            pass

    config = Lazy.from_class(Dummy, items=[2])

    assert "invalid Dummy configuration" in repr(config)
    with pytest.raises(TypeError, match="not a configurable type"):
        str(config)


def test_introspection_does_not_validate_the_signature():
    class Dummy(Parsable):
        def __init__(self, items: list[int] = [1]):
            pass

    config = Lazy.from_class(Dummy, items=[2])

    assert not hasattr(config, "_ipython_canary_method_should_not_exist_")
    assert isinstance(copy.copy(config), Lazy)


def test_copy_with_an_empty_group_is_not_a_no_op():
    with pytest.raises(TypeError, match="not a nested configuration"):
        Lazy.from_class(DummyNested).copy({"c": {}})


def test_Lazy_copy_limits_a_class_switch_to_what_is_allowed():
    class Other(Parsable):
        def __init__(self, z: int = 1):
            pass

    with pytest.raises(TypeError, match="not one of DummyNested"):
        Lazy.from_class(DummyNested).copy({"_class": Other}, allowed=(DummyNested,))

    class Sub(DummyNested):
        pass

    switched = Lazy.from_class(DummyNested).copy(
        {"_class": Sub}, allowed=(DummyNested,)
    )
    assert switched.cls is Sub
    assert switched == Lazy.from_class(Sub)

    with pytest.raises(TypeError, match="not one of Other"):
        Lazy.from_class(DummyNested).copy({"_class": Sub}, allowed=(Other,))

    class Holder(Parsable):
        def __init__(self, inner: DummyNested = DummyNested.as_lazy()):
            pass

    # `allowed` applies to a nested `_class` too, not only the node copy()
    # was called on, and to a built object as well as a configuration.
    with pytest.raises(TypeError, match="not one of Other"):
        Holder.as_lazy().copy({"inner._class": Sub}, allowed=(Other,))
    with pytest.raises(TypeError, match="not one of Other"):
        Holder().copy({"inner._class": Sub}, allowed=(Other,))


def test_a_nested_subclass_round_trips_and_keeps_shared_values():
    class Base(Parsable):
        def __init__(self, x: int = 1):
            pass

    class Sub(Base):
        def __init__(self, x: int = 1, y: int = 2):
            pass

    class Outer(Parsable):
        def __init__(self, inner: Base = Base.as_lazy()):
            pass

    config = Outer.as_lazy(inner=Sub.as_lazy(x=3, y=4))
    # The classes are local, so the tag has to be the class itself; an import
    # path would have nothing to import. Files use the string form.
    assert Outer.from_dict(config.to_dict(class_tag=True)) == config

    switched = Outer.as_lazy(inner=Base.as_lazy(x=8)).copy(
        {"inner._class": Sub, "inner.y": 9}
    )
    assert switched.inner.cls is Sub
    assert switched.inner.x == 8
    assert switched.inner.y == 9


def test_a_class_switch_uses_the_new_defaults():
    class Adam(Parsable):
        def __init__(self, lr: float = 1e-3):
            pass

    class AdamW(Adam):
        def __init__(self, lr: float = 1e-4, wd: float = 0.01):
            pass

    class Params(Parsable):
        def __init__(self, opt: Adam = Adam.as_lazy()):
            pass

    # Untouched values take the subclass defaults, not the base defaults.
    switched = Params.as_lazy().copy({"opt._class": AdamW})
    assert switched.opt.cls is AdamW
    assert switched.opt.lr == 1e-4
    assert switched.opt.wd == 0.01

    # A value that was actually set on the base is kept.
    overridden = Params.as_lazy(opt=Adam.as_lazy(lr=0.5)).copy({"opt._class": AdamW})
    assert overridden.opt.lr == 0.5
    assert overridden.opt.wd == 0.01


def test_varargs_are_not_recorded_and_do_not_crash():
    class WithKwargs(Parsable):
        def __init__(self, n: int = 1, **kwargs):
            self.kwargs = kwargs

    obj = WithKwargs(n=2, extra=3)
    assert obj.kwargs == {"extra": 3}
    assert obj.to_dict() == {"n": 2}

    class WithArgs(Parsable):
        def __init__(self, n: int = 1, *args):
            self.args = args

    built = WithArgs(4, 5, 6)
    assert built.args == (5, 6)
    assert built.to_dict() == {"n": 4}


def test_varargs_cannot_be_stored_in_a_strict_configuration():
    class WithKwargs(Parsable):
        def __init__(self, n: int = 1, **kwargs):
            pass

    with pytest.raises(TypeError, match="'extra'"):
        Lazy.from_class(WithKwargs, extra=3).signature

    class WithArgs(Parsable):
        def __init__(self, n: int = 1, *args):
            pass

    with pytest.raises(TypeError, match="1 positional argument"):
        Lazy.from_class(WithArgs, 1, 2).signature


def test_a_self_reference_is_reported_instead_of_recursing():
    module = ModuleType("cycle")
    exec(
        "from __future__ import annotations\n"
        "from parsonaut import Parsable\n"
        "class Node(Parsable):\n"
        "    def __init__(self, child: Node, n: int = 1):\n"
        "        pass\n"
        "class A(Parsable):\n"
        "    def __init__(self, other: B):\n"
        "        pass\n"
        "class B(Parsable):\n"
        "    def __init__(self, other: A):\n"
        "        pass\n",
        module.__dict__,
    )

    with pytest.raises(TypeError, match="Node -> Node refers to itself"):
        module.Node.as_lazy().signature
    with pytest.raises(TypeError, match="A -> B -> A refers to itself"):
        module.A.as_lazy().signature


def test_to_eager_names_the_values_it_is_missing():
    with pytest.raises(TypeError, match=r"no value for 'a'.*--a"):
        Lazy.from_class(DummyNested).to_eager()

    class TwoMissing(Parsable):
        def __init__(self, first: str, second: int, third: int = 3):
            pass

    with pytest.raises(TypeError, match=r"'first', 'second'.*--first --second"):
        Lazy.from_class(TwoMissing).to_eager()


def test_to_eager_does_not_blame_a_constructor_error_on_a_missing_value():
    class Picky(Parsable):
        def __init__(self, needed: int, given: int = 1):
            raise TypeError("something else entirely")

    with pytest.raises(TypeError, match="something else entirely"):
        Lazy.from_class(Picky).to_eager(needed=1)
