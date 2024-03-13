import pytest

from parsonaut.parsable import (
    Missing,
    MissingType,
    create_dict_from_class_init_args,
    get_class_import_path,
    get_class_init_signature,
    import_class_from_path,
)
from parsonaut.testing import DummyFlat, DummyNested


def test_get_class_init_signature_flat():
    defaults, annotations = get_class_init_signature(DummyFlat)
    assert defaults == {"a": Missing, "b": Missing, "c": 3.14}
    assert annotations == {"a": MissingType, "b": str | int, "c": float}

    defaults, annotations = get_class_init_signature(DummyFlat, 1, c=2.71)
    assert defaults == {"a": 1, "b": Missing, "c": 2.71}
    assert annotations == {"a": MissingType, "b": str | int, "c": float}


def test_get_class_init_signature_nested():
    defaults, annotations = get_class_init_signature(DummyNested)
    assert defaults == {"a": Missing, "b": Missing, "c": 3.14}
    assert annotations == {"a": DummyFlat, "b": str, "c": float}

    a = DummyFlat(1, "2")
    defaults, annotations = get_class_init_signature(DummyNested, a, c=2.71)
    assert defaults == {"a": a, "b": Missing, "c": 2.71}
    assert annotations == {"a": DummyFlat, "b": str, "c": float}


def test_get_class_init_signature_raises_for_mismatched_types():

    get_class_init_signature(DummyNested, b="hello")
    get_class_init_signature(DummyNested, c=0.1)

    with pytest.raises(TypeError):
        get_class_init_signature(DummyNested, 1)

    with pytest.raises(TypeError):
        get_class_init_signature(DummyNested, c=1)


def test_get_class_init_signature_raises_if_init_does_not_have_self():
    class A:
        def __init__(a, b: str | int, c: float = 3.14):
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


def test_create_dict_from_class_init_args_flat():
    assert create_dict_from_class_init_args(DummyFlat, 1, "2", c=2.71) == {
        "_type": "parsonaut.testing.DummyFlat",
        "a": 1,
        "b": "2",
        "c": 2.71,
    }


def test_create_dict_from_class_init_args_nested():
    assert create_dict_from_class_init_args(
        DummyNested, DummyFlat(1, 2), "2", c=2.71
    ) == {
        "_type": "parsonaut.testing.DummyNested",
        "a": {
            "_type": "parsonaut.testing.DummyFlat",
            "a": 1,
            "b": 2,
            "c": 3.14,
        },
        "b": "2",
        "c": 2.71,
    }


def test_create_dict_from_class_init_args_raises_for_wrong_types():
    with pytest.raises(TypeError):
        create_dict_from_class_init_args(DummyFlat, b=0.1)

    with pytest.raises(TypeError):
        create_dict_from_class_init_args(DummyNested, 1, "2", c=2.71)

    class HasNoToDict:
        def __init__(self, a: int):
            pass

    class NestedWithWrongType:
        def __init__(self, a: HasNoToDict):
            pass

    with pytest.raises(TypeError):
        create_dict_from_class_init_args(NestedWithWrongType, HasNoToDict(1))

    class SomeInvalidType:
        pass

    class ToDictReturnsInvalidType:
        def __init__(self):
            pass

        def to_dict(self):
            return {"a": SomeInvalidType()}

    class NestedWithWrongType:
        def __init__(self, a: ToDictReturnsInvalidType):
            pass

    with pytest.raises(AssertionError):
        create_dict_from_class_init_args(
            NestedWithWrongType, ToDictReturnsInvalidType()
        )

    class ToDictReturnsNonStringKeys:
        def __init__(self):
            pass

        def to_dict(self):
            return {1: "hello"}

    class NestedWithWrongType:
        def __init__(self, a: ToDictReturnsNonStringKeys):
            pass

    with pytest.raises(AssertionError):
        create_dict_from_class_init_args(
            NestedWithWrongType, ToDictReturnsNonStringKeys()
        )
