from parsonaut.signature import Missing, MissingType, get_signature


class DummyFlat:
    def __init__(self, a, b: str, c: float = 3.14):
        pass


class DummyNested:
    def __init__(self, a: DummyFlat, b: str, c: float = 3.14):
        pass


def test_get_class_init_signature_flat():

    signature = get_signature(DummyFlat)
    assert signature == {
        "a": (MissingType, Missing),
        "b": (str, Missing),
        "c": (float, 3.14),
    }

    signature = get_signature(DummyFlat, 1, c=2.71)
    assert signature == {
        "a": (MissingType, 1),
        "b": (str, Missing),
        "c": (float, 2.71),
    }


def test_get_class_init_signature_nested():

    signature = get_signature(DummyNested)
    assert signature == {
        "a": (DummyFlat, Missing),
        "b": (str, Missing),
        "c": (float, 3.14),
    }

    a = DummyFlat(1, "2")
    signature = get_signature(DummyNested, a, c=2.71)
    assert signature == {
        "a": (DummyFlat, a),
        "b": (str, Missing),
        "c": (float, 2.71),
    }
