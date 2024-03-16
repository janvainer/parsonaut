from .parsable import Lazy


class DummyFlat:
    def __init__(self, a, b: str, c: float = 3.14):
        self.a = a
        self.b = b
        self.c = c

    def to_dict(self):
        return {
            "_type": "parsonaut.testing.DummyFlat",
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }


class DummyNested:
    def __init__(self, a: Lazy[DummyFlat, ...], b: str, c: float = 3.14):
        self.a = a
        self.b = b
        self.c = c

    def to_dict(self):
        return {
            "_type": "parsonaut.testing.DummyFlat",
            "a": self.a.to_dict(),
            "b": self.b,
            "c": self.c,
        }
