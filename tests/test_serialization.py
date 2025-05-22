import tempfile
from pathlib import Path

import pytest

from parsonaut.serialization import Serializable


# Testable subclass
class DummySerializable(Serializable):
    def __init__(self, value):
        self.value = value

    def to_dict(self, with_class_tag_as_str):
        d = {"value": self.value}
        if with_class_tag_as_str:
            d["_class"] = "test_serialization.DummySerializable"
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["value"])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def test_serializable_to_from_json():
    obj = DummySerializable(123)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "object.json"
        obj.to_file(path)
        loaded = DummySerializable.from_file(path)
        assert isinstance(loaded, DummySerializable)
        assert loaded.value == 123

        loaded = Serializable.from_file(path)
        assert isinstance(loaded, DummySerializable)
        assert loaded.value == 123


def test_serializable_to_from_yaml():
    obj = DummySerializable(456)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "object.yaml"
        obj.to_file(path)
        loaded = DummySerializable.from_file(path)
        assert isinstance(loaded, DummySerializable)
        assert loaded.value == 456


def test_serializable_invalid_extension():
    obj = DummySerializable(999)
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "object.txt"
        with pytest.raises(ValueError):
            obj.to_file(bad_path)
        with pytest.raises(ValueError):
            DummySerializable.from_file(bad_path)
