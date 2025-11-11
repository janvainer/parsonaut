import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from parsonaut import Choices, Parsable
from parsonaut.typecheck import Missing


class DummySerializableModule(nn.Module, Parsable):
    def __init__(self, value: int, non_parsable: nn.Module = nn.ReLU()) -> None:
        super().__init__()
        self.value = value
        self.layer = nn.Linear(value, value)
        self.activation = non_parsable


def test_from_to_checkpoint():
    module = DummySerializableModule(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "module"
        path.mkdir(parents=True)
        module.to_checkpoint(path)
        module2 = DummySerializableModule.from_checkpoint(path)
        assert module2.activation.__class__ == nn.ReLU
        assert module.value == module2.value
        sd2 = module2.state_dict()
        for k, v in module.state_dict().items():
            torch.testing.assert_close(v, sd2[k])


class SubClass1(Parsable):
    def __init__(self, x: int, y: str = "default"):
        self.x = x
        self.y = y


class SubClass2(Parsable):
    def __init__(self, x: int, z: str = "default"):
        self.x = x
        self.z = z


class SubClasses(Choices):
    sub1 = SubClass1.as_lazy(x=5)
    subclass2 = SubClass2.as_lazy(x=10, z="custom")


class TestMe(Parsable):
    def __init__(self, name: str, value: int, nested: SubClasses = SubClasses.sub1):
        self.name = name
        self.value = value


def test_Parsable_from_dict_simple():
    a = TestMe.from_dict({"name": "example", "value": 42, "nested.x": 7})
    assert a.name == "example"
    assert a.value == 42
    assert a.nested.x == 7


def test_Parsable_from_dict_subclass_change():
    c = TestMe.from_dict(
        {
            "name": "example",
            "value": 42,
            "nested._class": SubClass2,
            "nested.z": "changed",
        }
    )
    assert c.name == "example"
    assert c.value == 42
    assert c.nested.cls == SubClass2
    assert (
        c.nested.x == Missing
    )  # x takes default value from SubClass2, not from Choices!
    assert c.nested.z == "changed"
