import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from parsonaut import Lazy, Parsable


class DummySerializableModule(nn.Module, Parsable):
    def __init__(self, value: int, non_parsable: nn.Module = nn.ReLU()) -> None:
        super().__init__()
        self.value = value
        self.layer = nn.Linear(value, value)
        self.activation = non_parsable


def test_from_checkpoint_forwards_arguments_the_config_cannot_record():
    class Net(nn.Module, Parsable):
        def __init__(self, n: int = 2, activation: nn.Module = nn.Identity()):
            super().__init__()
            self.linear = nn.Linear(n, n)
            self.activation = activation

    module = Net(4, activation=nn.Tanh())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt"
        module.to_checkpoint(path)

        # What the config cannot store is rebuilt from the default.
        plain = Net.from_checkpoint(path)
        assert isinstance(plain.activation, nn.Identity)

        # The caller can hand that argument back in.
        restored = Net.from_checkpoint(path, activation=nn.Tanh())
        assert isinstance(restored.activation, nn.Tanh)
        assert restored.linear.out_features == 4


def test_from_to_checkpoint():
    module = DummySerializableModule(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        # The checkpoint directory does not have to exist yet.
        path = Path(tmpdir) / "module"
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


class SubClass2(SubClass1):
    def __init__(self, x: int, z: str = "default"):
        self.x = x
        self.z = z


class Experiment(Parsable):
    def __init__(
        self,
        name: str,
        value: int,
        nested: SubClass1 = SubClass1.as_lazy(),
    ):
        self.name = name
        self.value = value
        self.nested = nested


def test_Parsable_works_with_a_frozen_dataclass():
    # The configuration used to be assigned through the instance, which a
    # frozen dataclass refuses.
    @dataclass(frozen=True)
    class Frozen(Parsable):
        n: int = 1

    built = Frozen(n=2)
    assert built.n == 2
    assert built.as_lazy() == Frozen.as_lazy(n=2)
    assert built.copy({"n": 3}).to_eager().n == 3


def test_Parsable_from_dict_simple():
    a = Experiment.from_dict({"name": "example", "value": 42, "nested.x": 7})
    assert a.name == "example"
    assert a.value == 42
    assert a.nested.x == 7


def test_Parsable_to_eager_is_a_no_op_on_a_built_object():
    obj = SubClass1(x=1)
    assert obj.to_eager() is obj

    with pytest.raises(TypeError, match="already built"):
        obj.to_eager(x=2)


def test_Parsable_configs_and_objects_share_the_same_surface():
    # The point of typing every factory as the class itself: whichever of the
    # two you hold, the same calls work.
    for obj in (SubClass1.as_lazy(x=1), SubClass1(x=1)):
        assert obj.to_eager().x == 1
        assert obj.to_dict() == {"x": 1, "y": "default"}
        assert obj.copy({"x": 2}).x == 2
        assert obj.copy().to_eager().x == 1


def test_Parsable_as_lazy_on_an_instance_returns_its_own_config():
    obj = SubClass1(x=1, y="set")

    assert obj.as_lazy() == SubClass1.as_lazy(x=1, y="set")

    with pytest.raises(TypeError, match="already configured"):
        # A type checker rejects this too - the instance form takes no arguments.
        obj.as_lazy(x=2)  # type: ignore[call-arg]


def test_Parsable_parse_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["prog", "--name", "cli", "--value", "7", "--nested.x", "9"]
    )

    hp = Experiment.parse_args()
    assert hp.name == "cli"
    assert hp.value == 7
    assert hp.nested.x == 9
    assert hp.to_eager().name == "cli"


def test_Parsable_parse_args_can_preset_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--name", "cli"])

    hp = Experiment.parse_args(value=42)
    assert hp.name == "cli"
    assert hp.value == 42


def test_as_lazy_reaches_the_config_behind_either_form():
    config = SubClass1.as_lazy(x=1)
    built = SubClass1(x=1)

    assert config.as_lazy() is config
    assert isinstance(built.as_lazy(), Lazy)
    assert built.as_lazy() == config.as_lazy()
    # Which is how the rest of the Lazy API stays reachable.
    assert config.as_lazy().cls is SubClass1
    assert built.as_lazy().signature == config.as_lazy().signature


def test_copy_always_returns_a_configuration():
    built = SubClass1(x=1)
    built.runtime_state = "set after __init__"  # type: ignore[attr-defined]

    again = built.copy({"x": 2})

    # Called on a live object it still gives back a configuration, so nothing
    # the object picked up along the way comes with it.
    assert isinstance(again, Lazy)
    assert not hasattr(again.to_eager(), "runtime_state")
    assert again.to_eager().x == 2

    # And it is the same type when called on a configuration.
    assert isinstance(SubClass1.as_lazy(x=1).copy({"x": 2}), Lazy)


def test_Parsable_without_a_config_explains_itself():
    orphan = SubClass1.__new__(SubClass1)

    with pytest.raises(AttributeError, match="not created through Parsable"):
        orphan.to_dict()


def test_Parsable_supports_a_custom_new():
    class WithNew(Parsable):
        def __new__(cls):
            return super().__new__(cls)

        def __init__(self, a: int = 1):
            self.a = a

    # pyright checks the constructor against the custom __new__, which is
    # exactly the signature mismatch this test is about.
    assert WithNew(a=5).a == 5  # type: ignore[call-arg]
    assert WithNew(a=5).to_dict() == {"a": 5}  # type: ignore[call-arg]


def test_Parsable_forwards_arguments_to_a_custom_new_that_wants_them():
    seen = []

    class WithNew(Parsable):
        def __new__(cls, a: int = 1):
            seen.append(a)
            return super().__new__(cls)

        def __init__(self, a: int = 1):
            self.a = a

    assert WithNew(a=5).a == 5
    assert seen == [5]


def test_Parsable_does_not_hide_errors_raised_by_a_custom_new():
    class Exploding(Parsable):
        def __new__(cls, a: int = 1):
            raise TypeError("boom")

        def __init__(self, a: int = 1):
            pass

    with pytest.raises(TypeError, match="boom"):
        Exploding()


def test_Parsable_init_runs_once_with_a_config_available():
    seen = []

    class Counter(Parsable):
        def __init__(self, a: int = 1):
            self.a = a
            seen.append((a, self.as_lazy().a))

    Counter(3)
    assert seen == [(3, 3)]


def test_Parsable_skips_init_when_new_returns_something_else():
    # Stock Python skips __init__ when __new__ returns a foreign object.
    class Foreign(Parsable):
        def __new__(cls, *args, **kwargs):
            return "not a Foreign"

        def __init__(self, x: int = 1):
            raise RuntimeError("__init__ should not run")

    assert Foreign() == "not a Foreign"


def test_Parsable_from_dict_rejects_an_unrelated_class():
    with pytest.raises(TypeError, match="not one of SubClass1"):
        SubClass1.from_dict({"_class": "parsonaut.lazy.Lazy"})

    # A subclass is a legitimate specialisation.
    assert SubClass1.from_dict({"_class": SubClass2, "x": 2}).as_lazy().cls is SubClass2
