import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from parsonaut import Parsable
from parsonaut.serialization import (
    Serializable,
    format_of,
    maybe_import,
)


# Testable subclass
class DummySerializable(Serializable):
    def __init__(self, value):
        self.value = value

    def to_dict(
        self,
        *,
        class_tag=False,
        tuples_as_lists=False,
        skip_missing=False,
    ):
        d = {"value": self.value}
        if class_tag:
            d["_class"] = "test_serialization.DummySerializable"
        return d

    @classmethod
    def from_dict(cls, dct):
        return cls(dct["value"])

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


class ParsableSerializable(Parsable):
    def __init__(self, value: int = 1, value2: tuple[int, ...] = (1, 2, 3)):
        pass


class WithRequired(Parsable):
    def __init__(self, name: str, t: tuple[int, int] = (1, 2)):
        pass


class NestedSerializable(Parsable):
    def __init__(self, inner: WithRequired = WithRequired.as_lazy()):
        pass


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_parsable_serializable_to_from_file(extension):
    from parsonaut.lazy import typecheck_eager

    with typecheck_eager():
        obj = ParsableSerializable(value=42, value2=(4, 5, 6))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"parsable.{extension}"
            obj.to_file(path)
            loaded = ParsableSerializable.from_file(path).to_eager()
            assert isinstance(loaded, ParsableSerializable)
            assert loaded._cfg.value == 42
            assert loaded._cfg.value2 == (4, 5, 6)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_nested_config_with_missing_values_round_trips(extension):
    config = NestedSerializable.as_lazy()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"config.{extension}"
        # Values that were never set are skipped rather than serialized as
        # `MissingType`, which neither json nor safe yaml can represent.
        config.to_file(path)
        assert NestedSerializable.from_file(path) == config


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_to_file_creates_the_directory_it_writes_into(extension):
    config = ParsableSerializable.as_lazy(value=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runs" / "exp1" / f"config.{extension}"
        config.to_file(path)
        assert ParsableSerializable.from_file(path) == config


def test_to_file_does_not_treat_a_remote_path_as_a_directory(tmp_path, monkeypatch):
    # `s3://bucket/key` has no directory to create, and must not end up as a
    # local directory named `s3:` either.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParsableSerializable.as_lazy().to_file("s3://parsonaut-test/config.yaml")

    assert list(tmp_path.iterdir()) == []


def test_import_works_in_a_clean_interpreter():
    # The optional smart_open lookup used to reach for `importlib.util`
    # without importing it, which only worked if something else had already.
    subprocess.run([sys.executable, "-c", "import parsonaut"], check=True)


def test_format_of():
    assert format_of("config.yaml") == "yaml"
    assert format_of("config.yml") == "yaml"
    assert format_of("config.json") == "json"
    # Every suffix is considered, so a compressed config still counts as yaml.
    assert format_of("config.yaml.gz") == "yaml"

    with pytest.raises(ValueError, match="Unknown serialization format"):
        format_of("config.txt")


def test_a_config_saved_from_a_script_reloads_via_import(tmp_path):
    # `python train.py` records `_class: __main__.Model`. Loading has to work
    # from another file that imported the module, where `__main__` is that file.
    (tmp_path / "train.py").write_text(
        textwrap.dedent(
            """
            from parsonaut import Parsable

            class Model(Parsable):
                def __init__(self, n: int = 1):
                    pass

            class Opt(Parsable):
                def __init__(self, lr: float = 0.1):
                    pass

            class Params(Parsable):
                def __init__(
                    self,
                    model: Model = Model.as_lazy(),
                    opt: Opt = Opt.as_lazy(),
                ):
                    pass

            if __name__ == "__main__":
                Params.as_lazy(
                    model=Model.as_lazy(n=4),
                    opt=Opt.as_lazy(lr=0.5),
                ).to_file("cfg.yaml")
            """
        )
    )
    (tmp_path / "eval.py").write_text(
        textwrap.dedent(
            """
            import train

            cfg = train.Params.from_file("cfg.yaml")
            assert cfg.model.cls is train.Model, cfg.model.cls
            assert cfg.model.n == 4
            assert cfg.opt.cls is train.Opt
            assert cfg.opt.lr == 0.5
            """
        )
    )

    subprocess.run([sys.executable, "train.py"], cwd=tmp_path, check=True)
    subprocess.run([sys.executable, "eval.py"], cwd=tmp_path, check=True)


def test_maybe_import():
    assert maybe_import(Serializable) is Serializable
    assert maybe_import("parsonaut.serialization.Serializable") is Serializable

    with pytest.raises(ValueError, match="fully qualified"):
        maybe_import("Serializable")

    with pytest.raises(ImportError, match="Could not import"):
        maybe_import("parsonaut.serialization.Nope")
