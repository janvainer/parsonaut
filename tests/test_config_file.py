import json
from dataclasses import dataclass

import pytest

from parsonaut import Parsable
from parsonaut.parse import ArgumentParser


class SGD(Parsable):
    def __init__(self, params=None, lr: float = 1e-3, momentum: float = 0.0):
        pass


class Model(Parsable):
    def __init__(self, in_channels: int = 4, out_channels: int = 2):
        pass


class Wide(Model):
    def __init__(self, in_channels: int = 4, out_channels: int = 2, depth: int = 1):
        pass


class Narrow(Model):
    """Drops `out_channels` and adds `depth`, the awkward subclass shape."""

    def __init__(self, in_channels: int = 4, depth: int = 1):
        pass


@dataclass
class Box(Parsable):
    model: Model = Model.as_lazy()


class Labeled(Parsable):
    def __init__(
        self,
        country: str = "US",
        flag: bool = False,
        ratio: str = "1:1",
        lr: float = 0.1,
        bits: tuple[bool, ...] = (),
    ):
        pass


@dataclass
class Params(Parsable):
    model: Model = Model.as_lazy()
    opt: SGD = SGD.as_lazy()
    seed: int = 0


@pytest.fixture
def config(tmp_path):
    def write(text, name="config.yaml"):
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    return write


def parse(argv, **kwargs):
    parser = ArgumentParser(**kwargs)
    parser.add_options(Params.as_lazy())
    return parser.parse_args(argv)


def test_config_file_provides_defaults(config):
    path = config("seed: 7\nmodel.in_channels: 16\n")

    hp = parse(["--config", path])
    assert hp.seed == 7
    assert hp.model.in_channels == 16
    # Untouched values keep the class defaults.
    assert hp.model.out_channels == 2


def test_command_line_overrides_the_config_file(config):
    path = config("seed: 7\nmodel.in_channels: 16\n")

    hp = parse(["--config", path, "--seed", "99"])
    assert hp.seed == 99
    assert hp.model.in_channels == 16


def test_config_file_accepts_the_equals_form(config):
    path = config("seed: 7\n")
    assert parse([f"--config={path}"]).seed == 7


def test_config_file_accepts_nested_and_flat_keys(config):
    nested = config("model:\n  in_channels: 16\n", "nested.yaml")
    flat = config("model.in_channels: 16\n", "flat.yaml")

    assert parse(["--config", nested]).model.in_channels == 16
    assert parse(["--config", flat]).model.in_channels == 16


def test_config_file_can_be_json(config):
    path = config(json.dumps({"seed": 7}), "config.json")
    assert parse(["--config", path]).seed == 7


def test_a_saved_config_round_trips_through_the_flag(tmp_path):
    path = tmp_path / "saved.yaml"
    Params.as_lazy(seed=3, model=Model.as_lazy(in_channels=9)).to_file(path)

    hp = parse(["--config", str(path)])
    assert hp.seed == 3
    assert hp.model.in_channels == 9


def test_config_file_is_kept_out_of_the_parsed_configuration(config):
    path = config("seed: 7\n")
    hp = parse(["--config", path])

    assert hp.to_dict() == {
        "model": {"in_channels": 4, "out_channels": 2},
        "opt": {"lr": 1e-3, "momentum": 0.0},
        "seed": 7,
    }


def test_a_reused_parser_forgets_plain_defaults_from_a_config_file(config):
    path = config("extra: 5\nm.in_channels: 8\n")

    parser = ArgumentParser()
    parser.add_argument("--extra", type=int, default=0)
    parser.add_options(Model.as_lazy(), dest="m")

    first = parser.parse_args(["--config", path])
    assert first.extra == 5
    assert first.m.in_channels == 8

    second = parser.parse_args([])
    assert second.extra == 0
    assert second.m.in_channels == 4


def test_set_defaults_survives_a_config_file_on_a_reused_parser(config):
    path = config("extra: 5\n")

    parser = ArgumentParser()
    parser.add_argument("--extra", type=int, default=0)
    parser.set_defaults(extra=7)
    parser.add_options(Model.as_lazy(), dest="m")

    assert parser.parse_args(["--config", path]).extra == 5
    assert parser.parse_args([]).extra == 7


def test_config_file_can_specialize_a_nested_class(config):
    path = config(
        f"model:\n  _class: {Wide.__module__}.Wide\n  in_channels: 8\n  depth: 5\n"
    )

    hp = parse(["--config", path])
    assert hp.model.cls is Wide
    assert hp.model.in_channels == 8
    assert hp.model.depth == 5
    assert hp.model.out_channels == 2


def test_the_parser_can_be_reused_with_different_config_files(config):
    first = config("seed: 1\n", "first.yaml")
    second = config("seed: 2\n", "second.yaml")

    parser = ArgumentParser()
    parser.add_options(Params.as_lazy())

    assert parser.parse_args(["--config", first]).seed == 1
    assert parser.parse_args(["--config", second]).seed == 2
    # The files must not leak into a run without one.
    assert parser.parse_args([]).seed == 0


def test_unknown_key_in_a_config_file_names_the_file(config):
    path = config("sed: 7\n")

    with pytest.raises(ValueError, match=r"config\.yaml: .*'sed'"):
        parse(["--config", path])


def test_bad_value_in_a_config_file_names_the_file(config):
    path = config("seed: not_an_int\n")

    with pytest.raises(TypeError, match=r"config\.yaml:"):
        parse(["--config", path])


def test_config_file_cannot_switch_the_root_to_an_unrelated_class(config):
    path = config(f"_class: {__name__}.Model\n")

    with pytest.raises(TypeError, match="Cannot switch the configuration"):
        parse(["--config", path])


def test_an_empty_config_file_is_no_overrides(config):
    assert parse(["--config", config("")]).seed == 0
    assert parse(["--config", config("# just a comment\n")]).seed == 0
    assert parse(["--config", config("", "empty.json")]).seed == 0
    assert parse(["--config", config("   \n", "blank.json")]).seed == 0


def test_a_config_file_that_is_not_a_mapping(config):
    path = config("- one\n- two\n")

    with pytest.raises(
        ValueError, match="expected a mapping of values, got list"
    ) as exc:
        parse(["--config", path])
    # `load_dict` already names the file; wrapping it again doubled the path.
    assert str(exc.value).count(path) == 1


def test_a_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        parse(["--config", str(tmp_path / "nope.yaml")])


def test_config_file_can_be_yml(config):
    path = config("seed: 7\n", "config.yml")
    assert parse(["--config", path]).seed == 7


def test_config_file_with_an_unknown_extension(config):
    path = config("seed: 7\n", "config.txt")

    with pytest.raises(ValueError, match="Unknown serialization format"):
        parse(["--config", path])


def test_config_flag_can_be_renamed_or_disabled(config):
    path = config("seed: 7\n")

    assert parse([f"--cfg={path}"], config_flag="--cfg").seed == 7

    parser = ArgumentParser(config_flag=None)
    parser.add_options(Params.as_lazy())
    assert "--config" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", path])


def test_config_flag_reports_a_clash_with_a_parameter():
    class HasConfig(Parsable):
        def __init__(self, config: str = "x"):
            pass

    parser = ArgumentParser()
    parser.add_options(HasConfig.as_lazy())

    with pytest.raises(ValueError, match="clashes with an argument"):
        parser.parse_args([])

    parser = ArgumentParser(config_flag=None)
    parser.add_options(HasConfig.as_lazy())
    assert parser.parse_args(["--config", "x.yaml"]).config == "x.yaml"


def test_one_config_file_serves_several_destinations(config):
    path = config("m.in_channels: 8\nopt.lr: 0.5\nextra: 5\n")

    parser = ArgumentParser()
    parser.add_argument("--extra", type=int, default=0)
    parser.add_options(Model.as_lazy(), dest="m")
    parser.add_options(SGD.as_lazy(), dest="opt")

    args = parser.parse_args(["--config", path])
    assert args.m.in_channels == 8
    assert args.opt.lr == 0.5
    assert args.extra == 5


def test_parse_args_exposes_the_config_flag(monkeypatch, config):
    path = config("seed: 7\n")
    monkeypatch.setattr("sys.argv", ["prog", "--config", path, "--seed", "8"])

    hp = Params.parse_args()
    assert hp.seed == 8


def test_config_file_can_switch_to_a_subclass_with_different_fields(config):
    path = config(
        f"model:\n  _class: {Narrow.__module__}.Narrow\n  in_channels: 8\n  depth: 2\n"
    )

    parser = ArgumentParser()
    parser.add_options(Box.as_lazy())

    hp = parser.parse_args(["--config", path, "--model.depth", "9"])
    assert hp.model.cls is Narrow
    assert hp.model.in_channels == 8
    assert hp.model.depth == 9

    # A flag that only existed on the base class is no longer an argument.
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", path, "--model.out_channels", "3"])

    # The next run, with no file, is the base class again.
    base = parser.parse_args([])
    assert base.model.cls is Model
    assert base.model.out_channels == 2


def test_yaml_keeps_strings_that_look_like_bools_or_times(config):
    path = config("country: NO\nflag: yes\nratio: 16:9\nlr: 1.0e-3\nbits: [yes, no]\n")

    parser = ArgumentParser()
    parser.add_options(Labeled.as_lazy())
    hp = parser.parse_args(["--config", path])

    assert hp.country == "NO"
    assert hp.flag is True
    assert hp.ratio == "16:9"
    assert hp.lr == 0.001
    assert hp.bits == (True, False)


def test_config_flag_appears_in_help():
    parser = ArgumentParser(prog="demo")
    parser.add_options(Params.as_lazy())

    assert "--config path" in parser.format_help()
