import argparse
import io
import sys
from argparse import ArgumentTypeError
from typing import Literal

import pytest

from parsonaut import Lazy, Parsable
from parsonaut.lazy import Missing
from parsonaut.parse import (
    BOOL_FALSE_FLAGS,
    BOOL_TRUE_FLAGS,
    ArgumentParser,
    _add_option,
    str2bool,
)


def parse_option(typ, default, argv):
    """What a single `typ` option makes of `argv`, given `default`."""
    parser = argparse.ArgumentParser()
    _add_option(parser, "hello", default, typ)
    return parser.parse_args(argv).hello


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (str, "hello"),
        (float, 0.5),
        (int, 2),
    ],
)
def test_option_of_a_basic_type(typ, value):
    assert parse_option(typ, value, []) == value
    # A value with no default has to be read off the command line.
    assert parse_option(typ, Missing, ["--hello", str(value)]) == value


@pytest.mark.parametrize("value", BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS)
def test_option_of_bool_type(value):
    expected = str2bool(value)
    assert parse_option(bool, value, []) == expected
    assert parse_option(bool, Missing, ["--hello", str(expected)]) == expected


@pytest.mark.parametrize(
    ("typ", "value"),
    [(tuple[typ], (typ(),)) for typ in [int, float, str]]
    + [(tuple[typ, typ], (typ(), typ())) for typ in [int, float, str]]
    + [(tuple[typ, ...], (typ(), typ(), typ())) for typ in [int, float, str]],
)
def test_option_of_a_flat_tuple_type(typ, value):
    assert parse_option(typ, value, []) == value

    argv = ["--hello", *(str(x) for x in value)]
    assert parse_option(typ, Missing, argv) == value


@pytest.mark.parametrize(
    ("typ", "length"),
    [(tuple[bool], 1), (tuple[bool, bool], 2), (tuple[bool, ...], 3)],
)
@pytest.mark.parametrize("value", BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS)
def test_option_of_a_flat_tuple_of_bools(typ, length, value):
    expected = (str2bool(value),) * length

    assert parse_option(typ, expected, []) == expected
    assert parse_option(typ, Missing, ["--hello", *([value] * length)]) == expected


class Inner(Parsable):
    def __init__(
        self,
        x,
        a: str,
        b: int = 1,
    ) -> None:
        pass


class InnerSub(Inner):
    pass


class Inner2(Parsable):
    def __init__(
        self,
        aa: str,
        bb: int | None = 1,
    ) -> None:
        pass


class Outer(Parsable):
    def __init__(
        self,
        c: Inner = Inner.as_lazy(),
        d: str = "hello",
    ) -> None:
        pass


class WithOptional(Parsable):
    def __init__(
        self,
        b: int | None = 1,
    ) -> None:
        pass


def test_ArgumentParser_add_options_flat():
    parser = ArgumentParser()
    parser.add_options(Inner.as_lazy())

    args = parser.parse_args(["--a", "3"])
    assert args == Inner.as_lazy(a="3")


def test_ArgumentParser_add_options_nested():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy())

    args = parser.parse_args(["--c.a", "3", "--d", "okay"])
    assert args == Outer.as_lazy(
        c=Inner.as_lazy(a="3"),
        d="okay",
    )


def test_ArgumentParser_keeps_sibling_args_out_of_a_dest_group():
    parser = ArgumentParser()
    parser.add_argument("--c_lr", type=float, default=0.1)
    parser.add_options(Inner.as_lazy(a="x"), dest="c")

    args = parser.parse_args([])
    assert args.c_lr == 0.1
    assert args.c == Inner.as_lazy(a="x")


def test_ArgumentParser_can_be_reused():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy())

    assert parser.parse_args(["--d", "first"]).d == "first"
    assert parser.parse_args(["--d", "second"]).d == "second"


def test_ArgumentParser_optional_tuple_with_a_none_default():
    class OptionalTuple(Parsable):
        def __init__(self, xs: tuple[int, ...] | None = None) -> None:
            pass

    parser = ArgumentParser()
    parser.add_options(OptionalTuple.as_lazy())

    assert parser.parse_args([]).xs is None
    assert parser.parse_args(["--xs", "1", "2"]).xs == (1, 2)


def test_ArgumentParser_requires_flags_to_be_spelled_out():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy())

    with pytest.raises(SystemExit):
        parser.parse_args(["--d.", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--conf", "x.yaml"])


def test_ArgumentParser_abbreviations_can_be_switched_back_on(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("d: from-file\n")

    parser = ArgumentParser(allow_abbrev=True)
    parser.add_options(Outer.as_lazy())

    assert parser.parse_args(["--conf", str(path)]).d == "from-file"


def test_a_bool_option_rejects_a_value_that_is_neither():
    with pytest.raises(SystemExit):
        parse_option(bool, True, ["--hello", "maybe"])

    with pytest.raises(ArgumentTypeError, match="Boolean value expected"):
        str2bool("maybe")


@pytest.mark.parametrize(
    ("typ", "default", "text", "expected"),
    [
        (Literal["train", "eval"], "train", "eval", "eval"),
        (Literal[1, 2, 3], 1, "2", 2),
        (Literal[1.0, 2.5], 1.0, "2.5", 2.5),
        (Literal[False, True], False, "yes", True),
    ],
)
def test_option_of_a_literal(typ, default, text, expected):
    assert parse_option(typ, default, []) == default
    assert parse_option(typ, Missing, ["--hello", text]) == expected


def test_a_literal_option_rejects_a_value_outside_its_choices():
    with pytest.raises(SystemExit):
        parse_option(Literal["train", "eval"], "train", ["--hello", "nope"])


def test_a_union_of_literals_is_one_option():
    typ = Literal["train"] | Literal["eval"]
    assert parse_option(typ, "train", ["--hello", "eval"]) == "eval"
    assert parse_option(typ | None, "train", ["--hello"]) is None


def test_an_optional_literal_accepts_a_bare_flag_as_none():
    typ = Literal["train", "eval"] | None
    assert parse_option(typ, "train", ["--hello"]) is None
    assert parse_option(typ, None, ["--hello", "eval"]) == "eval"


def test_literal_options_are_described_in_help():
    parser = argparse.ArgumentParser()
    _add_option(parser, "mode", "train", Literal["train", "eval"])

    help_text = parser.format_help()
    assert "--mode {train,eval}" in help_text


def test_unsupported_option_types_are_reported():
    with pytest.raises(TypeError, match="not a supported option type"):
        parse_option(list[str], Missing, [])


@pytest.mark.parametrize("typ", [bool, int, str, float, tuple[int, int]])
def test_an_unset_option_stays_missing(typ):
    assert parse_option(typ, Missing, []) is Missing


def test_ArgumentParser_format_help():
    parser = ArgumentParser(prog="demo")
    parser.add_options(Outer.as_lazy())

    help_text = parser.format_help()
    assert "--c.a str" in help_text
    assert "--d str" in help_text
    assert "_class" not in help_text


def test_ArgumentParser_parse_known_args():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy(), dest="cfg")

    args, remaining = parser.parse_known_args(["--cfg.d", "set", "--unknown", "1"])
    assert args.cfg.d == "set"
    assert remaining == ["--unknown", "1"]


def test_ArgumentParser_set_defaults():
    parser = ArgumentParser()
    parser.add_argument("--extra", type=int)
    parser.set_defaults(extra=7, only_a_default="x")

    args = parser.parse_args([])
    assert args.extra == 7
    assert args.only_a_default == "x"
    assert parser.parse_args(["--extra", "1"]).extra == 1

    parser = ArgumentParser()
    parser.set_defaults(extra=7)
    with pytest.raises(ValueError, match="other args are present"):
        parser.add_options(Outer.as_lazy())


def test_ArgumentParser_argument_groups():
    parser = ArgumentParser(prog="demo")
    group = parser.add_argument_group("extras", description="not config")
    group.add_argument("--extra", type=int, default=1)
    parser.add_options(Outer.as_lazy(), dest="cfg")

    assert parser.parse_args([]).extra == 1
    assert parser.parse_args(["--extra", "2"]).extra == 2

    help_text = parser.format_help()
    assert "extras" in help_text
    assert "not config" in help_text


def test_ArgumentParser_mutually_exclusive_groups():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--a", action="store_true")
    group.add_argument("--b", action="store_true")

    assert parser.parse_args(["--a"]).a is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--a", "--b"])


def test_ArgumentParser_usage_and_error(capsys):
    parser = ArgumentParser(prog="demo")
    parser.add_options(Outer.as_lazy())

    assert parser.format_usage().startswith("usage: demo")
    parser.print_usage()
    assert capsys.readouterr().out == parser.format_usage()

    with pytest.raises(SystemExit):
        parser.error("something is wrong")
    captured = capsys.readouterr()
    assert "usage: demo" in captured.err
    assert "something is wrong" in captured.err
    assert captured.out == ""


class _WriteOnly(io.TextIOBase):
    """A stderr stand-in that raises if anything tries to read it back."""

    def __init__(self):
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        self.parts.append(s)
        return len(s)

    def readable(self) -> bool:
        return False

    def readline(self, size: int = -1) -> str:
        raise io.UnsupportedOperation("not readable")


def test_a_parse_error_survives_a_non_readable_stderr(monkeypatch):
    sink = _WriteOnly()
    monkeypatch.setattr(sys, "stderr", sink)

    parser = ArgumentParser(prog="demo")
    parser.add_options(Outer.as_lazy())

    with pytest.raises(SystemExit):
        parser.parse_args(["--nope"])

    text = "".join(sink.parts)
    assert "usage: demo" in text
    assert "unrecognized arguments" in text


def test_ArgumentParser_print_help(capsys):
    parser = ArgumentParser(prog="demo")
    parser.add_options(Outer.as_lazy())

    parser.print_help()
    assert capsys.readouterr().out == parser.format_help()


def test_ArgumentParser_add_options_rejects_a_non_config():
    parser = ArgumentParser()
    with pytest.raises(TypeError, match="Expected a configuration"):
        parser.add_options(Inner(x=1, a="built"))


def test_ArgumentParser_rejects_mixing_options_with_a_root_lazy():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy())
    with pytest.raises(ValueError, match="Cannot add more options"):
        parser.add_argument("--extra", type=int)

    parser = ArgumentParser()
    parser.add_argument("--extra", type=int)
    with pytest.raises(ValueError, match="other args are present"):
        parser.add_options(Outer.as_lazy())

    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy(), dest="a")
    with pytest.raises(ValueError, match="Duplicate destination"):
        parser.add_options(Outer.as_lazy(), dest="a")

    parser = ArgumentParser()
    with pytest.raises(ValueError, match="cannot contain a dot"):
        parser.add_options(Outer.as_lazy(), dest="a.b")


@pytest.mark.parametrize("options_first", [True, False])
def test_ArgumentParser_rejects_a_dest_taken_by_a_plain_argument(options_first):
    parser = ArgumentParser()
    if options_first:
        parser.add_options(Inner.as_lazy(a="x"), dest="c")
        parser.add_argument("--c", type=str)
    else:
        parser.add_argument("--c", type=str)
        parser.add_options(Inner.as_lazy(a="x"), dest="c")

    with pytest.raises(ValueError, match="Duplicate destination name: 'c'"):
        parser.parse_args([])


def test_ArgumentParser_rejects_a_dest_taken_by_an_argument_in_a_group():
    parser = ArgumentParser()
    parser.add_options(Inner.as_lazy(a="x"), dest="c")
    parser.add_argument_group("group").add_argument("--c", type=str)

    with pytest.raises(ValueError, match="Duplicate destination name: 'c'"):
        parser.parse_args([])


def test_ArgumentParser_does_not_expose_class_flags():
    parser = ArgumentParser()
    parser.add_options(Outer.as_lazy())

    with pytest.raises(SystemExit):
        parser.parse_args([f"--_class={__name__}.Inner"])
    with pytest.raises(SystemExit):
        parser.parse_args([f"--c._class={__name__}.Outer"])


def make_thing(width: int = 8, name: str = "m"):
    return f"{name}/{width}"


def test_ArgumentParser_parses_a_factory_function_config():
    parser = ArgumentParser()
    parser.add_options(Lazy.from_class(make_thing))

    assert parser.parse_args(["--width", "3"]).to_eager() == "m/3"


def test_ArgumentParser_simple_optional():
    parser = ArgumentParser()
    parser.add_options(WithOptional.as_lazy())
    args = parser.parse_args([])
    assert args == WithOptional.as_lazy(b=1)

    parser = ArgumentParser()
    parser.add_options(WithOptional.as_lazy())
    args = parser.parse_args(["--b", "5"])
    assert args == WithOptional.as_lazy(b=5)

    parser = ArgumentParser()
    parser.add_options(WithOptional.as_lazy())
    args = parser.parse_args(["--b"])
    assert args == WithOptional.as_lazy(b=None)


def test_parse_args_returns_the_namespace_it_was_given():
    parser = ArgumentParser()
    parser.add_argument("--extra", type=int, default=1)
    supplied = argparse.Namespace(kept="yes")

    out = parser.parse_args(["--extra", "3"], namespace=supplied)

    assert out is supplied
    assert out.extra == 3
    assert out.kept == "yes"
    assert not hasattr(out, "config")

    parser = ArgumentParser()
    parser.add_argument("--extra", type=int, default=0)
    parser.add_options(Inner.as_lazy(a="x"), dest="c")
    supplied = argparse.Namespace()

    out = parser.parse_args(["--c.a", "z", "--extra", "2"], namespace=supplied)

    assert out is supplied
    assert out.extra == 2
    assert out.c.a == "z"
    assert not hasattr(out, "c.a")


def test_optional_fixed_tuple_accepts_a_bare_flag_as_none():
    assert parse_option(tuple[int, int] | None, (1, 2), []) == (1, 2)
    assert parse_option(tuple[int, int] | None, (1, 2), ["--hello"]) is None
    assert parse_option(tuple[int, int] | None, None, ["--hello", "1", "2"]) == (1, 2)
    with pytest.raises(SystemExit):
        parse_option(tuple[int, int] | None, None, ["--hello", "1"])


def test_ArgumentParser_optional_tuple_with_a_bare_flag():
    class Tuples(Parsable):
        def __init__(
            self,
            xs: tuple[int, ...] | None = None,
            ys: tuple[int, ...] = (1,),
        ) -> None:
            pass

    parser = ArgumentParser()
    parser.add_options(Tuples.as_lazy())

    assert parser.parse_args(["--xs"]).xs is None
    assert parser.parse_args(["--ys"]).ys == ()


def test_ArgumentParser_marks_options_without_a_default():
    parser = ArgumentParser(prog="demo")
    parser.add_options(Inner.as_lazy())

    help_text = parser.format_help()
    assert "--a str" in help_text and "no default" in help_text
