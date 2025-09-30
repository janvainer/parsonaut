import pytest

from parsonaut import Choices, Lazy, Parsable
from parsonaut.lazy import Missing
from parsonaut.parse import BOOL_FALSE_FLAGS, BOOL_TRUE_FLAGS, ArgumentParser, str2bool


@pytest.mark.parametrize(
    ("typ", "value"),
    [
        (str, "hello"),
        (float, 0.5),
        (int, 2),
    ],
)
def test_ArgumentParser_add_option_basic_types(typ, value):

    # default gets passed to parser
    parser = ArgumentParser()
    parser.add_option("hello", value, typ)
    args = parser.parse_args([])
    assert args.hello == value

    # value is required and reads from CLI
    parser = ArgumentParser()
    parser.add_option("hello", Missing, typ)
    args = parser.parse_args(["--hello", str(value)])
    assert args.hello == value


@pytest.mark.parametrize(
    ("typ", "value"), [(bool, item) for item in BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS]
)
def test_ArgumentParser_add_option_bool(typ, value):

    # default gets passed to parser
    parser = ArgumentParser()
    parser.add_option("hello", value, typ)
    args = parser.parse_args([])

    value = str2bool(value)
    assert args.hello == value

    # value is required and reads from CLI
    parser = ArgumentParser()
    parser.add_option("hello", Missing, typ)
    args = parser.parse_args(["--hello", str(value)])
    assert args.hello == value


@pytest.mark.parametrize(
    ("typ", "value"),
    [(tuple[typ], (typ(),)) for typ in [int, float, str]]
    + [(tuple[typ, typ], (typ(), typ())) for typ in [int, float, str]]
    + [(tuple[typ, ...], (typ(), typ(), typ())) for typ in [int, float, str]],
)
def test_ArgumentParser_add_option_flat_tuple(typ, value):

    # default gets passed to parser
    parser = ArgumentParser()
    parser.add_option("hello", value, typ)
    args = parser.parse_args([])

    assert args.hello == value

    # value is required and reads from CLI
    parser = ArgumentParser()
    parser.add_option("hello", Missing, typ)
    args = parser.parse_args(["--hello"] + [str(x) for x in value])
    assert args.hello == value


@pytest.mark.parametrize(
    ("typ", "value"),
    [(tuple[bool], (val,)) for val in BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS]
    + [(tuple[bool, bool], (val, val)) for val in BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS]
    + [
        (tuple[bool, ...], (val, val, val))
        for val in BOOL_TRUE_FLAGS + BOOL_FALSE_FLAGS
    ],
)
def test_ArgumentParser_add_flat_tuple_with_bools(typ, value):

    value_bool = tuple([str2bool(v) for v in value])
    # default gets passed to parser
    parser = ArgumentParser()
    parser.add_option("hello", value_bool, typ)
    args = parser.parse_args([])

    assert args.hello == value_bool

    # value is required and reads from CLI
    parser = ArgumentParser()
    parser.add_option("hello", Missing, typ)
    args = parser.parse_args(["--hello"] + list(value))
    assert args.hello == value_bool


class Inner(Parsable):
    def __init__(
        self,
        x,
        a: str,
        b: int = 1,
    ) -> None:
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
        c: Lazy[Inner, ...] = Inner.as_lazy(),
        d: str = "hello",
    ) -> None:
        pass


class Choice(Choices):
    I1 = Inner.as_lazy()
    I2 = Inner2.as_lazy()


class Outer2(Parsable):
    def __init__(
        self,
        c: Choice = Choice.I1,
        d: str = "hello",
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


def test_ArgumentParser_choices():
    parser = ArgumentParser()
    parser.add_options(Outer2.as_lazy())
    args = parser.parse_args(["--c", "I2", "--c.aa", "something"])
    assert args == Outer2.as_lazy(
        c=Inner2.as_lazy(aa="something"),
    )


class WithOptional(Parsable):
    def __init__(
        self,
        b: int | None = 1,
    ) -> None:
        pass


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
