import pytest

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
