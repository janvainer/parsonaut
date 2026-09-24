import json
import re
from functools import cache
from pathlib import Path

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
JSON_SUFFIXES = (".json",)


class Serializable:
    """Reads and writes itself as yaml or json, via `to_dict` / `from_dict`."""

    # Keyword-only so that subclasses are free to add their own parameters.
    def to_dict(
        self,
        *,
        skip_missing: bool = False,
    ) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct: dict):
        raise NotImplementedError

    @classmethod
    def from_file(cls, path):
        if cls is Serializable:
            raise TypeError(
                "Serializable.from_file() cannot choose a class. Call it on "
                "the class the file configures, for example Model.from_file(path)."
            )
        return cls.from_dict(load_dict(path))

    def to_file(self, path) -> None:
        # Values that were never set are left out rather than written as a
        # sentinel neither format can represent. Tuples are stored as lists
        # by `save_dict`, which is the form both formats can write.
        save_dict(self.to_dict(skip_missing=True), path)


def format_of(path) -> str:
    """Whether `path` names a yaml or a json file.

    Every suffix is considered rather than only the last, so that a compressed
    `config.yaml.gz` still counts as yaml.
    """
    suffixes = set(Path(path).suffixes)
    if suffixes.intersection(JSON_SUFFIXES):
        return "json"
    if suffixes.intersection(YAML_SUFFIXES):
        return "yaml"
    raise ValueError(
        f"Unknown serialization format for: {path} "
        f"(expected one of {', '.join(YAML_SUFFIXES + JSON_SUFFIXES)})"
    )


def load_dict(path) -> dict:
    """Read a yaml or json file into a plain dict."""
    is_json = format_of(path) == "json"
    try:
        with open_best(path, "r") as f:
            raw = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No such file: {path}") from exc

    # An empty file simply has nothing to say. json rejects that as a
    # document, while yaml already returns None; both mean "no overrides".
    if not raw.strip():
        return {}
    dct = json.loads(raw) if is_json else yaml.load(raw, Loader=_ConfigLoader)

    if dct is None:
        return {}
    if not isinstance(dct, dict):
        raise ValueError(
            f"{path}: expected a mapping of values, got {type(dct).__name__}."
        )
    return _lists_to_tuples(dct)


def _tuples_to_lists(value):
    """Tuples become lists, which is what json and yaml can store."""
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    return value


def _lists_to_tuples(value):
    """Lists become tuples. A file written here stored every tuple that way."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


def save_dict(dct: dict, path) -> None:
    """Write a plain dict as yaml or json, according to the file extension."""
    dct = _tuples_to_lists(dct)
    is_json = format_of(path) == "json"
    with open_best(path, "w") as f:
        if is_json:
            json.dump(dct, f, indent=4)
        else:
            # safe_dump keeps configs readable and loadable by `yaml.safe_load`;
            # it raises instead of emitting `!!python/object` tags.
            yaml.safe_dump(dct, f, sort_keys=True)


class _ConfigLoader(yaml.SafeLoader):
    """YAML 1.2-style scalars: numbers and null, not `yes` or `16:9`."""

    # An empty dict on the class itself, so PyYAML does not copy SafeLoader's
    # YAML 1.1 resolvers (`yes`/`NO` as bools, `16:9` as an integer, octal).
    yaml_implicit_resolvers: dict = {}


def _register_config_resolvers() -> None:
    int_re = re.compile(
        r"^(?:[-+]?0b[0-1_]+|[-+]?0x[0-9a-fA-F_]+|[-+]?(?:0|[1-9][0-9_]*))$",
        re.X,
    )
    # Same floats as PyYAML, without sexagesimal (`16:9`). A bare `1e-3` stays
    # text, matching PyYAML; `1.0e-3` is a float.
    float_re = re.compile(
        r"^(?:"
        r"[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?"
        r"|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)"
        r"|\.(?:nan|NaN|NAN)"
        r")$",
        re.X,
    )
    null_re = re.compile(r"^(?:~|null|Null|NULL)$", re.X)
    # YAML 1.2 booleans. `yes` and `no` stay text.
    bool_re = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$", re.X)
    _ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:bool", bool_re, list("tTfF"))
    _ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:null", null_re, list("~nN"))
    _ConfigLoader.add_implicit_resolver(
        "tag:yaml.org,2002:int", int_re, list("-+0123456789")
    )
    _ConfigLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float", float_re, list("-+0123456789.")
    )


_register_config_resolvers()


def open_best(pth, mode):
    """Open a local path, or a remote one if `smart_open` is installed."""
    if set(mode) & set("wxa"):
        _make_parent_dir(pth)

    opener = _smart_open()
    return opener(pth, mode) if opener is not None else open(pth, mode)


@cache
def _smart_open():
    """`smart_open.open` if it is installed, looked up once."""
    try:
        from smart_open import open as open_  # type: ignore[import-not-found]
    except ImportError:
        return None
    return open_


def _make_parent_dir(pth) -> None:
    """Create the directory a local path is about to be written to.

    Remote locations such as `s3://bucket/key` have no directories to create,
    and must not be mistaken for a relative path.
    """
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(pth)):
        return
    Path(pth).parent.mkdir(parents=True, exist_ok=True)
