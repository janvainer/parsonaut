import importlib
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
JSON_SUFFIXES = (".json",)

#: The key a serialized configuration records its class under.
TYPE_NAME = "_class"


class Serializable:
    """Reads and writes itself as yaml or json, via `to_dict` / `from_dict`."""

    # Keyword-only so that subclasses are free to add their own parameters.
    def to_dict(
        self,
        *,
        class_tag: bool | str = False,
        tuples_as_lists: bool = False,
        skip_missing: bool = False,
    ) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct: dict):
        raise NotImplementedError

    @classmethod
    def from_file(cls, path):
        dct = load_dict(path)
        if cls is Serializable:
            # The caller does not know the class, so read it back from the file.
            cls = maybe_import(dct[TYPE_NAME])
        return cls.from_dict(dct)

    def to_file(self, path) -> None:
        # A file has to name the class it holds and spell tuples as lists to
        # be readable again, and values that were never set are left out
        # rather than written as a sentinel neither format can represent.
        save_dict(
            self.to_dict(class_tag="str", tuples_as_lists=True, skip_missing=True),
            path,
        )


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
    return dct


def save_dict(dct: dict, path) -> None:
    """Write a plain dict as yaml or json, according to the file extension."""
    is_json = format_of(path) == "json"
    with open_best(path, "w") as f:
        if is_json:
            json.dump(dct, f, indent=4)
        else:
            # safe_dump keeps configs readable and loadable by `yaml.safe_load`;
            # it raises instead of emitting `!!python/object` tags.
            yaml.safe_dump(dct, f, sort_keys=True)


def maybe_import(cls_or_str: Any, *, fallback: Any = None) -> Any:
    """Resolve a `module.ClassName` string to the class itself.

    Warning: this imports the named module, so configs should be treated with
    the same care as any other executable input.

    A class saved from a script is tagged `__main__.ClassName`, which only
    exists on the process that saved it. `fallback` is the class the caller
    asked for (as in `Model.from_file`); a nested tag is resolved by finding
    the one loaded class of that name.
    """
    if not isinstance(cls_or_str, str):
        return cls_or_str

    if "." not in cls_or_str:
        raise ValueError(
            "Expected a fully qualified class name such as 'my_module.MyClass', "
            f"got {cls_or_str!r}."
        )
    module_name, class_name = cls_or_str.rsplit(".", 1)
    if module_name == "__main__":
        return _import_main_class(class_name, cls_or_str, fallback)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        # A class defined inside a function is tagged `module.Name` but is not
        # an attribute of that module. The caller already has it in hand.
        if _is_tagged_class(fallback, module_name, class_name):
            return fallback
        raise ImportError(f"Could not import {cls_or_str!r}: {exc}") from exc


def _is_tagged_class(cls, module_name: str, class_name: str) -> bool:
    return (
        isinstance(cls, type)
        and cls.__module__ == module_name
        and cls.__name__ == class_name
    )


def _import_main_class(class_name: str, qualname: str, fallback):
    main = sys.modules.get("__main__")
    on_main = getattr(main, class_name, None) if main is not None else None
    # The running script, or `from train import Model`, binds the name here.
    if isinstance(on_main, type) and (fallback is None or on_main is fallback):
        return on_main

    # `import train; train.Model.from_file(...)` names the class to load even
    # though the file still says `__main__.Model`.
    if isinstance(fallback, type) and fallback.__name__ == class_name:
        return fallback

    matches = _classes_named(class_name)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        modules = ", ".join(sorted({item.__module__ for item in matches}))
        raise ImportError(
            f"Could not import {qualname!r}: {class_name} is defined in more "
            f"than one loaded module ({modules})."
        )
    detail = (
        f"module '__main__' has no attribute {class_name!r}"
        if not isinstance(on_main, type)
        else f"module '__main__' has a different {class_name!r}"
    )
    raise ImportError(f"Could not import {qualname!r}: {detail}.")


def _classes_named(class_name: str) -> list[type]:
    """Classes defined (not merely re-exported) under `class_name`."""
    found: list[type] = []
    seen: set[int] = set()
    for module in list(sys.modules.values()):
        if module is None:
            continue
        candidate = getattr(module, class_name, None)
        if (
            isinstance(candidate, type)
            and candidate.__name__ == class_name
            and candidate.__module__ == getattr(module, "__name__", None)
            and id(candidate) not in seen
        ):
            seen.add(id(candidate))
            found.append(candidate)
    return found


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
