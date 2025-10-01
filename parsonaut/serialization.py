import importlib
import json
from pathlib import Path

import yaml


class DictSerializable:
    def to_dict(self, with_class_tag_as_str, tuples_as_lists) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplementedError


class YamlMixin(DictSerializable):
    def to_yaml(self, pth):
        dct = self.to_dict(with_class_tag_as_str=True, tuples_as_lists=True)
        save_yaml(dct, pth)

    @classmethod
    def from_yaml(cls, pth):
        dct = load_yaml(pth)
        return cls.from_dict(dct)


class JsonMixin(DictSerializable):
    def to_json(self, pth: str):
        dct = self.to_dict(with_class_tag_as_str=True, tuples_as_lists=True)
        save_json(dct, pth)

    @classmethod
    def from_json(cls, pth: str):
        dct = load_json(pth)
        return cls.from_dict(dct)


class Serializable(YamlMixin, JsonMixin):
    @classmethod
    def from_file(cls, path):
        return load_serializable(path, cls)

    def to_file(self, path) -> None:
        save_serializable(self, path)


def save_serializable(config: Serializable, path) -> None:
    if extension_contains(".json", path):
        config.to_json(path)
    elif extension_contains(".yaml", path):
        config.to_yaml(path)
    else:
        raise ValueError(f"Unknown serialization format for: {path}")


def load_serializable(path, cls):
    if extension_contains(".json", path):
        dct = load_json(path)
    elif extension_contains(".yaml", path):
        dct = load_yaml(path)
    else:
        raise ValueError(f"Unknown serialization format for: {path}")

    if cls == Serializable:
        cls = maybe_import(dct["_class"])
        return cls.from_dict(dct)
    else:
        return cls.from_dict(dct)


def extension_contains(ext: str, path) -> bool:
    return any(ext == sfx for sfx in Path(path).suffixes)


def load_yaml(pth):
    with open_best(pth, "r") as f:
        return yaml.safe_load(f)


def save_yaml(dct, pth):
    with open_best(pth, "w") as f:
        yaml.dump(dct, f)


def load_json(pth):
    with open_best(pth, "r") as f:
        return json.load(f)


def save_json(dct, pth):
    with open_best(pth, "w") as f:
        json.dump(dct, f, indent=4)


def maybe_import(cls_or_str):
    if isinstance(cls_or_str, str):
        module_name, class_name = cls_or_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    else:
        cls = cls_or_str
    return cls


def open_best(pth, mode):
    if is_module_available("smart_open"):
        from smart_open import open as open_

        return open_(pth, mode)
    else:
        return open(pth, mode)


def is_module_available(*modules: str) -> bool:
    import importlib

    return all(importlib.util.find_spec(m) is not None for m in modules)
