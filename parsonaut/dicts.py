"""Converting between nested dicts and the dotted keys a config file uses."""


def flatten_dict(dct: dict) -> dict:
    """Turn nested keys into dotted ones.

    An empty dict is kept as a value of its own; dropping it would silently
    turn `{"a": {}}` into no update at all.
    """

    def _flatten(dct, prefix: str):
        out = list()
        for k, v in dct.items():
            if prefix:
                k = f"{prefix}.{k}"
            if isinstance(v, dict) and v:
                out.extend(_flatten(v, prefix=k))
            else:
                out.append((k, v))
        return out

    flat: dict = {}
    for key, value in _flatten(dct, ""):
        if key in flat:
            raise ValueError(f"{key!r} is specified more than once.")
        flat[key] = value
    return flat


def unflatten_dict(flat: dict) -> dict:
    """Turn dotted keys into nested ones."""
    base: dict = {}
    for key, value in flat.items():
        root = base
        path = key

        if "." in key:
            *parts, key = key.split(".")
            for depth, part in enumerate(parts):
                if part in root and not isinstance(root[part], dict):
                    raise ValueError(
                        f"{'.'.join(parts[: depth + 1])} is used both as a "
                        f"value and as a group of values, in {path!r}."
                    )
                root = root.setdefault(part, {})

        if isinstance(root.get(key), dict):
            raise ValueError(
                f"{path} is used both as a value and as a group of values."
            )
        root[key] = value

    return base
