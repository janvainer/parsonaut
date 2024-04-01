from typing import Any, Callable, Type


class MissingType:
    pass


Missing = MissingType()


def get_signature(func: Callable, *args, **kwargs) -> dict[str, tuple[Type, Any]]:
    from inspect import _empty, signature

    sig = signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    ret = dict()
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue

        value = bound.arguments.get(
            param_name, param.default if param.default != _empty else Missing
        )
        annotation = param.annotation if param.annotation != _empty else MissingType
        ret[param_name] = (annotation, value)

    return ret
