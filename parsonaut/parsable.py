from abc import ABCMeta
from typing import Callable, Type

from .lazy import Lazy, P, T
from .serialization import Serializable, is_module_available, open_best


class ParsableMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):

        cfg = Lazy.from_class(cls, *args, skip_non_parsable=True, **kwargs)

        # https://stackoverflow.com/a/73923070/8378586
        obj = cls.__new__(cls, *args, **kwargs)
        obj._cfg = cfg
        # Initialize the final object
        obj.__init__(*args, **kwargs)
        return obj


class class_or_instance_method(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, instance, owner):
        if instance is not None:
            class_or_instance = instance
        else:
            class_or_instance = owner

        def newfunc(*args, **kwargs):
            return self.f(class_or_instance, *args, **kwargs)

        return newfunc


if is_module_available("torch"):
    import torch
else:
    torch = None


class Parsable(Serializable, metaclass=ParsableMeta):

    _cfg: Lazy

    @class_or_instance_method
    def as_lazy(cls_or_self, *args, **kwargs) -> Lazy:
        if isinstance(cls_or_self, Parsable):
            assert not args, "Cannot override once configured Parsable."
            assert not kwargs, "Cannot override once configured Parsable."
            return cls_or_self._cfg
        else:
            return Lazy.from_class(cls_or_self, *args, **kwargs)

    def to_dict(
        self,
        with_class_tag: bool = False,
        with_class_tag_as_str: bool = False,
        flatten: bool = False,
        tuples_as_lists: bool = False,
    ):
        return self._cfg.to_dict(
            with_class_tag=with_class_tag,
            with_class_tag_as_str=with_class_tag_as_str,
            flatten=flatten,
            tuples_as_lists=tuples_as_lists,
        )

    @classmethod
    def from_dict(cls, dct) -> Lazy:
        return Lazy.from_class(cls).from_dict(dct)

    @classmethod
    def parse_args(cls: Type[T] | Callable[P, T], *args, **kwargs) -> Lazy[T, P]:
        from .parse import ArgumentParser

        parser = ArgumentParser()
        parser.add_options(cls.as_lazy(*args, **kwargs))
        params = parser.parse_args()
        return params

    # Methods related to torch serialization
    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()

    @classmethod
    def from_checkpoint(cls, pth):
        assert (
            torch is not None
        ), f"Loading {cls} from checkpoint requires torch installed."

        pth = str(pth).rstrip("/")
        obj = cls.from_file(f"{pth}/config.yaml").to_eager()

        with open_best(f"{pth}/weights.pt", "rb") as f:
            state_dict = torch.load(f)
        obj.load_state_dict(state_dict)
        return obj

    def to_checkpoint(self, pth):
        assert (
            torch is not None
        ), f"Saving {self.__class__} to checkpoint requires torch installed."

        pth = str(pth)
        self.to_file(f"{pth}/config.yaml")
        state_dict = self.state_dict()
        with open_best(f"{pth}/weights.pt", "wb") as f:
            torch.save(state_dict, f)
