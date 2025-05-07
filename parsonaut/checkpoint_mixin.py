from .lazy import Parsable
from .serialization import is_module_available, open_best

if is_module_available("torch"):
    import torch

    class CheckpointMixin(Parsable):
        def state_dict(self):
            raise NotImplementedError()

        def load_state_dict(self):
            raise NotImplementedError()

        @classmethod
        def from_checkpoint(cls, pth):
            pth = str(pth).rstrip("/")
            obj = cls.from_file(f"{pth}/config.yaml").to_eager()

            with open_best("{pth}/weights.pt", "r") as f:
                state_dict = torch.load(f)
            obj.load_state_dict(state_dict)
            return obj

        def to_checkpoint(self, pth):
            pth = str(pth)
            self.to_file(f"{pth}/config.yaml")
            state_dict = self.state_dict()
            with open_best(f"{pth}/weights.pt", "w") as f:
                torch.save(state_dict, f)

else:

    class CheckpointMixin(Parsable):
        def __init_subclass__(cls, **kwargs):
            raise ImportError(
                "torch is not installed. Please install PyTorch to use this module."
            )
