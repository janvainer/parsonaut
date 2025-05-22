import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from parsonaut import Parsable


class DummySerializableModule(nn.Module, Parsable):
    def __init__(self, value: int):
        super().__init__()
        self.value = value
        self.layer = nn.Linear(value, value)


def test_from_to_checkpoint():
    module = DummySerializableModule(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "module"
        path.mkdir(parents=True)
        module.to_checkpoint(path)
        module2 = DummySerializableModule.from_checkpoint(path)
        assert module.value == module2.value
        sd2 = module2.state_dict()
        for k, v in module.state_dict().items():
            torch.testing.assert_close(v, sd2[k])
