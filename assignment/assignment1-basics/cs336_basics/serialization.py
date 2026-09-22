import torch
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

# 将参数持久化存入磁盘中，先拼再存
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    dic = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(dic, out)

#从磁盘中读取参数
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    dic = torch.load(src)
    model.load_state_dict(dic["model"])
    optimizer.load_state_dict(dic["optimizer"])
    return dic["iteration"]