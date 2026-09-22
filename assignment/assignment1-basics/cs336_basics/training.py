import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    N = len(dataset)
    # 从 dataset 里随机取 batch_size 个起始点
    start = np.random.randint(0, N - context_length, size=batch_size)
    offsets = np.arange(context_length)
    idx = start[:, None] + offsets
    x = dataset[idx]
    y = dataset[idx + 1]
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return (x, y)


    