import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        size = (out_features, in_features)
        self.weight = torch.nn.Parameter(torch.empty(size, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)
        
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        Y = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return Y

        