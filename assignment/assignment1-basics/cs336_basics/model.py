import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
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



class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        size = (num_embeddings, embedding_dim)
        self.weight = torch.nn.Parameter(torch.empty(size, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)
    
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]

class RMSnorm(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        size = (d_model,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(size, device=device, dtype=dtype))
        torch.nn.init.ones_(self.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        rms = torch.sqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        x_rms = x / rms
        Y = einsum(self.weight, x_rms, "d_model, ... d_model -> ... d_model")
        return Y.to(in_dtype)

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)       

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = SiLU(self.w1.forward(x))
        y2 = self.w3.forward(x)
        y_final = self.w2.forward(y1 * y2)
        return y_final
