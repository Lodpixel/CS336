import torch
import math
from einops import einsum, rearrange

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
        # rms 这一步逐行归约，keepdim 用于保证维度不消失，留下一，便于后续广播
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
        # 直接创建三个 Linear 进行复用
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按公式逐步操作：SwiGLU(x) = W_2 * (SiLU(W_1 * x) dot_* (W_3 * x))
        y1 = SiLU(self.w1.forward(x))
        y2 = self.w3.forward(x)
        y_final = self.w2.forward(y1 * y2)
        return y_final

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0
        # 一次初始化就可以把 sin 和 cos 值给提前计算出来，只需一次初始化，不需要每个 layer 重新算
        table_x = torch.arange(0, max_seq_len)
        exp_tensor = torch.arange(0, d_k // 2) * 2.0 / d_k
        table_y = 1.0 / torch.pow(theta, exp_tensor)
        mid_table = einsum(table_x, table_y, "max_seq_len, pair -> max_seq_len pair")
        # 存到 register_buffer，这样在数据搬到 gpu 的时候会自动一起被搬过去
        self.register_buffer("cos", torch.cos(mid_table), persistent=False)
        self.register_buffer("sin", torch.sin(mid_table), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        # cos 和 sin 的维度：(... seq_len d_k // 2)
        cos_table = self.cos[token_positions]
        sin_table = self.sin[token_positions]
        # x rearrange 后的维度：(... d_k // 2 2)
        x_pairs = rearrange(x, "... (pair two) -> ... pair two", two=2)
        a_pair = x_pairs[..., 0] * cos_table - x_pairs[..., 1] * sin_table
        b_pair = x_pairs[..., 0] * sin_table + x_pairs[..., 1] * cos_table
        rotated_pair = torch.stack([a_pair, b_pair], dim=-1)
        ans = rearrange(rotated_pair, "... pair two -> ... (pair two)")
        return ans

RoPE = RotaryPositionalEmbedding

def softmax(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    # 直接取值
    maxnum = x.amax(dim=dim, keepdim=True)
    mid = x - maxnum
    mid_exp = torch.exp(mid)
    sum_line = mid_exp.sum(dim=dim, keepdim=True)
    ans = mid_exp / sum_line
    return ans