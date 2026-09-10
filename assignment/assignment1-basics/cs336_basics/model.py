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

def scaled_dot_product_attention(
    # Q: (..., queries, d_k), K: (..., keys d_k), V: (..., keys, d_v)
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    d_k = Q.shape[-1]
    mid = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = mid / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = softmax(scores, -1)
    final = einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return final

class multihead_self_attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        # 有四个 Linear 模块，W_Q, W_K, W_V, W_O(d_model * d_model)
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.device = device
        self.h = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def multihead(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        q_arr = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.h)
        k_arr = rearrange(K, "... seq (h d) -> ... h seq d", h=self.h)
        v_arr = rearrange(V, "... seq (h d) -> ... h seq d", h=self.h)
        if token_positions is not None:
            rope_module = RoPE(self.theta, self.d_model // self.h, self.max_seq_len, self.device)
            q_arr = rope_module.forward(q_arr, token_positions)
            k_arr = rope_module.forward(k_arr, token_positions)
        # 利用广播来简化 mask 的编写
        casual_mask = torch.tril(torch.ones(q_arr.shape[-2], k_arr.shape[-2], dtype=torch.bool))
        ans_arr = scaled_dot_product_attention(q_arr, k_arr, v_arr, casual_mask)
        ans_arr = rearrange(ans_arr, "... h seq d -> ... seq (h d)")
        return ans_arr

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        data = self.multihead(q, k, v, token_positions)
        return self.output_proj.forward(data)

class transformer_block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.ln1 = RMSnorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSnorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
    ):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        # first layer: y = x + MHA(RMSNorm(x))
        y = x + self.attn.forward(self.ln1.forward(x), token_positions)
        # second layer: y = y + FFN(RMSNorm(y))
        y = y + self.ffn.forward(self.ln2.forward(y))
        return y
        
class transformer_lm(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # num_layers 个 transformer layer
        self.layers = torch.nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, rope_theta, context_length, device, dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.token_embeddings.forward(x)
        for block in self.layers:
            x = block.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x

    
        

        
