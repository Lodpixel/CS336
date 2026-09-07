import torch

class Embedding(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        size = (num_embeddings, embedding_dim)
        self.weight = torch.nn.Parameter(torch.empty(size, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)
    
    def forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]
