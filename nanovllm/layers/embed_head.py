import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        
        # macOS single device - no tensor parallelism
        self.tp_rank = 0
        self.tp_size = 1
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings
        self.vocab_start_idx = 0
        self.vocab_end_idx = num_embeddings
            
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Single device - load full weight
        assert param.data.size() == loaded_weight.size(), f"Size mismatch: {param.data.size()} vs {loaded_weight.size()}"
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # Single device - use normal embedding
        y = F.embedding(x, self.weight)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        
        # Single device - no gather needed
        return logits
