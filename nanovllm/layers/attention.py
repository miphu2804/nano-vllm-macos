import torch
from torch import nn
from nanovllm.utils.context import get_context

# Optional imports for optimized attention
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        # Initialize proper KV cache
        self.k_cache = []  # List to store cached keys for each position
        self.v_cache = []  # List to store cached values for each position
        self.cache_initialized = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Get batch size and reshape to proper attention dimensions
        batch_size = q.size(0)
        
        # Reshape from [batch_size, hidden_size] to [batch_size, num_heads, head_dim]
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        context = get_context()
        
        # Use standard attention for all cases (simpler and more reliable)
        o = self._standard_attention(q, k, v, context)
            
        return o.reshape(batch_size, self.num_heads * self.head_dim)

    def _standard_attention(self, q, k, v, context):
        """Standard scaled dot-product attention with proper KV caching"""
        # Expand k and v to match number of heads if using grouped query attention
        if self.num_kv_heads != self.num_heads:
            expand_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(expand_factor, dim=1)
            v = v.repeat_interleave(expand_factor, dim=1)
        
        if context.is_prefill:
            # During prefill, we process the full sequence
            return self._prefill_attention(q, k, v, context)
        else:
            # During decode, we append to cache and use full history
            return self._decode_attention(q, k, v, context)

    def _prefill_attention(self, q, k, v, context):
        """Attention for prefill phase - processes full input sequence"""
        seq_len, num_heads, head_dim = q.shape
        
        # Initialize cache with the full sequence
        self.k_cache = [k[i] for i in range(seq_len)]  # List of tensors, one per position
        self.v_cache = [v[i] for i in range(seq_len)]
        self.cache_initialized = True
        
        # Compute attention scores using einsum for clarity
        # q: [seq_len, num_heads, head_dim]
        # k: [seq_len, num_heads, head_dim]  
        # We want: [seq_len, num_heads, seq_len] (each query attends to all keys)
        attn_scores = torch.einsum('ihd,jhd->ihj', q, k) * self.scale
        
        # Apply causal mask for autoregressive generation
        if seq_len > 1:
            # Create causal mask: [seq_len, seq_len]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            # Apply mask using broadcasting: mask shape [seq_len, seq_len] broadcasts to [seq_len, num_heads, seq_len]
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute output: attn_weights @ v
        # attn_weights: [seq_len, num_heads, seq_len]
        # v: [seq_len, num_heads, head_dim]
        # We want: [seq_len, num_heads, head_dim]
        o = torch.einsum('ihj,jhd->ihd', attn_weights, v)
        
        return o

    def _decode_attention(self, q, k, v, context):
        """Attention for decode phase - uses KV cache for efficiency"""
        if not self.cache_initialized:
            # If cache not initialized, treat as first token
            self.k_cache = [k[0]]  # Only one token
            self.v_cache = [v[0]]  # Only one token
            self.cache_initialized = True
        else:
            # Append new token to cache
            self.k_cache.append(k[0])  # k[0] is the new token
            self.v_cache.append(v[0])  # v[0] is the new token
        
        # Stack cached keys and values to create full sequence
        k_full = torch.stack(self.k_cache, dim=0)  # [seq_len, num_heads, head_dim]
        v_full = torch.stack(self.v_cache, dim=0)  # [seq_len, num_heads, head_dim]
        
        # q is for the current (last) token only: [1, num_heads, head_dim]
        # Compute attention with full history using einsum
        # q: [1, num_heads, head_dim], k_full: [seq_len, num_heads, head_dim]
        # Result: [1, num_heads, seq_len]
        attn_scores = torch.einsum('ihd,jhd->ihj', q, k_full) * self.scale
        
        # No causal mask needed in decode phase - current token can attend to all previous tokens
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute output: [1, num_heads, seq_len] @ [seq_len, num_heads, head_dim] -> [1, num_heads, head_dim]
        o = torch.einsum('ihj,jhd->ihd', attn_weights, v_full)
        
        return o

    def reset_cache(self):
        """Reset the KV cache - call this between different sequences"""
        self.k_cache = []
        self.v_cache = []
        self.cache_initialized = False