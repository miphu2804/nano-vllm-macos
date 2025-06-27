import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int = 0, event=None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = True  # Always use eager on macOS
        self.rank = 0  # Always single device on macOS
        
        # macOS device detection
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        # Device-specific cleanup
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def warmup_model(self):
        # Device-specific cache cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Use a smaller warmup sequence to avoid memory issues
        # especially on MPS where large causal masks can cause problems
        warmup_len = min(512, self.config.max_model_len)  # Use max 512 tokens for warmup
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, warmup_len
        num_seqs = min(max_num_batched_tokens // warmup_len, self.config.max_num_seqs, 1)  # At most 1 sequence
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        
        # Device-specific cache cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        
        # Device-specific memory allocation for macOS
        if torch.backends.mps.is_available():
            # MPS doesn't have memory info APIs - use conservative defaults
            total = 8 * 1024**3  # Assume 8GB available
            used = 0
            peak = 0
            current = 0
            free = total
        else:
            # CPU - use conservative defaults
            total = 8 * 1024**3  # Assume 8GB available
            free = total // 2
            used = total // 2
            peak = 0
            current = 0
            
        num_kv_heads = hf_config.num_key_value_heads  # No world_size division on macOS
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        
        # Ensure we have at least 1 block
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = 1
            
        # Create kv_cache on the correct device
        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, hf_config.head_dim,
            device=self.device
        )
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        
        # Create tensors for macOS devices
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        # Create tensors for macOS devices
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        # Create tensors for macOS devices
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
            
        # Create tensors for macOS devices
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=self.device)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Always use eager execution on macOS
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # Reset attention cache at the start of each new sequence (prefill)
        if is_prefill:
            self.reset_attention_cache()
            
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    def reset_attention_cache(self):
        """Reset attention cache for all layers - call between different sequences"""
        for module in self.model.modules():
            if hasattr(module, 'reset_cache'):
                module.reset_cache()
