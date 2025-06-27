import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        
        # Handle different temperature values
        if temperatures.numel() == 1:
            # Single temperature for all sequences
            temp = temperatures.item()
            if temp == 0.0:
                return greedy_tokens
            else:
                # Apply temperature scaling
                scaled_logits = logits / temp
                probs = torch.softmax(scaled_logits, dim=-1, dtype=torch.float)
                # Use standard multinomial sampling instead of Gumbel
                sample_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).squeeze(-1)
                return sample_tokens.view(greedy_tokens.shape)
        else:
            # Multiple temperatures - handle each sequence separately
            batch_size = logits.size(0)
            vocab_size = logits.size(-1)
            sample_tokens = torch.zeros_like(greedy_tokens)
            
            for i in range(batch_size):
                temp = temperatures[i].item()
                if temp == 0.0:
                    sample_tokens[i] = greedy_tokens[i]
                else:
                    # Apply temperature scaling
                    scaled_logits = logits[i] / temp
                    probs = torch.softmax(scaled_logits, dim=-1, dtype=torch.float)
                    sample_tokens[i] = torch.multinomial(probs, num_samples=1).squeeze()
            
            return sample_tokens
