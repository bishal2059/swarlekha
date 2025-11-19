import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class GPT2LikeEncoder(nn.Module):
    """Custom wrapper around GPT-2 to predict VQ codes given tokens + conditioning embeddings.
    We use Hugging Face's GPT2Model but keep the integration points so you can replace
    with a pure-from-scratch transformer if you want.
    """
    def __init__(self, config: GPT2Config, conditioning_dim=1024):
        super().__init__()
        self.gpt = GPT2Model(config)
        # projection heads to predict codebook logits
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size if hasattr(config,'vocab_size') else 8192)
        # optional cross-attention: we'll concatenate conditioning embeddings as additional tokens
        self.conditioning_dim = conditioning_dim


    def forward(self, input_ids, attention_mask=None, conditioning_embeds=None):
    # if conditioning_embeds is provided, we prepend them as embeddings (simple method)
        if conditioning_embeds is not None:
            # conditioning_embeds: (B, C, D) -> we flatten to tokens
            B, C, D = conditioning_embeds.shape
            cond_flat = conditioning_embeds.view(B, C*D)
            # naive approach: project to embedding dim and prepend
            cond_proj = cond_flat.unsqueeze(1) # (B,1,C*D)
            # NOTE: better way: use Perceiver resampler -> see conditioning_encoder
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        logits = self.to_logits(last_hidden)
        return logits