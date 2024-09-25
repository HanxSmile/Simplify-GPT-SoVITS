import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from transformers import AutoTokenizer
from .base_transformer import BaseTransformer, TransformerForwardResult
from .modules import TransformerBlock, RMSNorm, KVCache, sample
from .model_config import BaseModelArgs
from gpt_sovits.common.registry import registry


@registry.register_model("dual_ar_transformer")
class DualARTransformer(BaseTransformer):
    def __init__(self, config: BaseModelArgs, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.dim)

        # The equivalent bs is so large that sdpa doesn't work
        self.fast_layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=False) for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.dim,
            config.codebook_size,
            bias=False,
        )

        self.apply(self._init_weights)

    def setup_caches(
            self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        head_dim = self.config.dim // self.config.n_head

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def forward(
            self,
            inp: Tensor,
            key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        parent_result = super().forward(inp, key_padding_mask)
        token_logits = parent_result.logits
        x = parent_result.hidden_states

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
                    None, None, :fast_seq_len, :fast_seq_len
                    ]  # (B, N, Q, K)
        fast_freqs_cis = self.freqs_cis[:fast_seq_len]

        # Drop the last token and rotate left
        codebooks = inp[:, 1:-1, 1:]
        codebooks = F.pad(codebooks, (0, 1), value=0)
        codebook_embeddings = self.fast_embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)
        b, s = x.size(0), x.size(2)
        x = rearrange(x, "b n s d -> (b s) n d")  # flatten the batch and seq_len

        # Remove padded part
        codebooks = rearrange(codebooks, "b n s -> (b s) n")
        codebook_mask = (codebooks == 0).all(dim=-1)

        if torch.all(codebook_mask):
            # If all codebooks are padded, we keep first 8 to make sure the model runs
            codebook_mask[:8] = False

        x_bs, x_len = x.size(0), x.size(1)
        x = x[~codebook_mask]

        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, fast_freqs_cis, fast_mask, use_reentrant=True)
            else:
                x = layer(x, fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        # Re-pad the codebook_logits
        buffer = torch.zeros(
            x_bs,
            x_len,
            codebook_logits.size(-1),
            device=codebook_logits.device,
            dtype=codebook_logits.dtype,
        )
        buffer[~codebook_mask] = codebook_logits
        codebook_logits = buffer

        assert codebook_logits.shape[1] == self.config.num_codebooks
        codebook_logits = rearrange(
            codebook_logits,
            "(b s) n d -> b s n d",
            b=b,
            s=s,
            n=self.config.num_codebooks,
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward_generate_fast(
            self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(1, 1, -1)

        fast_mask = self.causal_mask[
                    None, None, input_pos, : self.config.num_codebooks
                    ]  # (B, N, Q, K)
        fast_freqs_cis = self.freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits

    def decode_one_token(
            self,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            previous_tokens: torch.Tensor = None,
            **sampling_kwargs,
    ) -> torch.Tensor:
        x = self.forward_generate(x, input_pos)

        sampling_kwargs_main = sampling_kwargs.copy()
        sampling_kwargs_main["temperature"] = 0.1
        sampling_kwargs_main["top_p"] = 0.1
        sampling_kwargs_main["repetition_penalty"] = 1.0

        codebooks = [
            sample(
                x.logits,
                previous_tokens=None,  # Disable repetition penalty for the token codebook
                **sampling_kwargs_main,
            )[0]
        ]

        x = x.hidden_states

        # Cleanup the cache
        for layer in self.fast_layers:
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

        for codebook_idx in range(self.config.num_codebooks):
            input_pos = torch.tensor([codebook_idx], device=x.device, dtype=torch.long)
            logits = self.forward_generate_fast(x, input_pos)
            a = sample(
                logits,
                previous_tokens=(
                    previous_tokens[codebook_idx + 1]
                    if previous_tokens is not None
                    else None
                ),
                **sampling_kwargs,
            )[0]
            x = self.fast_embeddings(a)
            codebooks.append(a)

        return torch.stack(codebooks, dim=0)
