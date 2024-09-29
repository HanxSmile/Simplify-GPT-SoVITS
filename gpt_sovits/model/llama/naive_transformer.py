import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import rearrange
from transformers import AutoTokenizer
from .modules import RMSNorm, sample
from .base_transformer import BaseTransformer, BaseTransformerForwardResult, TransformerForwardResult
from .model_config import BaseModelArgs
from gpt_sovits.common.registry import registry


@registry.register_model("ar_transformer")
class NaiveTransformer(BaseTransformer):
    def __init__(self, config: BaseModelArgs, tokenizer: AutoTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        self.codebook_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.codebook_output = nn.Linear(
            config.dim,
            config.codebook_size * config.num_codebooks,
            bias=False,
        )

        self.apply(self._init_weights)

    def decode(self, result: BaseTransformerForwardResult) -> TransformerForwardResult:
        token_logits = result.logits
        x = result.hidden_states

        # Codebook
        codebook_logits = self.codebook_output(self.codebook_norm(x))
        codebook_logits = rearrange(
            codebook_logits, "b n (c d) -> b n c d", c=self.config.num_codebooks
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward(
            self,
            inp: Tensor,
            key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        result = super().forward(
            inp=inp,
            key_padding_mask=key_padding_mask,
        )
        return self.decode(result)

    def forward_generate(
            self,
            x: Tensor,
            input_pos: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward_generate(x, input_pos)
        return self.decode(result)

    def decode_one_token(
            self,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            previous_tokens: torch.Tensor = None,
            **sampling_kwargs,
    ) -> torch.Tensor:
        x = self.forward_generate(x, input_pos)

        sampling_kwargs_main = sampling_kwargs.copy()

        codebooks = [
            sample(
                x.logits,
                previous_tokens=None,  # Disable repetition penalty for the token codebook
                **sampling_kwargs_main,
            )[0]
        ]

        for i in range(self.config.num_codebooks):
            codebooks.append(
                sample(
                    x.codebook_logits[:, :, i],
                    previous_tokens=(
                        previous_tokens[i + 1] if previous_tokens is not None else None
                    ),
                    **sampling_kwargs,
                )[0]
            )

        return torch.stack(codebooks, dim=0)
