import torch
import logging
import torch.nn as nn
from contextlib import nullcontext
from tqdm.auto import tqdm
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from torch import Tensor
from typing import Optional
from transformers import AutoTokenizer
from pathlib import Path
from .modules import TransformerBlock, precompute_freqs_cis, find_multiple, KVCache, RMSNorm
from .lora import setup_lora
from .model_config import BaseModelArgs
from .quantize import WeightOnlyInt4QuantHandler, WeightOnlyInt8QuantHandler


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    SEMANTIC_TOKEN = "<|semantic|>"
    IM_START_TOKEN = "<|im_start|>"
    IM_END_TOKEN = "<|im_end|>"
    CODEBOOK_PAD_TOKEN_ID = 0

    def __init__(
            self, config: BaseModelArgs, tokenizer: AutoTokenizer, init_weights: bool = True
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.semantic_token_id = tokenizer.convert_tokens_to_ids(self.SEMANTIC_TOKEN)

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(
                dim=config.dim,
                n_head=config.n_head,
                n_local_heads=config.n_local_heads,
                head_dim=config.head_dim,
                dropout=config.dropout,
                attention_qkv_bias=config.attention_qkv_bias,
                intermediate_size=config.intermediate_size,
                norm_eps=config.norm_eps,
                use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        if init_weights:
            self.apply(self._init_weights)

    def setup_caches(
            self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def embed(self, x: Tensor) -> Tensor:
        vocab_embeds = [self.embeddings(x[:, 0])]
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(x[:, i + 1] + i * self.config.codebook_size)
            emb[x[:, 0] != self.semantic_token_id] = 0
            vocab_embeds.append(emb)

        x = torch.stack(vocab_embeds, dim=3)
        x = x.sum(dim=3)

        return x

    def forward(
            self,
            inp: Tensor,
            key_padding_mask: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(2)

        # Here we want to merge the embeddings of the codebooks
        x = self.embed(inp)

        freqs_cis = self.freqs_cis[:seq_len]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
            self,
            x: Tensor,
            input_pos: Optional[Tensor] = None,
            return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile
        assert (
                self.max_seq_len != -1 and self.max_batch_size != -1
        ), "Please call setup_caches before forward_generate"

        x = self.embed(x)

        mask = self.causal_mask[
               None, None, input_pos, : self.max_seq_len
               ]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def decode_one_token(
            self,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            previous_tokens: torch.Tensor = None,
            **sampling_kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode_n_tokens(
            self,
            cur_token: torch.Tensor,
            input_pos: torch.Tensor,
            num_new_tokens: int,
            im_end_id: int = 4,
            **sampling_kwargs,
    ):
        previous_tokens = torch.zeros(
            (self.config.num_codebooks + 1, self.config.max_seq_len),
            dtype=torch.int,
            device=cur_token.device,
        )

        for i in tqdm(range(num_new_tokens)):
            # We need to get windowed repeat penalty
            win_size = 16
            if i < win_size:
                window = previous_tokens[:, :win_size]
            else:
                window = previous_tokens[:, i - win_size: i]

            with (
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                )
                if torch.cuda.is_available()
                else nullcontext()
            ):  # Actually better for Inductor to codegen attention here
                next_token = self.decode_one_token(
                    x=cur_token,
                    input_pos=input_pos,
                    previous_tokens=window,
                    **sampling_kwargs,
                )

            input_pos += 1
            cur_token = next_token.view(1, self.config.num_codebooks + 1, -1)
            previous_tokens[:, i: i + 1] = next_token.view(
                self.config.num_codebooks + 1, -1
            )

            if cur_token[0, 0, -1] == im_end_id:
                break

        return previous_tokens[:, : i + 1]

    @torch.no_grad()
    @torch.inference_mode()
    def generate(
            self,
            prompt: torch.Tensor,
            max_new_tokens: int,
            **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        im_end_id = self.tokenizer.convert_tokens_to_ids(self.IM_END_TOKEN)
        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(1)

        if max_new_tokens:
            if T + max_new_tokens > self.config.max_seq_len:
                max_new_tokens = self.config.max_seq_len - T
                logging.info(f"Truncating max_new_tokens to {max_new_tokens}")

            T_new = T + max_new_tokens
        else:
            T_new = self.config.max_seq_len
            max_new_tokens = T_new - T

        device, dtype = prompt.device, prompt.dtype

        codebook_dim = 1 + self.config.num_codebooks
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(
            (codebook_dim, self.config.max_seq_len), dtype=dtype, device=device
        )
        empty[:, :T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        # Use non-accelerated version for now, to avoid compilation overhead

        next_token = self.decode_one_token(
            prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
        )
        seq[:, T: T + 1] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        x = self.decode_n_tokens(
            next_token.view(1, codebook_dim, -1),
            input_pos,
            max_new_tokens - 1,
            im_end_id=im_end_id,
            **sampling_kwargs,
        )
        # x = torch.cat(generated_tokens, dim=1)
        seq = seq[:, : T + 1 + x.size(1)]
        seq[:, T + 1:] = x

        return seq

    @property
    def device(self):
        return list(self.parameters())[0].device

    def encode_tokens(
            self,
            string,
            prompt_tokens=None,
    ):

        string = f"{self.IM_START_TOKEN}user\n{string}{self.IM_END_TOKEN}{self.IM_START_TOKEN}assistant\n"

        new_tokens = self.tokenizer.encode(
            string,
            add_special_tokens=False,
            max_length=10 ** 6,
            truncation=False,
        )
        tokens = torch.tensor([new_tokens], dtype=torch.int, device=self.device)

        # Codebooks
        zeros = (
                torch.ones((self.config.num_codebooks, tokens.size(1)), dtype=torch.int, device=self.device)
                * self.CODEBOOK_PAD_TOKEN_ID
        )
        prompt = torch.cat((tokens, zeros), dim=0)  # [1+num_codebooks, seq_len_1]

        if prompt_tokens is None:
            return prompt

        # Get prompt tokens
        if prompt_tokens.ndim == 3:
            assert (
                    prompt_tokens.shape[0] == 1
            ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2
        data = prompt_tokens + 1

        if prompt_tokens.shape[0] > self.config.num_codebooks:
            logging.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {self.config.num_codebooks}, getting first {self.config.num_codebooks} codebooks"
            )
            data = data[:self.config.num_codebooks]

        # Add pad token for each codebook
        data = torch.cat(
            (data, torch.zeros((data.size(0), 1), dtype=torch.int, device=self.device)),
            dim=1,
        )  # [num_codebooks, seq_len_2 + 1]

        # Since 1.0, we use <|semantic|>
        s0_token_id = self.tokenizer.convert_tokens_to_ids(self.SEMANTIC_TOKEN)
        end_token_id = self.tokenizer.convert_tokens_to_ids(self.IM_END_TOKEN)
        main_token_ids = (
                torch.ones((1, data.size(1)), dtype=torch.int, device=self.device) * s0_token_id
        )  # [1, seq_len_2 + 1]
        main_token_ids[0, -1] = end_token_id

        data = torch.cat((main_token_ids, data), dim=0)  # [1 + num_codebooks, seq_len_2 + 1]
        prompt = torch.cat((prompt, data), dim=1)  # [1 + num_codebooks, seq_len_1 + seq_len_2 + 1]

        return prompt

    @classmethod
    def build_from_cfg(cls, cfg):

        model_cfg = cfg.model

        config = BaseModelArgs(
            vocab_size=model_cfg.vocab_size,
            n_layer=model_cfg.n_layer,
            n_head=model_cfg.n_head,
            dim=model_cfg.dim,
            intermediate_size=model_cfg.intermediate_size,
            n_local_heads=model_cfg.n_local_heads,
            head_dim=model_cfg.head_dim,
            rope_base=model_cfg.rope_base,
            norm_eps=model_cfg.norm_eps,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
            tie_word_embeddings=model_cfg.tie_word_embeddings,
            attention_qkv_bias=model_cfg.attention_qkv_bias,
            codebook_size=model_cfg.codebook_size,
            num_codebooks=model_cfg.num_codebooks,
            use_gradient_checkpointing=model_cfg.use_gradient_checkpointing,
            initializer_range=model_cfg.initializer_range,
            n_fast_layer=model_cfg.get("n_fast_layer", 4),
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        model = cls(config, tokenizer=tokenizer)

        lora_config = cfg.get("lora", None)

        if lora_config is not None:
            setup_lora(model, lora_config)
            logging.info(f"LoRA setup: {lora_config}")
        load_weights = cfg.get("load_weights", False)

        if load_weights is False:
            logging.info("Randomly initialized model")
            return model

        path = cfg.ckpt

        if "int8" in str(Path(path)):
            logging.info("Using int8 weight-only quantization!")

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        if "int4" in str(Path(path)):
            logging.info("Using int4 quantization!")
            path_comps = path.name.split("-")
            assert path_comps[-2].startswith("g")
            groupsize = int(path_comps[-2][1:])

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        weights = torch.load(Path(path), map_location="cpu", mmap=True)

        if "state_dict" in weights:
            logging.warning(
                "Using a TextToSemantic LightningModule checkpoint, "
                "please make sure it is a full model, not a LoRA model."
            )
            weights = weights["state_dict"]

        if next(iter(weights.keys())).startswith("model."):
            logging.info(
                f"Remove prefix 'model.' created by TextToSemantic LightningModule from keys"
            )
            new_weights = OrderedDict()
            for k, v in weights.items():
                new_weights[k.replace("model.", "")] = v
            weights = new_weights

        # Verify the name and shape of parameters since strict=False in load_state_dict.
        for k, v in model.named_parameters():
            if k not in weights:
                logging.warning(f"No weight for {k}")
            elif v.shape != weights[k].shape:
                logging.warning(
                    f"Shape mismatch for {k}: {v.shape} vs {weights[k].shape}"
                )

        err = model.load_state_dict(weights, strict=False, assign=True)
        logging.info(f"Loaded weights with error: {err}")

        return model
