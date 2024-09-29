import torch.nn as nn
import torch
import torchaudio
import logging
import numpy as np
from typing import Tuple, List
from gpt_sovits.text.TextProcessor import TextProcessor
from gpt_sovits.common.registry import registry


@registry.register_model("fish_speech")
class FishSpeech(nn.Module):
    def __init__(
            self,
            vqgan_model,
            text2semantic_model,
            text_converter_cfg,
            cut_method
    ):
        super(FishSpeech, self).__init__()
        self.vqgan_model = vqgan_model
        self.text2semantic_model = text2semantic_model
        self.text_converter = self._init_text_converter(text_converter_cfg)
        self.text_processor = TextProcessor(
            self.text_converter,
            None,
            None,
            cut_method=cut_method
        )
        self.prompt_registered = False
        self.prompt_buffer = dict()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _init_text_converter(self, cfg):
        converter_name = cfg.converter_cls
        converter_cls = registry.get_converter_class(converter_name)
        return converter_cls.build_from_cfg(cfg)

    def register_prompt(self, inputs):
        prompt_text, prompt_audio_path = inputs["prompt_text"], inputs["prompt_audio"]
        prompt_tokens = self._get_prompt_semantic(prompt_audio_path)
        prompt_text = self.text_converter.normalize(prompt_text)
        prompt_tokens = self.text2semantic_model.encode_tokens(prompt_text, prompt_tokens)
        self.prompt_buffer["audio_prompt"] = prompt_tokens
        self.prompt_registered = True

        self.text2semantic_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.text2semantic_model.config.max_seq_len,
            dtype=next(self.text2semantic_model.parameters()).dtype,
        )

    @torch.no_grad()
    def _get_prompt_semantic(self, ref_wav_path: str):
        audio, sr = torchaudio.load(str(ref_wav_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, sr, self.vqgan_model.spec_transform.sample_rate
        )

        audios = audio[None].to(self.device)
        logging.info(
            f"Loaded audio with {audios.shape[2] / self.vqgan_model.spec_transform.sample_rate:.2f} seconds"
        )
        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        indices = self.vqgan_model.encode(audios, audio_lengths)[0][0]

        indices = indices.to(self.device).long()
        logging.info(f"Generated indices of shape {indices.shape}")

        return indices

    def audio_postprocess(
            self,
            audio: List[torch.Tensor],
            sr: int,
            fragment_interval: float = 0.3
    ) -> Tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
            int(self.generate_cfg.sampling_rate * fragment_interval),
        ).to(self.device)

        for i, audio_fragment in enumerate(audio):
            max_audio = torch.abs(audio_fragment).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio_fragment /= max_audio
            audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
            audio[i] = audio_fragment.cpu().numpy()

        audio = np.concatenate(audio, 0)
        audio = (audio * 32768).astype(np.int16)

        return sr, audio

    @torch.no_grad()
    def generate(
            self,
            inputs,
            top_p=1,
            temperature=1,
            repetition_penalty=1.35,
            max_new_tokens=0,
            fragment_interval=0.3,
            *args,
            **kwargs
    ):
        temperature = torch.tensor(temperature, device=self.device, dtype=torch.float)
        top_p = torch.tensor(top_p, device=self.device, dtype=torch.float)
        repetition_penalty = torch.tensor(repetition_penalty, device=self.device, dtype=torch.float)

        text = inputs["text"]

        if not self.prompt_registered:
            self.register_prompt(inputs)

        audio_prompt = self.prompt_buffer["audio_prompt"]
        text_lst = self.text_processor.segment_text(text)
        results = []

        for sub_text in text_lst:
            encoded = self.text2semantic_model.encode_tokens(sub_text)
            model_inputs = torch.cat([audio_prompt, encoded], dim=1)
            y = self.text2semantic_model.generate(
                model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            codes = y[1:, model_inputs.size(1):-1].clone()
            codes = codes - 1
            assert (codes >= 0).all(), f"Negative code found"
            code_length = torch.tensor([codes.shape[1]], device=self.device)
            fake_audio, _ = self.vqgan_model.decode(
                indices=codes[None], feature_lengths=code_length
            )[0, 0].float().cpu()
            results.append(fake_audio)

        return self.audio_postprocess(
            results,
            self.vqgan_model.spec_transform.sample_rate,
            fragment_interval
        )

    @classmethod
    def build_from_cfg(cls, cfg):

        vqgan_cfg = cfg.vqgan
        text2semantic_cfg = cfg.text2semantic

        vqgan_cls = registry.get_model_class(vqgan_cfg.model_cls)
        vqgan_model = vqgan_cls.build_from_cfg(vqgan_cfg)

        text2semantic_cls = registry.get_model_class(text2semantic_cfg.model_cls)
        text2semantic_model = text2semantic_cls.build_from_cfg(text2semantic_cfg)

        cut_method = cfg.get("cut_method", "cut5")
        text_converter_cfg = cfg.text_converter
        return cls(
            vqgan_model=vqgan_model,
            text2semantic_model=text2semantic_model,
            text_converter_cfg=text_converter_cfg,
            cut_method=cut_method,
        )
