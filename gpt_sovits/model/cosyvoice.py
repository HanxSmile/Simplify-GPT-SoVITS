import torch.nn as nn
import torch
import torchaudio
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, List
from gpt_sovits.text.TextProcessor import TextProcessor
from gpt_sovits.common.registry import registry
import torchaudio.compliance.kaldi as kaldi
from whisper.tokenizer import get_tokenizer
import whisper


class CosyVoiceBase(nn.Module):
    def __init__(
            self,
            is_instruct,
            cosyvoice_model,
            tokenizer,
            allowed_special,
            speech_tokenizer_session,
            feat_extractor,
            campplus_session,
            text_converter_cfg,
            cut_method,
            spk2info=None,
    ):
        super(CosyVoiceBase, self).__init__()
        self.is_instruct = is_instruct
        self.cosyvoice_model = cosyvoice_model
        self.tokenizer = tokenizer
        self.allowed_special = allowed_special
        self.speech_tokenizer_session = speech_tokenizer_session
        self.campplus_session = campplus_session
        self.feat_extractor = feat_extractor
        self.text_converter = self._init_text_converter(text_converter_cfg)
        self.text_processor = TextProcessor(
            self.text_converter,
            None,
            None,
            cut_method=cut_method
        )
        self.prompt_registered = False
        self.prompt_buffer = dict()
        self.spk2info = spk2info or dict()

    def _init_text_converter(self, cfg):
        converter_name = cfg.converter_cls
        converter_cls = registry.get_converter_class(converter_name)
        return converter_cls.build_from_cfg(cfg)

    def _extract_text_token(self, text):
        text = self.text_processor.normalize_text(text)
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(
            None,
            {self.speech_tokenizer_session.get_inputs()[0].name:
                 feat.detach().cpu().numpy(),
             self.speech_tokenizer_session.get_inputs()[1].name:
                 np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(
            speech,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(
            None,
            {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(
                dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def list_avaliable_spks(self):
        spks = list(self.spk2info.keys())
        return spks

    def inference(self, tts_text, speed=1.0):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        model_inputs = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
        }
        model_inputs.update(self.prompt_buffer)
        return self.cosyvoice_model.tts(**model_inputs, speed=speed)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def audio_postprocess(
            self,
            audio: List[torch.Tensor],
            sr: int,
            fragment_interval: float = 0.3
    ) -> Tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
            int(sr * fragment_interval),
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
            fragment_interval=0.3,
            *args,
            **kwargs
    ):

        text, audio_prompt, text_prompt = inputs["text"], inputs["audio_prompt"], inputs["text_prompt"]

        if not self.prompt_registered:
            self.register_prompt(inputs)

        audio_prompt = self.prompt_buffer["audio_prompt"]
        text_lst = self.text_processor.segment_text(text)
        text_lst = [self.text_processor.normalize_text(_) for _ in text_lst]
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
            )
            results.append(fake_audio[0, 0].float())

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


@registry.register_model("cosyvoice_zero_shot")
class CosyvoiceZeroShot(CosyVoiceBase):

    def register_prompt(self, inputs):
        prompt_text, prompt_audio_path = inputs["prompt_text"], inputs["prompt_audio"]
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        audio, sr = torchaudio.load(str(prompt_audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        prompt_speech_16k = torchaudio.functional.resample(
            audio, sr, 16_000
        )
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(audio)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)

        self.prompt_buffer.update({
            'prompt_text': prompt_text_token,
            'prompt_text_len': prompt_text_token_len,
            'llm_prompt_speech_token': speech_token,
            'llm_prompt_speech_token_len': speech_token_len,
            'flow_prompt_speech_token': speech_token,
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speech_feat,
            'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding
        })
        self.prompt_registered = True


@registry.register_model("cosyvoice_sft")
class CosyVoiceSFT(CosyVoiceBase):

    def register_prompt(self, inputs):
        spk_id = inputs["spk_id"]
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {
            'llm_embedding': embedding,
            'flow_embedding': embedding
        }
        self.prompt_buffer.update(model_input)
        self.prompt_registered = True


@registry.register_model("cosyvoice_cross_lingual")
class CosyVoiceCrossLingual(CosyVoiceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_instruct is False, 'Do not support cross_lingual inference'

    def register_prompt(self, inputs):
        prompt_audio_path = inputs["prompt_audio"]
        audio, sr = torchaudio.load(str(prompt_audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        prompt_speech_16k = torchaudio.functional.resample(
            audio, sr, 16_000
        )
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(audio)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)

        self.prompt_buffer.update({
            'flow_prompt_speech_token': speech_token,
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speech_feat,
            'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding
        })
        self.prompt_registered = True


@registry.register_model("cosyvoice_instruct")
class CosyVoiceInstruct(CosyVoiceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_instruct is True, "Do not support instruct inference"

    def register_prompt(self, inputs):
        spk_id, instruct_text = inputs["spk_id"], inputs["instruction"]
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {
            'flow_embedding': embedding
        }
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        self.prompt_buffer.update(model_input)
        self.prompt_registered = True


@registry.register_model("cosyvoice_vc")
class CosyVoiceVC(CosyVoiceBase):

    def frontend_vc(self, source_speech_16k, prompt_speech_16k):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {
            'source_speech_token': source_speech_token,
            'source_speech_token_len': source_speech_token_len,
            'flow_prompt_speech_token': prompt_speech_token,
            'flow_prompt_speech_token_len': prompt_speech_token_len,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_speech_feat_len': prompt_speech_feat_len,
            'flow_embedding': embedding
        }
        return model_input

    def inference_vc(self, source_speech_16k, prompt_speech_16k, speed=1.0):
        model_input = self.frontend_vc(source_speech_16k, prompt_speech_16k)
        return self.cosyvoice_model.vc(**model_input, speed=speed)
