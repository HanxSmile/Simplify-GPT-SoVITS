import torch.nn as nn
import torch
import os
import torchaudio
import onnxruntime
import whisper
import numpy as np
from typing import Tuple, List
from functools import partial
from gpt_sovits.text.TextProcessor import TextProcessor
from gpt_sovits.common.registry import registry
import torchaudio.compliance.kaldi as kaldi
from gpt_sovits.model.cosyvoice_modules import CosyVoiceModel
from whisper.tokenizer import get_tokenizer
from .cosyvoice_modules.utils.audio import mel_spectrogram


class CosyVoiceBase(nn.Module):
    SAMPLE_RATE = 22050

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
            speed=1.0,
            *args,
            **kwargs
    ):

        text = inputs["text"]

        if not self.prompt_registered:
            self.register_prompt(inputs)

        text_lst = self.text_processor.segment_text(text)
        text_lst = [self.text_processor.normalize_text(_) for _ in text_lst]
        results = []

        for sub_text in text_lst:
            y = self.inference(sub_text, speed=speed)
            results.append(y.squeeze(0).float())

        return self.audio_postprocess(
            results,
            self.SAMPLE_RATE,
            fragment_interval
        )

    @classmethod
    def build_from_cfg(cls, cfg):
        is_instruct = "-Instruct" in cfg.ckpt
        sampling_rate = cfg.sampling_rate
        tokenizer = get_tokenizer(
            multilingual=cfg.tokenizer.multilingual,
            num_languages=cfg.tokenizer.num_languages,
            language=cfg.tokenizer.language,
            task=cfg.tokenizer.task
        )
        allow_special = cfg.allow_special
        feature_extractor = partial(
            mel_spectrogram,
            n_fft=cfg.feature_extractor.n_fft,
            num_mels=cfg.feature_extractor.num_mels,
            sampling_rate=sampling_rate,
            hop_size=cfg.feature_extractor.hop_size,
            win_size=cfg.feature_extractor.win_size,
            fmin=cfg.feature_extractor.fmin,
            fmax=cfg.feature_extractor.fmax,
            center=cfg.feature_extractor.center,
        )

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        campplus_model = os.path.join(cfg.ckpt, "campplus.onnx")
        campplus_session = onnxruntime.InferenceSession(
            campplus_model,
            sess_options=option,
            providers=["CPUExecutionProvider"])
        speech_tokenizer_model = os.path.join(cfg.ckpt, "speech_tokenizer_v1.onnx")
        speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                "CUDAExecutionProvider" if torch.cuda.is_available() else
                "CPUExecutionProvider"])
        spk2info = os.path.join(cfg.ckpt, "spk2info.pt")
        if os.path.exists(spk2info):
            spk2info = torch.load(spk2info, map_location=torch.device("cpu"))
        else:
            spk2info = {}
        cfg.cosyvoice_model.sampling_rate = sampling_rate
        cfg.cosyvoice_model.ckpt = cfg.ckpt
        cosyvoice_model = CosyVoiceModel.build_from_cfg(cfg.cosyvoice_model)

        cut_method = cfg.get("cut_method", "cut5")
        text_converter_cfg = cfg.text_converter

        return cls(
            is_instruct=is_instruct,
            cosyvoice_model=cosyvoice_model,
            tokenizer=tokenizer,
            allowed_special=allow_special,
            speech_tokenizer_session=speech_tokenizer_session,
            feat_extractor=feature_extractor,
            campplus_session=campplus_session,
            text_converter_cfg=text_converter_cfg,
            cut_method=cut_method,
            spk2info=spk2info,
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

    def frontend_vc(self, inputs):
        prompt_audio_path = inputs["prompt_audio"]
        audio, sr = torchaudio.load(str(prompt_audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        prompt_speech_16k = torchaudio.functional.resample(
            audio, sr, 16_000
        )
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        embedding = self._extract_spk_embedding(prompt_speech_16k)

        model_input = {
            'flow_prompt_speech_token': prompt_speech_token,
            'flow_prompt_speech_token_len': prompt_speech_token_len,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_speech_feat_len': prompt_speech_feat_len,
            'flow_embedding': embedding
        }
        self.prompt_buffer.update(model_input)
        self.prompt_registered = True

    def inference(self, source_speech_16k, speed=1.0):
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_inputs = {
            'source_speech_token': source_speech_token,
            'source_speech_token_len': source_speech_token_len,
        }
        model_inputs.update(self.prompt_buffer)
        return self.cosyvoice_model.vc(**model_inputs, speed=speed)

    def generate(self, inputs, speed=1.0, *args, **kwargs):
        audio_path = inputs["audio"]
        audio, sr = torchaudio.load(str(audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        speech_16k = torchaudio.functional.resample(
            audio, sr, 16_000
        )

        if not self.prompt_registered:
            self.register_prompt(inputs)

        result = self.inference(speech_16k, speed=speed).squeeze(0).float()

        return self.audio_postprocess(
            [result],
            self.SAMPLE_RATE,
            0.1,
        )
