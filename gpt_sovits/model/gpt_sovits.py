import torch.nn as nn
import torch
import librosa
import logging
import os
import traceback
import ffmpeg
import numpy as np
from typing import Tuple, List
from transformers import HubertModel, AutoModelForMaskedLM, AutoTokenizer
from gpt_sovits.model.vits import SynthesizerTrn
from gpt_sovits.model.vits.mel_processing import spectrogram_torch
from gpt_sovits.model.AR import Text2SemanticLightningModule
from gpt_sovits.text.TextProcessor import TextProcessor
from gpt_sovits.common.registry import registry


def clean_path(path_str: str):
    if path_str.endswith(('\\', '/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("音频加载失败")

    return np.frombuffer(out, np.float32).flatten()


@registry.register_model("gpt_sovits")
class GPT_SoVITS(nn.Module):
    def __init__(
            self,
            hubert_model_name,
            bert_model_name,
            vits_model_name,
            t2s_model_name,
            cut_method,
            text_converter_cfg,
            generate_cfg,
    ):
        super(GPT_SoVITS, self).__init__()
        self.generate_cfg = generate_cfg
        self.hubert_model = HubertModel.from_pretrained(hubert_model_name).eval()
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_name).eval()
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.text_converter = self._init_text_converter(text_converter_cfg)
        self.vits_model = self._init_vits_model(vits_model_name)
        self.t2s_model = self._init_t2s_model(t2s_model_name)
        self.text_processor = TextProcessor(
            self.text_converter,
            self.bert_model,
            self.bert_tokenizer,
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

    def _init_t2s_model(self, model_name):
        dict_s1 = torch.load(model_name, map_location=torch.device('cpu'))
        config = dict_s1["config"]
        self.generate_cfg.max_sec = config["data"]["max_sec"]
        self.generate_cfg.hz = 50
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.eval()
        return t2s_model

    def _init_vits_model(self, model_name):
        dict_s2 = torch.load(model_name, map_location=torch.device('cpu'))
        hps = dict_s2["config"]
        assert dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] != 322, "Only support version 2"
        filter_length = hps["data"]["filter_length"]
        segment_size = hps["train"]["segment_size"]
        hop_length = hps["data"]["hop_length"]
        n_speakers = hps["data"]["n_speakers"]

        self.generate_cfg.filter_length = filter_length
        self.generate_cfg.segment_size = segment_size
        self.generate_cfg.sampling_rate = hps["data"]["sampling_rate"]
        self.generate_cfg.hop_length = hop_length
        self.generate_cfg.win_length = hps["data"]["win_length"]
        self.generate_cfg.n_speakers = n_speakers
        self.generate_cfg.semantic_frame_rate = "25hz"

        kwargs = hps["model"]
        vits_model = SynthesizerTrn(
            filter_length // 2 + 1,
            segment_size // hop_length,
            n_speakers=n_speakers,
            **kwargs
        )
        if hasattr(vits_model, "enc_q"):
            del vits_model.enc_q
        vits_model = vits_model.eval()
        vits_model.load_state_dict(dict_s2["weight"], strict=False)
        return vits_model

    def register_prompt(self, inputs):
        prompt_text, prompt_audio_path = inputs["prompt_text"], inputs["prompt_audio"]
        ref_audio_paths = inputs.get("ref_audio", [prompt_audio_path])

        audio_prompt = self._get_prompt_semantic(prompt_audio_path)
        ref_audio_specs = [self._get_ref_spec(_) for _ in ref_audio_paths]
        _, prompt_text_phones, prompt_text_bert_features = self.text_processor.process_single(prompt_text, self.device)

        self.prompt_buffer["audio_prompt"] = audio_prompt
        self.prompt_buffer["prompt_text_phones"] = prompt_text_phones
        self.prompt_buffer["prompt_text_bert_features"] = prompt_text_bert_features
        self.prompt_buffer["ref_audio_specs"] = ref_audio_specs
        self.prompt_registered = True

    def _get_ref_spec(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(self.generate_cfg.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.generate_cfg.filter_length,
            self.generate_cfg.sampling_rate,
            self.generate_cfg.hop_length,
            self.generate_cfg.win_length,
            center=False,
        )
        return spec.to(self.device)

    @torch.no_grad()
    def _get_prompt_semantic(self, ref_wav_path: str):
        zero_wav = np.zeros(
            int(self.generate_cfg.sampling_rate * 0.3),
        )
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 10 * sr or wav16k.shape[0] < 3 * sr:
            logging.warning("参考音频在3~10秒范围外，请更换！")
        wav16k = torch.from_numpy(wav16k).float()
        zero_wav_torch = torch.from_numpy(zero_wav).float()
        wav16k = wav16k.to(self.device)
        zero_wav_torch = zero_wav_torch.to(self.device)

        wav16k = torch.cat([zero_wav_torch, wav16k, zero_wav_torch])
        hubert_feature = self.hubert_model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = self.vits_model.extract_latent(hubert_feature)

        prompt_semantic = codes[0, 0].to(self.device)
        return prompt_semantic

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
            speed=1,
            top_k=5,
            top_p=1,
            temperature=1,
            repetition_penalty=1.35,
            fragment_interval=0.3,
    ):
        text = inputs["text"]

        if not self.prompt_registered:
            prompt_text, prompt_audio_path = inputs["prompt_text"], inputs["prompt_audio"]
            ref_audio_paths = inputs.get("ref_audio", [prompt_audio_path])
            audio_prompt = self._get_prompt_semantic(prompt_audio_path)
            ref_audio_specs = [self._get_ref_spec(_) for _ in ref_audio_paths]
            _, prompt_text_phones, prompt_text_bert_features = self.text_processor.process_single(prompt_text,
                                                                                                  self.device)
        else:
            audio_prompt = self.prompt_buffer["audio_prompt"]
            ref_audio_specs = self.prompt_buffer["ref_audio_specs"]
            prompt_text_phones = self.prompt_buffer["prompt_text_phones"]
            prompt_text_bert_features = self.prompt_buffer["prompt_text_bert_features"]

        all_data = self.text_processor.process(text, self.device)
        results = []
        for item in all_data:
            phones, bert_feature = item["phones"], item["bert_feature"]
            all_bert_feature = torch.cat([prompt_text_bert_features, bert_feature], dim=1)
            all_phone_ids = torch.LongTensor(prompt_text_phones + phones).to(self.device)
            all_phone_lens = torch.tensor(all_phone_ids.shape[-1]).to(self.device)

            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phone_ids[None],
                all_phone_lens[None],
                audio_prompt[None],
                all_bert_feature[None],
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=self.generate_cfg.hz * self.generate_cfg.max_sec,
                repetition_penalty=repetition_penalty,
                max_len=max(all_bert_feature.shape[-1], all_phone_ids.shape[-1]),
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            phones = torch.LongTensor(phones).to(self.device)
            audio_fragment = (self.vits_model.decode(
                pred_semantic, phones[None], ref_audio_specs, speed=speed
            ).detach()[0, 0, :])
            results.append(audio_fragment)
        return self.audio_postprocess(
            results,
            self.generate_cfg.sampling_rate,
            fragment_interval
        )

    @classmethod
    def build_from_cfg(cls, cfg):
        hubert_model_name = cfg.hubert_model_name
        bert_model_name = cfg.bert_model_name
        vits_model_name = cfg.vits_model_name
        t2s_model_name = cfg.t2s_model_name
        cut_method = cfg.get("cut_method", "cut5")
        text_converter_cfg = cfg.text_converter
        generate_cfg = cfg.generate_cfg

        return cls(
            hubert_model_name=hubert_model_name,
            bert_model_name=bert_model_name,
            t2s_model_name=t2s_model_name,
            vits_model_name=vits_model_name,
            cut_method=cut_method,
            text_converter_cfg=text_converter_cfg,
            generate_cfg=generate_cfg,
        )
