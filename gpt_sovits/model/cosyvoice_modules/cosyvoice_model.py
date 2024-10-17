# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.nn import functional as F
import torch.nn as nn
from gpt_sovits.common.registry import registry


@registry.register_model("cosyvoice_model")
class CosyVoiceModel(nn.Module):

    def __init__(
            self,
            llm: torch.nn.Module,
            flow: torch.nn.Module,
            hift: torch.nn.Module
    ):
        super().__init__()
        self.llm = llm
        self.flow = flow
        self.hift = hift

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.llm.half()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding):
        result = self.llm.inference(
            text=text.to(self.device),
            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
            prompt_text=prompt_text.to(self.device),
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(
                self.device),
            prompt_speech_token=llm_prompt_speech_token.to(self.device),
            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]],
                                                 dtype=torch.int32).to(self.device),
            embedding=llm_embedding.to(self.device).half())
        return torch.tensor(result)

    def token2wav(
            self,
            token,
            prompt_token,
            prompt_feat,
            embedding,
            speed=1.0
    ):
        tts_mel = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                self.device),
            embedding=embedding.to(self.device))

        hift_cache_source = torch.zeros(1, 1, 0)
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
        return tts_speech

    def tts(
            self,
            text,
            flow_embedding,
            llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            speed=1.0,
            **kwargs
    ):

        tts_speech_token = self.llm_job(
            text, prompt_text, llm_prompt_speech_token, llm_embedding
        )
        this_tts_speech_token = tts_speech_token.unsqueeze(dim=0)
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            speed=speed
        )
        return this_tts_speech

    def vc(
            self,
            source_speech_token,
            flow_prompt_speech_token,
            prompt_speech_feat,
            flow_embedding,
            speed=1.0,
            **kwargs
    ):

        tts_speech_token = torch.tensor(source_speech_token.flatten().tolist())
        this_tts_speech_token = torch.tensor(tts_speech_token).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            speed=speed)
        return this_tts_speech.cpu()

    @classmethod
    def build_from_cfg(cls, cfg):
        sampling_rate = cfg.sampling_rate
        text_encoder_input_size = cfg.text_encoder_input_size
        llm_input_size = cfg.llm_input_size
        llm_output_size = cfg.llm_output_size
        spk_embed_dim = cfg.spk_embed_dim

        llm_cfg = cfg.llm
        llm_cfg.text_encoder_input_size = text_encoder_input_size
        llm_cfg.llm_input_size = llm_input_size
        llm_cfg.llm_output_size = llm_output_size
        llm_cfg.spk_embed_dim = spk_embed_dim
        llm_cls = registry.get_model_class(llm_cfg.model_cls)
        llm = llm_cls.build_from_cfg(llm_cfg)

        flow_cfg = cfg.flow
        flow_cfg.spk_embed_dim = spk_embed_dim
        flow_cls = registry.get_model_class(flow_cfg.model_cls)
        flow = flow_cls.build_from_cfg(flow_cfg)

        hift_cfg = cfg.hift
        hift_cfg.sampling_rate = sampling_rate
        hift_cls = registry.get_model_class(hift_cfg.model_cls)
        hift = hift_cls.build_from_cfg(hift_cfg)

        model = cls(
            llm, flow, hift
        )

        if cfg.get("ckpt", None) is not None:
            model.load('{}/llm.pt'.format(cfg.ckpt),
                       '{}/flow.pt'.format(cfg.ckpt),
                       '{}/hift.pt'.format(cfg.ckpt))
        if cfg.get("load_jit", False):
            model.load_jit('{}/llm.text_encoder.fp16.zip'.format(cfg.ckpt),
                           '{}/llm.llm.fp16.zip'.format(cfg.ckpt),
                           '{}/flow.encoder.fp32.zip'.format(cfg.ckpt))
        return model
