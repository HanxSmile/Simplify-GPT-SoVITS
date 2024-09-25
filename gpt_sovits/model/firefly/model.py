import math
import torch
import torch.nn as nn
from .modules import sequence_mask, ConvNeXtEncoder, HiFiGANGenerator
from .spec_transform import LogMelSpectrogram
from .finite_scalar_quantize import DownsampleFiniteScalarQuantize
from gpt_sovits.common.registry import registry


@registry.register_model("filefly_vqgan")
class FireflyArchitecture(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            head: nn.Module,
            quantizer: nn.Module,
            spec_transform: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def forward(self, x: torch.Tensor, template=None, mask=None) -> torch.Tensor:
        if self.spec_transform is not None:
            x = self.spec_transform(x)

        x = self.backbone(x)
        if mask is not None:
            x = x * mask

        if self.quantizer is not None:
            vq_result = self.quantizer(x)
            x = vq_result.z

            if mask is not None:
                x = x * mask

        x = self.head(x, template=template)

        if x.ndim == 2:
            x = x[:, None, :]

        if self.vq is not None:
            return x, vq_result

        return x

    def encode(self, audios, audio_lengths):
        audios = audios.float()

        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        # Encode
        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths) -> torch.Tensor:
        mel_masks = sequence_mask(
            feature_lengths * self.downsample_factor,
            indices.shape[2] * self.downsample_factor,
        )
        mel_masks_float_conv = mel_masks[:, None, :].float()
        audio_lengths = (
                feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        )

        audio_masks = sequence_mask(
            audio_lengths,
            indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length,
        )
        audio_masks_float_conv = audio_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv

        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.backbone, "remove_parametrizations"):
            self.backbone.remove_parametrizations()

        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build_from_cfg(cls, cfg):
        spec_transform_cfg = cfg.spec_transform
        spec_transform = LogMelSpectrogram(
            sample_rate=spec_transform_cfg.sample_rate,
            n_mels=spec_transform_cfg.n_mels,
            n_fft=spec_transform_cfg.n_fft,
            hop_length=spec_transform_cfg.hop_length,
            win_length=spec_transform_cfg.win_length
        )
        backbone_cfg = cfg.backbone
        backbone = ConvNeXtEncoder(
            input_channels=backbone_cfg.input_channels,
            depths=list(backbone_cfg.depths),
            dims=list(backbone_cfg.dims),
            drop_path_rate=backbone_cfg.drop_path_rate,
            kernel_size=backbone_cfg.kernel_size,
        )
        head_cfg = cfg.head
        head = HiFiGANGenerator(
            hop_length=head_cfg.hop_length,
            upsample_rates=head_cfg.upsample_rates,
            upsample_kernel_sizes=head_cfg.upsample_kernel_sizes,
            resblock_kernel_sizes=head_cfg.resblock_kernel_sizes,
            resblock_dilation_sizes=head_cfg.resblock_dilation_sizes,
            num_mels=head_cfg.num_mels,
            upsample_initial_channel=head_cfg.upsample_initial_channel,
            pre_conv_kernel_size=head_cfg.pre_conv_kernel_size,
            post_conv_kernel_size=head_cfg.post_conv_kernel_size,
        )

        quantizer_cfg = cfg.quantizer
        quantizer = DownsampleFiniteScalarQuantize(
            input_dim=quantizer_cfg.input_dim,
            n_groups=quantizer_cfg.n_groups,
            n_codebooks=quantizer_cfg.n_codebooks,
            levels=quantizer_cfg.levels,
            downsample_factor=quantizer_cfg.downsample_factor,
        )

        model = cls(
            backbone=backbone,
            head=head,
            quantizer=quantizer,
            spec_transform=spec_transform,
        )
        ckpt = cfg.get("ckpt", None)
        if not ckpt:
            return model

        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }
        model.load_state_dict(state_dict, strict=False)
        return model
