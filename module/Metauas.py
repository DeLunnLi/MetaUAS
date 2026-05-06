from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder, get_preprocessing_params
from torchvision import transforms

from .alignment import AlignmentModule


def _import_smp_decoders():
    try:
        from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

        return FPNDecoder, UnetDecoder
    except ImportError:
        pass
    from segmentation_models_pytorch.fpn.decoder import FPNDecoder
    try:
        from segmentation_models_pytorch.unet.model import UnetDecoder
    except ImportError:
        from segmentation_models_pytorch.unet.decoder import UnetDecoder
    return FPNDecoder, UnetDecoder


FPNDecoder, UnetDecoder = _import_smp_decoders()


def _make_unet_decoder(
    cls,
    *,
    encoder_channels,
    decoder_channels,
    n_blocks,
    attention_type,
    num_coam_layers,
):
    params = inspect.signature(cls.__init__).parameters
    kw: dict = {
        "encoder_channels": encoder_channels,
        "decoder_channels": decoder_channels,
        "n_blocks": n_blocks,
        "attention_type": attention_type,
    }
    if "num_coam_layers" in params:
        kw["num_coam_layers"] = num_coam_layers
    if "use_batchnorm" in params:
        kw["use_batchnorm"] = True
    if "use_norm" in params and "use_batchnorm" not in kw:
        kw["use_norm"] = "batchnorm"
    return cls(**kw)


class MetaUAS(nn.Module):
    """
    Same topology as https://github.com/gaobb/MetaUAS/blob/main/metauas.py
    Forward: (query, prompt) -> sigmoid mask logits, same as your training loop.
    Optional: forward_batch(batch_dict) with keys query_image / prompt_image for eval parity.
    """

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        encoder_depth: int,
        decoder_depth: int,
        num_crossfa_layers: int,
        alignment_type: str,
        fusion_policy: str,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_alignment_layers = num_crossfa_layers
        self.alignment_type = alignment_type
        self.fusion_policy = fusion_policy

        align_input_channels = [448, 160, 56]
        align_hidden_channels = [224, 80, 28]
        encoder_channels = [3, 48, 32, 56, 160, 448]
        decoder_channels = [256, 128, 64, 64, 48]
        segmentation_channels = 256

        self.encoder = get_encoder(
            self.encoder_name,
            in_channels=3,
            depth=self.encoder_depth,
            weights="imagenet",
        )
        preparams = get_preprocessing_params(self.encoder_name, pretrained="imagenet")
        self.preprocess = transforms.Normalize(preparams["mean"], preparams["std"])

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.decoder_name == "unet":
            encoder_out_channels = list(encoder_channels[self.encoder_depth - self.decoder_depth :])
            if self.fusion_policy == "cat":
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i + 1)] *= 2
            num_coam_layers = self.num_alignment_layers if self.fusion_policy == "cat" else 0
            self.decoder = _make_unet_decoder(
                UnetDecoder,
                encoder_channels=encoder_out_channels,
                decoder_channels=decoder_channels,
                n_blocks=self.decoder_depth,
                attention_type="scse",
                num_coam_layers=num_coam_layers,
            )
        elif self.decoder_name == "fpn":
            encoder_out_channels = list(encoder_channels)
            if self.fusion_policy == "cat":
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i + 1)] = 2 * encoder_out_channels[-(i + 1)]
            self.decoder = FPNDecoder(
                encoder_channels=encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=decoder_channels[-1],
                dropout=0.2,
                merge_policy="add",
            )
        elif self.decoder_name == "fpnadd":
            encoder_out_channels = list(encoder_channels)
            if self.fusion_policy == "cat":
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i + 1)] = 2 * encoder_out_channels[-(i + 1)]
            self.decoder = FPNDecoder(
                encoder_channels=encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="add",
            )
        elif self.decoder_name == "fpncat":
            encoder_out_channels = list(encoder_channels)
            if self.fusion_policy == "cat":
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i + 1)] = 2 * encoder_out_channels[-(i + 1)]
            self.decoder = FPNDecoder(
                encoder_channels=encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="cat",
            )
        else:
            raise ValueError(f"Unsupported decoder_name: {self.decoder_name}")

        self.alignment = nn.ModuleList()
        if self.alignment_type in {"sa", "na", "ha"}:
            self.alignment = nn.ModuleList(
                [
                    AlignmentModule(
                        input_channels=align_input_channels[i],
                        hidden_channels=align_hidden_channels[i],
                        alignment_type=self.alignment_type,
                        fusion_policy=self.fusion_policy,
                    )
                    for i in range(self.num_alignment_layers)
                ]
            )

        if self.decoder_name == "fpncat":
            self.mask_head = nn.Conv2d(segmentation_channels * 4, 1, kernel_size=1, stride=1, padding=0)
        elif self.decoder_name == "fpnadd":
            self.mask_head = nn.Conv2d(segmentation_channels, 1, kernel_size=1, stride=1, padding=0)
        else:
            self.mask_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, query: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        query_input = self.preprocess(query)
        prompt_input = self.preprocess(prompt)

        with torch.no_grad():
            query_encoded_features = self.encoder(query_input)
            prompt_encoded_features = self.encoder(prompt_input)

        q_feats = list(query_encoded_features)
        p_feats = list(prompt_encoded_features)
        for i in range(len(self.alignment)):
            q_feats[-(i + 1)] = self.alignment[i](q_feats[-(i + 1)], p_feats[-(i + 1)])

        query_decoded_features = self.decoder(*q_feats[self.encoder_depth - self.decoder_depth :])

        if self.decoder_name in {"fpn", "fpncat", "fpnadd"}:
            output = F.interpolate(
                self.mask_head(query_decoded_features), scale_factor=4, mode="bilinear", align_corners=False
            )
        elif self.decoder_name == "unet":
            if self.decoder_depth == 4:
                output = F.interpolate(
                    self.mask_head(query_decoded_features), scale_factor=2, mode="bilinear", align_corners=False
                )
            else:
                output = self.mask_head(query_decoded_features)
        else:
            output = self.mask_head(query_decoded_features)

        return output.sigmoid()

    def forward_batch(self, batch: dict) -> torch.Tensor:
        return self.forward(batch["query_image"], batch["prompt_image"])
