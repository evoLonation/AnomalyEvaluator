from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated, Optional, Tuple, cast
import open_clip
from transformers import (
    CLIPImageProcessor,
    CLIPImageProcessorFast,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)
import transformers.models.clip.modeling_clip as tc
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from data import MVTecAD
from detector import DetectionResult, Detector
from PIL import Image
from typecheck import Float, typechecker, check_type, Int, Bool


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    input_image_size: tuple[int, int] = (518, 518)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, model: tc.CLIPEncoderLayer):
        super().__init__()
        self.model = model
        self.embed_dim = self.model.embed_dim

    @typechecker
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "N S {self.embed_dim}"],
        attention_mask: Float[torch.Tensor, "N S S"] | None,
        causal_attention_mask: Float[torch.Tensor, "N S S"] | None = None,
    ) -> Float[torch.Tensor, "N S {self.embed_dim}"]:
        return self.model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, model: tc.CLIPVisionEmbeddings, config: CLIPConfig):
        super().__init__()
        self.model = model

        self.embed_dim = self.model.embed_dim
        self.input_H, self.input_W = config.input_image_size
        assert (
            self.input_H % self.model.patch_size == 0
            and self.input_W % self.model.patch_size == 0
        )
        self.patch_num = (self.input_H // self.model.patch_size) * (
            self.input_W // self.model.patch_size
        )

        self.interpolate_pos_encoding = config.input_image_size != (
            self.model.image_size,
            self.model.image_size,
        )

    @typechecker
    def forward(
        self, pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"]
    ) -> Float[torch.Tensor, "N {self.patch_num}+1 {self.embed_dim}"]:
        return self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )


@typechecker
@dataclass
class CLIPVisionOutput:
    cls_token: Float[torch.Tensor, "N projection_dim"]
    patch_tokens: Float[torch.Tensor, "N patch_num projection_dim"] | None


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self, model: tc.CLIPVisionTransformer, projection: nn.Linear, config: CLIPConfig
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(model.embeddings, config)
        self.pre_layernorm = model.pre_layrnorm
        self.encoder_layers = nn.ModuleList(
            [
                CLIPEncoderLayer(cast(tc.CLIPEncoderLayer, layer))
                for layer in model.encoder.layers
            ]
        )
        self.post_layernorm = model.post_layernorm
        self.projection = projection

        self.input_H, self.input_W = self.embeddings.input_H, self.embeddings.input_W
        self.patch_num = self.embeddings.patch_num
        self.projection_dim = projection.out_features

    @typechecker
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"],
    ) -> CLIPVisionOutput:
        hidden_states = self.embeddings(pixel_values=pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states=hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        tokens = self.projection(hidden_states)
        check_type(
            tokens, Float, torch.Tensor, f"N {self.patch_num + 1} {self.projection_dim}"
        )

        return CLIPVisionOutput(
            cls_token=tokens[:, 0],
            patch_tokens=tokens[:, 1:],
        )


class CLIPTextTransformer(nn.Module):
    def __init__(self, model: tc.CLIPTextTransformer, projection: nn.Linear):
        super().__init__()
        self.embeddings = model.embeddings
        self.encoder_layers = nn.ModuleList(
            [
                CLIPEncoderLayer(cast(tc.CLIPEncoderLayer, layer))
                for layer in model.encoder.layers
            ]
        )
        self.layernorm = model.final_layer_norm
        self.projection = projection

        self.projection_dim = projection.out_features
        self.eos_token_id = model.config.eos_token_id

    @typechecker
    def forward(
        self,
        input_ids: Int[torch.Tensor, "N S"],
        attention_mask: Bool[torch.Tensor, "N S"],
    ) -> Float[torch.Tensor, "N {self.projection_dim}"]:
        hidden_states = self.embeddings(
            input_ids=input_ids,
        )
        hidden_states = check_type(
            hidden_states, Float, torch.Tensor, f"N S {self.embeddings.embed_dim}"
        )

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.shape, hidden_states.dtype, hidden_states.device
        )
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )

        hidden_states = self.layernorm(hidden_states)
        assert self.eos_token_id == 2
        hidden_state = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            input_ids.argmax(dim=-1),
        ]
        token = self.projection(hidden_state)
        return token


class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        model: CLIPModel = CLIPModel.from_pretrained(config.model_name)
        self.vision = CLIPVisionTransformer(
            model.vision_model,
            model.visual_projection,
            config,
        )
        self.text = CLIPTextTransformer(
            model.text_model,
            model.text_projection,
        )
        self.logit_scale = model.logit_scale

        self.input_H, self.input_W = self.vision.input_H, self.vision.input_W

    @typechecker
    def forward(
        self,
        input_ids: Int[torch.Tensor, "NT S"],
        attention_mask: Bool[torch.Tensor, "NT S"],
        pixel_values: Float[torch.Tensor, "NI C=3 {self.input_H} {self.input_W}"],
    ) -> Float[torch.Tensor, "NI NT"]:
        text_features = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_outputs: CLIPVisionOutput = self.vision(
            pixel_values=pixel_values,
        )
        image_features = image_outputs.cls_token
        text_features = check_type(text_features, Float, torch.Tensor, "NT D")
        image_features = check_type(image_features, Float, torch.Tensor, "NI D")
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image * self.logit_scale.exp()

        return logits_per_image


class CLIPDetector(Detector):
    def __init__(self):
        super().__init__(name="CLIPDetector")
        self.model_name = "openai/clip-vit-large-patch14-336"
        self.clip_model = CLIPModel.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self.preprocessor = CLIPProcessor.from_pretrained(self.model_name)

        self.normal_prompt = "a photo of a normal object"
        self.anomaly_prompt = "a photo of a broken or anomalous object"

    @torch.no_grad()
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        batch_size = len(image_paths)
        images = [Image.open(path) for path in image_paths]

        # use preprocessor
        inputs = self.preprocessor(
            text=[self.normal_prompt, self.anomaly_prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        print(inputs["input_ids"])
        print(inputs["attention_mask"])
        pixel_values: torch.Tensor = inputs["pixel_values"]
        print(pixel_values.shape)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_image = check_type(
            logits_per_image, Float, torch.Tensor, f"{batch_size} 2"
        )
        pred_scores = F.softmax(logits_per_image, dim=1)[:, 1]

        # use image processor and tokenizer separately
        tokenized_text = self.tokenizer([self.normal_prompt, self.anomaly_prompt])
        processed_images = self.image_processor(images=images)
        input_ids = tokenized_text["input_ids"]
        max_ids = max([len(x) for x in input_ids])
        input_ids = torch.stack(
            [
                torch.tensor(x + [self.tokenizer.pad_token_id] * (max_ids - len(x)))
                for x in input_ids
            ]
        )
        attention_mask = tokenized_text["attention_mask"]
        attention_mask = torch.stack(
            [torch.tensor(x + [0] * (max_ids - len(x))) for x in attention_mask]
        )
        pixel_values2 = torch.tensor(np.stack(processed_images["pixel_values"]))
        print(input_ids)
        print(attention_mask)
        print(pixel_values2.shape)
        assert torch.allclose(pixel_values, pixel_values2), (
            torch.abs(pixel_values - pixel_values2).max(),
        )

        outputs2 = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values2,
        )
        logits_per_image2 = outputs2.logits_per_image
        logits_per_image2 = check_type(
            logits_per_image2, Float, torch.Tensor, f"{batch_size} 2"
        )
        pred_scores2 = F.softmax(logits_per_image2, dim=1)[:, 1]

        assert torch.allclose(pred_scores, pred_scores2), (pred_scores, pred_scores2)

        return DetectionResult(
            pred_scores=pred_scores.numpy(),
            anomaly_maps=np.zeros((len(image_paths), 518, 518), dtype=np.float32),
        )


if __name__ == "__main__":
    detector = CLIPDetector()
    result = detector(
        [
            "/mnt/ssd/home/zhaozy/hdd/mvtec_anomaly_detection/bottle/test/broken_large/000.png"
        ],
        "bottle",
    )
    print(result)
