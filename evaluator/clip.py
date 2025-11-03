from dataclasses import dataclass
import beartype
import beartype.door
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, ParamSpec, TypeVar, cast, Callable
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)
import transformers.models.clip.modeling_clip as tc
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from PIL import Image
from jaxtyping import Float, Int, Bool, jaxtyped

from .detector import DetectionResult, Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    input_image_size: tuple[int, int] = (518, 518)


def _returns_nn_module_call(*args):
    return nn.Module.__call__


_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_call_signature(
    forward_func: Callable[_P, _R],
) -> Callable[..., Callable[_P, _R]]:
    return _returns_nn_module_call


class CLIPEncoderLayer(nn.Module):
    def __init__(self, model: tc.CLIPEncoderLayer):
        super().__init__()
        self.model = model
        self.embed_dim = self.model.embed_dim

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "N S {self.embed_dim}"],
        attention_mask: Float[torch.Tensor, "N 1 S S"] | None = None,
        causal_attention_mask: Float[torch.Tensor, "N 1 S S"] | None = None,
    ) -> Float[torch.Tensor, "N S {self.embed_dim}"]:
        return self.model(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )[0]

    @generate_call_signature(forward)
    def __call__(self): ...


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
        print("self.interpolate_pos_encoding =", self.interpolate_pos_encoding)

    @jaxtyped(typechecker=None)
    def forward(
        self, pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"]
    ) -> Float[torch.Tensor, "N {self.patch_num}+1 {self.embed_dim}"]:
        return self.model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )

    @generate_call_signature(forward)
    def __call__(self): ...


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

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"],
    ) -> CLIPVisionOutput:
        hidden_states = self.embeddings(pixel_values=pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states=hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        tokens: Float[torch.Tensor, f"N {self.patch_num+1} {self.projection_dim}"] = (
            self.projection(hidden_states)
        )

        return CLIPVisionOutput(
            cls_token=tokens[:, 0],
            patch_tokens=tokens[:, 1:],
        )

    @generate_call_signature(forward)
    def __call__(self): ...


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, model: tc.CLIPTextEmbeddings):
        super().__init__()
        self.model = model
        self.embed_dim = self.model.token_embedding.embedding_dim

    @jaxtyped(typechecker=None)
    def forward(
        self,
        inputs: Int[torch.Tensor, "N S"] | Float[torch.Tensor, "N S {self.embed_dim}"],
        start_pos: int | None = None,
    ) -> Float[torch.Tensor, "N S {self.embed_dim}"]:
        position_ids = (
            None
            if start_pos is None
            else torch.arange(start_pos, start_pos + inputs.shape[1]).unsqueeze(0)
        )
        return self.model(
            input_ids=inputs if len(inputs.shape) == 2 else None,
            inputs_embeds=inputs if len(inputs.shape) == 3 else None,
            position_ids=position_ids,
        )

    @generate_call_signature(forward)
    def __call__(self): ...


class LearnablePrompt(nn.Module):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        embedding: CLIPTextEmbeddings,
        prompt: list[str | int],
    ):
        super().__init__()
        self.embedding = embedding
        self.embed_dim = self.embedding.embed_dim

        self.learnables: nn.ParameterList = nn.ParameterList()
        bos_token_id = tokenizer.bos_token_id
        bos_embed = self.embedding(
            torch.tensor([bos_token_id]).unsqueeze(0), start_pos=0
        )
        self.bos_embeded = nn.Buffer(bos_embed)
        self.buffer_namer = lambda i: f"static_{i}"
        self.types: list[Literal["static", "learnable"]] = []
        now_token_len = 1  # start from bos token
        now_buffer_i = 0
        for x in prompt:
            if isinstance(x, int):
                self.learnables.append(nn.Parameter(torch.randn(x, self.embed_dim)))
                self.types.append("learnable")
                now_token_len += x
            else:
                token_ids: Int[torch.Tensor, "S"] = tokenizer(
                    x, add_special_tokens=False
                )[
                    "input_ids"
                ]  # pyright: ignore[reportAssignmentType]
                embeded = self.embedding(
                    token_ids.unsqueeze(0), start_pos=now_token_len
                )
                setattr(self, self.buffer_namer(now_buffer_i), nn.Buffer(embeded))
                self.types.append("static")
                now_token_len += len(token_ids)
                now_buffer_i += 1

        eos_token_id = tokenizer.eos_token_id
        eos_embed = self.embedding(
            torch.tensor([eos_token_id]).unsqueeze(0), start_pos=now_token_len
        )
        self.eos_embeded = nn.Buffer(eos_embed)
        now_token_len += 1

        self.token_length = now_token_len

    @jaxtyped(typechecker=None)
    def forward(self, max_token_length: int) -> tuple[
        Float[torch.Tensor, "{max_token_length} {self.embed_dim}"],
        Bool[torch.Tensor, "{max_token_length}"],
        int,  # eos position
    ]:
        assert max_token_length >= self.token_length, (
            max_token_length,
            self.token_length,
        )
        embeds = torch.zeros((max_token_length, self.embed_dim), dtype=torch.float32)
        embeds[0] = self.bos_embeded
        now_pos = 1
        now_static_i = 0
        now_learnable_i = 0
        for type in self.types:
            if type == "static":
                embed: torch.Tensor = getattr(self, self.buffer_namer(now_static_i))
                now_static_i += 1
            else:
                embed = self.learnables[now_learnable_i]
                now_learnable_i += 1
                embed = self.embedding(embed.unsqueeze(0), start_pos=now_pos)
            embeds[now_pos : now_pos + len(embed)] = embed
            now_pos += len(embed)
        embeds[now_pos] = self.eos_embeded
        now_pos += 1

        attention_mask = torch.zeros((max_token_length,), dtype=torch.bool)
        attention_mask[:now_pos] = True

        return embeds, attention_mask, now_pos - 1

    @generate_call_signature(forward)
    def __call__(self): ...


class CLIPTextTransformer(nn.Module):
    def __init__(self, model: tc.CLIPTextTransformer, projection: nn.Linear):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                CLIPEncoderLayer(cast(tc.CLIPEncoderLayer, layer))
                for layer in model.encoder.layers
            ]
        )
        self.layernorm = model.final_layer_norm
        self.projection = projection

        self.projection_dim = projection.out_features

    @jaxtyped(typechecker=None)
    def forward(
        self,
        embeds: Float[torch.Tensor, "N S {self.embeddings.embed_dim}"],
        eos_positions: Int[torch.Tensor, "N"],
        attention_mask: Bool[torch.Tensor, "N S"],
    ) -> Float[torch.Tensor, "N {self.projection_dim}"]:
        hidden_states = embeds

        causal_attention_mask = _create_4d_causal_attention_mask(
            attention_mask.shape, hidden_states.dtype, hidden_states.device
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
        hidden_state = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            eos_positions,
        ]
        token = self.projection(hidden_state)
        return token

    @generate_call_signature(forward)
    def __call__(self): ...


class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        model: CLIPModel = CLIPModel.from_pretrained(
            config.model_name, device_map=device
        )
        self.vision = CLIPVisionTransformer(
            model.vision_model,
            model.visual_projection,
            config,
        )
        self.embeddings = CLIPTextEmbeddings(model.text_model.embeddings)
        self.text = CLIPTextTransformer(
            model.text_model,
            model.text_projection,
        )
        self.logit_scale = model.logit_scale

        self.input_H, self.input_W = self.vision.input_H, self.vision.input_W

        self.eos_token_id = model.config.text_config.eos_token_id

    @jaxtyped(typechecker=None)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "NT S"],
        attention_mask: Bool[torch.Tensor, "NT S"],
        pixel_values: Float[torch.Tensor, "NI C=3 {self.input_H} {self.input_W}"],
    ) -> Float[torch.Tensor, "NI NT"]:
        text_embeds = self.embeddings(input_ids)
        assert self.eos_token_id != 2 
        eos_positions = (input_ids == self.eos_token_id).int().argmax(dim=-1)
        text_features: Float[torch.Tensor, "NT D"] = self.text(
            embeds=text_embeds,
            attention_mask=attention_mask,
            eos_positions=eos_positions,
        )
        image_outputs: CLIPVisionOutput = self.vision(
            pixel_values=pixel_values,
        )
        image_features: Float[torch.Tensor, "NI D"] = image_outputs.cls_token
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image * self.logit_scale.exp()

        return logits_per_image

    @generate_call_signature(forward)
    def __call__(self): ...


class CLIPDetector(Detector):
    def __init__(self):
        super().__init__(name="CLIPDetector2")
        self.model_name = "openai/clip-vit-large-patch14-336"
        # self.clip_model = CLIPModel.from_pretrained(self.model_name, device_map=device)
        self.preprocessor = CLIPProcessor.from_pretrained(self.model_name)

        self.clip = CLIP(
            # CLIPConfig(model_name=self.model_name, input_image_size=(518, 518))
            CLIPConfig(model_name=self.model_name, input_image_size=(336, 336))
        )

        self.normal_prompt = "a photo of a normal object"
        self.anomaly_prompt = "a photo of a broken or anomalous object"

    @torch.no_grad()
    def __call__(self, image_paths: list[str], class_name: str) -> DetectionResult:
        batch_size = len(image_paths)
        images = [Image.open(path) for path in image_paths]

        inputs = self.preprocessor(
            text=[self.normal_prompt, self.anomaly_prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        input_ids: Int[torch.Tensor, "NT S"] = inputs["input_ids"].to(device)
        attention_mask: torch.Tensor = inputs["attention_mask"].to(device)
        pixel_values: torch.Tensor = inputs["pixel_values"].to(device)
        # print(input_ids)
        # print(attention_mask)
        # print(pixel_values.shape)

        # use clip from transformers
        # outputs = self.clip_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pixel_values=pixel_values,
        # )
        # logits_per_image = outputs.logits_per_image
        # logits_per_image = check_type(
        #     logits_per_image, Float, torch.Tensor, f"{batch_size} 2"
        # )
        # pred_scores = F.softmax(logits_per_image, dim=1)[:, 1]

        # use CLIP
        attention_mask = attention_mask.bool()
        logits_per_image2 = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        pred_scores2 = F.softmax(logits_per_image2, dim=1)[:, 1]

        # assert torch.allclose(pred_scores, pred_scores2), (pred_scores, pred_scores2)
        # print(pred_scores.item(), pred_scores2.item())

        return DetectionResult(
            pred_scores=pred_scores2.cpu().numpy(),
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
