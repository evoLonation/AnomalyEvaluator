from dataclasses import dataclass
import beartype
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, ParamSpec, TypeVar, cast, Callable
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    CLIPVisionConfig,
)
import transformers.models.clip.modeling_clip as tc
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from jaxtyping import Float, Int, Bool, jaxtyped, Int64

from .data import ImageSize

from .detector import DetectionResult, Detector, TensorDetector
from .loss import focal_loss, binary_dice_loss


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image_size: tuple[int, int] = (518, 518)
    enable_vvv: bool = False


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
    patch_tokens: Float[torch.Tensor, "N patch_num projection_dim"]


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

        self.enable_vvv = config.enable_vvv

        if self.enable_vvv:
            vision_config: CLIPVisionConfig = cast(
                CLIPVisionConfig, model.encoder.config
            )
            encoder_layers_vvv: list[CLIPEncoderLayer] = []
            for layer in self.encoder_layers:
                layer_vvv = CLIPEncoderLayer(tc.CLIPEncoderLayer(vision_config))
                layer_vvv.load_state_dict(layer.state_dict())
                v_proj = layer_vvv.model.self_attn.v_proj
                layer_vvv.model.self_attn.k_proj = v_proj
                layer_vvv.model.self_attn.q_proj = v_proj
                layer_vvv.to(config.device)
                encoder_layers_vvv.append(layer_vvv)
            self.encoder_layers_vvv = nn.ModuleList(encoder_layers_vvv)

    @jaxtyped(typechecker=None)
    def forward(
        self,
        pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"],
    ) -> tuple[
        Float[torch.Tensor, "N {self.projection_dim}"],
        Float[torch.Tensor, "N {self.patch_num} {self.projection_dim}"],
    ]:
        embeds = self.embeddings(pixel_values=pixel_values)
        embeds = self.pre_layernorm(embeds)

        hidden_states = embeds
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states=hidden_states)
        hidden_states = self.post_layernorm(hidden_states)

        if not self.enable_vvv:
            tokens: Float[
                torch.Tensor, f"N {self.patch_num+1} {self.projection_dim}"
            ] = self.projection(hidden_states)
            return tokens[:, 0], tokens[:, 1:]
        else:
            hidden_states_vvv = embeds
            for layer in self.encoder_layers_vvv:
                hidden_states_vvv = layer(hidden_states=hidden_states_vvv)
            hidden_states_vvv = self.post_layernorm(hidden_states_vvv)
            return (
                self.projection(hidden_states[:, 0]),
                self.projection(hidden_states_vvv[:, 1:]),
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
        position_ids = None
        if start_pos is not None:
            position_ids = torch.arange(
                start_pos, start_pos + inputs.shape[1], device=inputs.device
            ).unsqueeze(0)
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
        config: CLIPConfig,
    ):
        super().__init__()
        self.embedding = embedding
        self.embed_dim = self.embedding.embed_dim

        self.learnables: nn.ParameterList = nn.ParameterList()
        bos_token_id = tokenizer.bos_token_id
        bos_embed = self.embedding(
            torch.tensor([bos_token_id], device=config.device).unsqueeze(0), start_pos=0
        ).squeeze(0, 1)
        self.bos_embed: Float[torch.Tensor, f"{self.embed_dim}"] = nn.Buffer(
            bos_embed, persistent=False
        )
        self.buffer_namer = lambda i: f"static_{i}"
        self.types: list[Literal["static", "learnable"]] = []
        now_token_len = 1  # start from bos token
        now_buffer_i = 0
        for x in prompt:
            if isinstance(x, int):
                self.learnables.append(
                    nn.Parameter(
                        torch.empty(x, self.embed_dim, device=config.device)
                    )  # todo: a better way to init weight
                )
                self.types.append("learnable")
                now_token_len += x
            else:
                token_id_list: list[int] = tokenizer(x, add_special_tokens=False)[
                    "input_ids"
                ]  # pyright: ignore[reportAssignmentType]
                token_ids = torch.tensor(token_id_list, device=config.device)
                embed: Float[torch.Tensor, f"{len(token_ids)} {self.embed_dim}"] = (
                    self.embedding(
                        token_ids.unsqueeze(0), start_pos=now_token_len
                    ).squeeze(0)
                )
                setattr(
                    self,
                    self.buffer_namer(now_buffer_i),
                    nn.Buffer(embed, persistent=False),
                )
                self.types.append("static")
                now_token_len += len(token_ids)
                now_buffer_i += 1

        eos_token_id = tokenizer.eos_token_id
        eos_embed = self.embedding(
            torch.tensor([eos_token_id], device=config.device).unsqueeze(0),
            start_pos=now_token_len,
        ).squeeze(0, 1)
        self.eos_embed: Float[torch.Tensor, f"{self.embed_dim}"] = nn.Buffer(
            eos_embed, persistent=False
        )
        now_token_len += 1

        self.token_length = now_token_len

        for learnable in self.learnables:
            nn.init.normal_(learnable, mean=0.0, std=0.02)

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
        embeds = torch.zeros(
            (max_token_length, self.embed_dim),
            dtype=torch.float32,
            device=self.bos_embed.device,
        )
        embeds[0] = self.bos_embed
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
                embed = self.embedding(embed.unsqueeze(0), start_pos=now_pos).squeeze(0)
            embeds[now_pos : now_pos + len(embed)] = embed
            now_pos += len(embed)
        embeds[now_pos] = self.eos_embed
        now_pos += 1

        attention_mask = torch.zeros(
            (max_token_length,), dtype=torch.bool, device=embeds.device
        )
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
            config.model_name, device_map=config.device
        )
        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(config.model_name)
        self.vision = CLIPVisionTransformer(
            model.vision_model,
            model.visual_projection,
            config,
        )
        self.normal_prompt = LearnablePrompt(
            tokenizer,
            CLIPTextEmbeddings(model.text_model.embeddings),
            [12, "a photo of a normal object"],
            config,
        )
        self.anomaly_prompt = LearnablePrompt(
            tokenizer,
            CLIPTextEmbeddings(model.text_model.embeddings),
            [12, "a photo of a broken or anomalous object"],
            config,
        )

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
        pixel_values: Float[torch.Tensor, "N C=3 {self.input_H} {self.input_W}"],
    ) -> tuple[
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2 {self.input_H} {self.input_W}"],
    ]:
        max_token_length = max(
            self.normal_prompt.token_length, self.anomaly_prompt.token_length
        )
        n_embeds, n_attention_mask, n_eos_pos = self.normal_prompt(max_token_length)
        a_embeds, a_attention_mask, a_eos_pos = self.anomaly_prompt(max_token_length)
        text_token: Float[torch.Tensor, "2 D"] = self.text(
            embeds=torch.stack([n_embeds, a_embeds]),
            eos_positions=torch.tensor([n_eos_pos, a_eos_pos]),
            attention_mask=torch.stack([n_attention_mask, a_attention_mask]),
        )
        text_token = text_token / text_token.norm(p=2, dim=-1, keepdim=True)
        cls_token, patch_tokens = self.vision(
            pixel_values=pixel_values,
        )

        cls_token: Float[torch.Tensor, "N D"] = cls_token
        cls_token = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = cls_token @ text_token.t()
        logits_per_image = logits_per_image * self.logit_scale.exp()
        logits_per_image = logits_per_image.softmax(dim=-1)

        patch_tokens: Float[torch.Tensor, "N P D"] = patch_tokens
        patch_tokens = patch_tokens / patch_tokens.norm(p=2, dim=-1, keepdim=True)

        logits_per_patch: Float[torch.Tensor, "N P 2"] = patch_tokens @ text_token.t()
        logits_per_patch = logits_per_patch * self.logit_scale.exp()
        logits_per_patch = logits_per_patch.softmax(dim=-1)

        patch_num = patch_tokens.shape[1]
        patch_H, patch_W = int(patch_num**0.5), int(patch_num**0.5)
        assert patch_num == patch_H * patch_W, (patch_num, patch_H, patch_W)
        logits_per_patch: Float[torch.Tensor, f"N {patch_H} {patch_W} 2"] = (
            logits_per_patch.view(
                logits_per_patch.shape[0], patch_H, patch_W, logits_per_patch.shape[2]
            )
        )
        logits_per_pixel: Float[torch.Tensor, f"N 2 {self.input_H} {self.input_W}"] = (
            F.interpolate(
                logits_per_patch.permute(0, 3, 1, 2),
                size=(self.input_H, self.input_W),
                mode="bilinear",
            )
        )

        return logits_per_image, logits_per_pixel

    @generate_call_signature(forward)
    def __call__(self): ...

    def get_loss(
        self,
        logits_per_image: Float[torch.Tensor, "N 2"],
        logits_per_pixel: Float[torch.Tensor, "N 2 H W"],
        image_labels: Bool[torch.Tensor, "N"],
        image_masks: Bool[torch.Tensor, "N H W"],
    ) -> tuple[
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:

        image_loss = F.cross_entropy(logits_per_image, image_labels.long())

        pixel_loss_focal = focal_loss(logits_per_pixel, image_masks)
        pixel_loss_dice_pos = binary_dice_loss(
            logits_per_pixel[:, 1, :, :], image_masks
        )
        pixel_loss_dice_neg = binary_dice_loss(
            logits_per_pixel[:, 0, :, :], ~image_masks
        )
        pixel_loss = pixel_loss_focal + pixel_loss_dice_pos + pixel_loss_dice_neg

        total_loss = image_loss + pixel_loss

        return total_loss, image_loss, pixel_loss


class CLIPDetector(TensorDetector):
    def __init__(self, clip: CLIP, image_size: ImageSize, device: torch.device):
        super().__init__(name="CLIPDetector", image_size=image_size)
        self.clip = clip
        self.device = device
        self.clip.to(self.device)

    @torch.no_grad()
    def __call__(
        self, images: Float[torch.Tensor, "N C H W"], class_name: str
    ) -> DetectionResult:
        self.clip.eval()
        images = images.to(self.device)
        logits_per_image, logits_per_pixel = self.clip(images)
        pred_scores = logits_per_image[:, 1]
        pred_masks = logits_per_pixel[:, 1, :, :]
        return DetectionResult(
            pred_scores=pred_scores.cpu().numpy(),
            anomaly_maps=pred_masks.cpu().numpy(),
        )


if __name__ == "__main__":
    pass
    # detector = CLIPDetector()
    # result = detector(
    #     [
    #         "/mnt/ssd/home/zhaozy/hdd/mvtec_anomaly_detection/bottle/test/broken_large/000.png"
    #     ],
    #     "bottle",
    # )
    # print(result)
