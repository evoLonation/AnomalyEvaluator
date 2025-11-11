"""
Test script to compare OpenCLIP and HuggingFace CLIP outputs
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from abc import ABC, abstractmethod
from typing import Callable, List

# Add paths for OpenCLIP
import open_clip

# Import HuggingFace CLIP
from transformers import CLIPConfig, CLIPModel
from evaluator.clip import CLIPVisionTransformer

# ----------------------------------------------------------------------------
# 1. 抽象基类 (Abstract Base Class)
# ----------------------------------------------------------------------------

class VisionTransformer(nn.Module, ABC):
    """
    VisionTransformer 基类
    
    这是一个抽象基类，用于统一不同 CLIP 视觉编码器的接口，
    以便进行标准化的预处理、特征提取和输出对比。
    
    子类必须实现:
    - name (property): 返回模型的唯一标识名称 (例如 'OpenCLIP')。
    - preprocess (property): 返回一个可调用的 torchvision transform 函数。
    - forward (method): 接受一个预处理后的图像批次张量，
      返回最终的 CLS 嵌入 (projected CLS token)。
    """
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        """返回模型的唯一标识名称"""
        pass

    @property
    @abstractmethod
    def preprocess(self) -> Callable:
        """返回一个可调用的 torchvision transform 函数"""
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        接受一个预处理后的图像批次张量，
        返回最终的 CLS 嵌入 (projected CLS token)。
        """
        pass

# ----------------------------------------------------------------------------
# 2. 具体实现子类 (Concrete Subclasses)
# ----------------------------------------------------------------------------

class OpenCLIPVisionTransformer(VisionTransformer):
    """
    OpenCLIP (ViT-L-14-336 @ 518) 实现
    """
    def __init__(self, device, image_size=518):
        super().__init__()
        print(f"Loading OpenCLIP model with image_size={image_size}...")
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14-336',
            force_image_size=image_size,
            pretrained='openai',
        )
        self.model = model.to(device)
        self.model.eval()
        
        self._preprocess = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    @property
    def name(self) -> str:
        return "OpenCLIP"

    @property
    def preprocess(self) -> Callable:
        return self._preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)


class HuggingFaceVisionTransformer(VisionTransformer):
    """
    HuggingFace CLIP (ViT-L-14-336 @ 518) 实现 - 使用 evaluator.clip 中的 CLIPVisionTransformer
    """
    def __init__(self, device, image_size=518):
        super().__init__()
        print("Loading HuggingFace CLIP model (using evaluator.clip wrapper)...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            device_map=device
        )
        self.vision_model = CLIPVisionTransformer(
            model.vision_model,
            model.visual_projection,
            (image_size, image_size),
            enable_vvv=False,
            device=device,
        )
        self.vision_model.eval()
        
        self._preprocess = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    @property
    def name(self) -> str:
        return "HuggingFace (evaluator.clip wrapper)"

    @property
    def preprocess(self) -> Callable:
        return self._preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_token, _ = self.vision_model(
            pixel_values=images,
            output_layers=[23],
            output_vvv=False,
        )
        return cls_token


class NativeHuggingFaceVisionTransformer(VisionTransformer):
    """
    原生 HuggingFace CLIP VisionTransformer (ViT-L-14-336 @ 518) 实现
    直接使用 transformers.CLIPVisionModel，不使用 evaluator 中的包装类
    """
    def __init__(self, device, image_size=518):
        super().__init__()
        print("Loading native HuggingFace CLIP VisionTransformer...")
        # config = CLIPConfig.from_pretrained(
        #     "openai/clip-vit-large-patch14-336",
        # )
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            # config=config
        )
        self.vision_model = model.vision_model.to(device)
        self.visual_projection = model.visual_projection.to(device)
        self.vision_model.eval()
        self.visual_projection.eval()
        
        self._preprocess = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    @property
    def name(self) -> str:
        return "HuggingFace (native transformers)"

    @property
    def preprocess(self) -> Callable:
        return self._preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        直接使用 transformers 的 CLIPVisionModel
        """
        vision_outputs = self.vision_model(pixel_values=images, interpolate_pos_encoding=True)
        # vision_outputs.pooler_output 是已经投影后的 CLS token
        pooled_output = vision_outputs.pooler_output  # (batch_size, hidden_size)
        # 再通过 visual_projection 投影到嵌入空间
        image_features = self.visual_projection(pooled_output)
        return image_features

# ----------------------------------------------------------------------------
# 3. 重构后的对比与主函数 (Refactored Comparison and Main)
# ----------------------------------------------------------------------------

def compare_outputs(models: List[VisionTransformer], test_image_path: str, device: torch.device):
    """Compare outputs for the same image across multiple models"""
    
    print(f"\n{'='*80}")
    print(f"Testing image: {test_image_path}")
    print(f"{'='*80}\n")
    
    img_pil = Image.open(test_image_path).convert('RGB')
    
    preprocessed_data = {}
    cls_tokens = {}
    
    # 1. Preprocess and Extract Features for all models
    for model in models:
        print(f"\n--- Processing with {model.name} ---")
        
        # 1a. Preprocessing
        transform = model.preprocess
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        preprocessed_data[model.name] = {
            "tensor": img_tensor,
            "mean": img_tensor.mean().item(),
            "std": img_tensor.std().item(),
        }
        print(f"    Input shape: {img_tensor.shape}")
        print(f"    Input mean: {preprocessed_data[model.name]['mean']:.6f}, std: {preprocessed_data[model.name]['std']:.6f}")

        # 1b. Feature Extraction
        with torch.no_grad():
            cls_token = model(img_tensor)
        cls_tokens[model.name] = cls_token
        print(f"    CLS token shape: {cls_token.shape}")
        print(f"    CLS token mean: {cls_token.mean():.6f}, std: {cls_token.std():.6f}")

    if len(models) < 2:
        print("Need at least two models to compare.")
        return

    # 2. Compare Preprocessing
    print(f"\n{'='*80}")
    print(f"Preprocessing Comparison")
    print(f"{'='*80}\n")
    
    baseline_name = models[0].name
    baseline_tensor = preprocessed_data[baseline_name]["tensor"]
    
    for i in range(1, len(models)):
        compare_name = models[i].name
        compare_tensor = preprocessed_data[compare_name]["tensor"]
        diff = (baseline_tensor - compare_tensor).abs()
        print(f"    Diff between '{baseline_name}' and '{compare_name}':")
        print(f"    - Mean: {diff.mean():.10f}, Max: {diff.max():.10f}")
        
    # 3. Compare CLS Tokens
    print(f"\n{'='*80}")
    print(f"CLS Token Comparison (vs '{baseline_name}')")
    print(f"{'='*80}")
    
    baseline_cls = cls_tokens[baseline_name]
    baseline_cls_norm = baseline_cls / baseline_cls.norm(dim=-1, keepdim=True)

    for i in range(1, len(models)):
        compare_name = models[i].name
        compare_cls = cls_tokens[compare_name]
        compare_cls_norm = compare_cls / compare_cls.norm(dim=-1, keepdim=True)
        
        print(f"\n--- Comparing '{baseline_name}' vs '{compare_name}' ---")
        
        # Normalized comparison
        print(f"    Normalized CLS token difference:")
        cls_diff = (baseline_cls_norm - compare_cls_norm).abs()
        cls_cosine = torch.cosine_similarity(baseline_cls_norm, compare_cls_norm, dim=-1)
        print(f"    - Cosine similarity: {cls_cosine.item():.10f}")
        print(f"    - Mean Diff: {cls_diff.mean():.10f}")
        print(f"    - Max Diff:  {cls_diff.max():.10f}")

        # Raw comparison
        print(f"\n    Raw (Non-normalized) CLS token difference:")
        raw_cls_diff = (baseline_cls - compare_cls).abs()
        print(f"    - Mean Diff: {raw_cls_diff.mean():.10f}, Max: {raw_cls_diff.max():.10f}")
        
        rel_cls_diff = raw_cls_diff / (baseline_cls.abs() + 1e-8)
        print(f"    - Relative Diff - mean: {rel_cls_diff.mean():.10f}, max: {rel_cls_diff.max():.10f}")

        # Element-wise comparison
        print(f"\n    First 10 elements comparison:")
        for j in range(min(10, baseline_cls.shape[-1])):
            base_val = baseline_cls[0, j].item()
            comp_val = compare_cls[0, j].item()
            diff = abs(base_val - comp_val)
            print(f"    Element {j}: {baseline_name[:10]:<10}={base_val:10.6f}, {compare_name[:10]:<10}={comp_val:10.6f}, Diff={diff:.6f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 1. Load models using the new classes
    image_size = 518
    models_to_compare = [
        OpenCLIPVisionTransformer(device, image_size=image_size),
        # HuggingFaceVisionTransformer(device, image_size=image_size),
        NativeHuggingFaceVisionTransformer(device, image_size=image_size),
        # 您可以在此处添加更多 VisionTransformer 的子类实例
        # e.g., MyCustomVisionTransformer(device)
    ]
    
    # Test with sample images
    test_images = [
        "/mnt/ssd/home/zhaozy/hdd/mvtec_anomaly_detection/bottle/test/broken_large/000.png",
        "/mnt/ssd/home/zhaozy/hdd/mvtec_anomaly_detection/bottle/test/good/000.png",
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            compare_outputs(models_to_compare, test_image, device)
        else:
            print(f"Image not found: {test_image}")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()