from pathlib import Path

from common.algo import pca_background_mask
from data.cached_impl import MVTecAD
from data.detection_dataset import DetectionDataset
from data.utils import ImageSize, Transform, to_pil_image
from evaluator.dinov3 import DINOv3VisionTransformer
from evaluator.vit import VisionTransformerBase
from torchvision.transforms import CenterCrop
from PIL import Image

mvtec_no_background_categories = ["carpet", "grid", "leather", "tile", "wood", "zipper"]
mvtec_special_threshold_categories = {
    "screw": 0.6,
}


def generate_ground_masks(
    save_dir: Path,
    dataset: DetectionDataset,
    vision: VisionTransformerBase,
    category_thresholds: dict[str, float] | None = None,
):
    resize = 512
    image_size = ImageSize.square(512)
    grid_size = (
        image_size.h // vision.get_patch_size(),
        image_size.w // vision.get_patch_size(),
    )
    if category_thresholds is None:
        category_thresholds = {cat: 0.5 for cat in dataset.get_categories()}
    else:
        set(category_thresholds.keys()).issubset(set(dataset.get_categories()))
    for category, threshold in category_thresholds.items():
        print(f"Generating ground masks for category: {category}")
        category_dir = save_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        tensor_data = dataset.get_tensor(
            category,
            Transform(resize=resize, image_transform=CenterCrop(image_size.hw())),
        )
        for idx in range(len(tensor_data)):
            image = tensor_data[idx].image
            features = vision(image.unsqueeze(0))

            # 生成背景掩码
            background_mask = pca_background_mask(
                features, grid_size=grid_size, threshold=threshold
            ).squeeze(0)

            background_mask = background_mask.reshape(grid_size)

            # 保存图像
            background_pil = to_pil_image(background_mask).resize(
                (resize, resize), resample=Image.Resampling.NEAREST
            )
            background_pil.save(category_dir / f"{idx}_ground_mask.png")


if __name__ == "__main__":
    import evaluator.reproducibility as repro
    from evaluator.train3 import get_vision_transformer

    repro.init(42)
    dataset = MVTecAD()

    vision = DINOv3VisionTransformer()

    save_directory = Path("results_analysis/ground_masks_dinov3")
    generate_ground_masks(
        save_directory,
        dataset,
        vision,
        category_thresholds=mvtec_special_threshold_categories,
    )
