from pathlib import Path
from data.detection_dataset import DetectionDataset


def create_symlink_structure(dataset: DetectionDataset, target_dir: Path):
    """
    根据DetectionDataset的meta_info创建符号链接目录结构

    目录结构:
    target_dir/
    ├── category1/
    │   ├── ground_truth/
    │   │   └── bad/         # mask符号链接
    │   ├── test/
    │   │   ├── good/        # 正常测试图像
    │   │   └── bad/         # 异常测试图像
    │   └── train/
    │       └── good/        # 正常训练图像
    """
    if not dataset.has_meta():
        raise ValueError("Dataset必须包含meta_info")

    for category in dataset.get_categories():
        category_dir = target_dir / category

        # 创建目录结构
        train_good_dir = category_dir / "train" / "good"
        test_good_dir = category_dir / "test" / "good"
        test_bad_dir = category_dir / "test" / "bad"
        gt_bad_dir = category_dir / "ground_truth" / "bad"

        train_good_dir.mkdir(parents=True, exist_ok=True)
        test_good_dir.mkdir(parents=True, exist_ok=True)
        test_bad_dir.mkdir(parents=True, exist_ok=True)
        gt_bad_dir.mkdir(parents=True, exist_ok=True)

        meta_dataset = dataset.get_meta(category)

        for idx, sample in enumerate(meta_dataset):
            image_path = Path(sample.image_path)
            image_name = f"{idx:04d}_{image_path.name}"

            if sample.label:  # 异常样本
                # 测试集异常图像
                link_path = test_bad_dir / image_name
                if not link_path.exists():
                    link_path.symlink_to(image_path)

                # ground_truth mask - 使用与异常图像相同的文件名
                assert sample.mask_path is not None, f"异常样本必须包含mask路径: {image_path}"
                mask_path = Path(sample.mask_path)
                mask_link_path = gt_bad_dir / (image_name.split('.')[0] + mask_path.suffix)
                if not mask_link_path.exists():
                    mask_link_path.symlink_to(mask_path)
            else:  # 正常样本 - 假设都是测试集
                link_path = test_good_dir / image_name
                if not link_path.exists():
                    link_path.symlink_to(image_path)
        train_dataset = dataset.get_train_meta(category)
        for idx, sample in enumerate(train_dataset):
            image_path = Path(sample)
            image_name = f"{idx:04d}_{image_path.name}"
            link_path = train_good_dir / image_name
            if not link_path.exists():
                link_path.symlink_to(image_path)

        print(f"创建完成: {category}")


if __name__ == "__main__":
    from data import RealIADDevidedByAngle
    dataset = RealIADDevidedByAngle()
    output_directory = Path("./Real-IAD-angle")
    create_symlink_structure(dataset, output_directory)