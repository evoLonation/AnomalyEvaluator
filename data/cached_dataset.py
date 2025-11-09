from pathlib import Path
from abc import abstractmethod
import json
from typing import override

from .utils import ImageSize
from .detection_dataset import (
    DetectionDataset,
    MetaDataset,
    MetaSample,
    CategoryMetaDataset,
    TensorDataset,
)
from .h5dataset import TensorH5Dataset


class CachedDataset(DetectionDataset):
    default_meta_save_dir = Path("data_cache/meta")
    default_tensor_save_dir = Path("data_cache/tensor")

    def __init__(
        self,
        name: str,
        data_dir: Path,
        meta_save_dir: Path | None = None,
        tensor_save_dir: Path | None = None,
    ) -> None:
        if meta_save_dir is None:
            meta_save_dir = self.default_meta_save_dir
        if tensor_save_dir is None:
            tensor_save_dir = self.default_tensor_save_dir
        meta_dataset = None
        if not MetaDataset.get_meta_csv_path(name, meta_save_dir).exists():
            category_datas = self.load_from_data_dir(data_dir)
            category_datas = {
                cat: CategoryMetaDataset(datas) for cat, datas in category_datas.items()
            }
            meta_dataset = MetaDataset(name, category_datas, data_dir)
            meta_dataset.to_csv(data_dir, meta_save_dir)

        self.name = name
        self.data_dir = data_dir
        self.tensor_save_dir = tensor_save_dir
        self.meta_save_dir = meta_save_dir
        self.meta_dataset = meta_dataset

    @override
    def get_meta_dataset(self) -> MetaDataset:
        if self.meta_dataset is None:
            self.meta_dataset = MetaDataset.from_csv(
                self.name, self.data_dir, self.meta_save_dir
            )
        return self.meta_dataset

    @override
    def get_tensor_dataset(self, image_size: ImageSize | None) -> TensorDataset:
        if not TensorH5Dataset.get_h5_path(
            self.name, self.tensor_save_dir, image_size
        ).exists():
            TensorH5Dataset.to_h5(
                self.get_meta_dataset(), self.tensor_save_dir, image_size
            )
        tensor_dataset = TensorH5Dataset.from_h5(
            self.name, self.tensor_save_dir, image_size
        )
        return tensor_dataset

    @classmethod
    @abstractmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]: ...


class MVTecLike(CachedDataset):
    def __init__(
        self,
        name: str,
        path: Path,
    ):
        super().__init__(name, path)

    good_category = "good"

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        categories = sorted(data_dir.iterdir())
        categories = [d.name for d in categories if d.is_dir()]

        image_suffixes = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp"]

        category_datas: dict[str, list[MetaSample]] = {}
        for category in categories:
            category_dir = data_dir / category / "test"
            if not category_dir.exists():
                raise ValueError(f"Category path {category_dir} does not exist.")

            samples: list[MetaSample] = []

            # 加载正常样本 (good文件夹)
            good_dir = category_dir / cls.good_category
            for img_file in sorted(good_dir.iterdir()):
                assert (
                    img_file.suffix in image_suffixes
                ), f"Unsupported image format: {img_file}"
                samples.append(
                    MetaSample(
                        image_path=str(img_file),
                        mask_path=None,
                        label=False,
                    )
                )

            # 加载异常样本 (除good外的所有文件夹)
            for anomaly_dir in sorted(category_dir.iterdir()):
                if not anomaly_dir.is_dir() or anomaly_dir.name == cls.good_category:
                    continue
                anomaly_mask_dir = (
                    data_dir / category / "ground_truth" / anomaly_dir.name
                )
                for img_file, mask_file in zip(
                    sorted(anomaly_dir.iterdir()), sorted(anomaly_mask_dir.iterdir())
                ):
                    assert mask_file.stem.startswith(
                        img_file.stem
                    ), f"Image and mask file names do not match: {img_file} vs {mask_file}"
                    samples.append(
                        MetaSample(
                            image_path=str(img_file),
                            mask_path=str(mask_file),
                            label=True,
                        )
                    )

            category_datas[category] = samples

        return category_datas


class MVTecAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_anomaly_detection").expanduser(),
    ):
        super().__init__("MVTecAD", path)


class VisA(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/VisA_pytorch/1cls").expanduser(),
    ):
        super().__init__("VisA", path)


class RealIAD(CachedDataset):
    def __init__(self, path: Path = Path("~/hdd/Real-IAD").expanduser()):
        super().__init__("RealIAD", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        json_dir = data_dir / "realiad_jsons"
        image_dir = data_dir / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()

        category_datas: dict[str, list[MetaSample]] = {}
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
            category: str = json_file.stem

            samples: list[MetaSample] = []

            for item in data["test"]:
                anomaly_class = item["anomaly_class"]
                correct_label = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                image_path = str(image_path)
                mask_path = (
                    image_dir / prefix / item["mask_path"] if correct_label else None
                )
                mask_path = str(mask_path) if mask_path is not None else None
                samples.append(
                    MetaSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        label=correct_label,
                    )
                )

            category_datas[category] = samples

        return category_datas


class RealIADDevidedByAngle(CachedDataset):
    def __init__(self, path: Path = Path("~/hdd/Real-IAD").expanduser()):
        super().__init__("RealIAD(angle)", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        category_datas = RealIAD.load_from_data_dir(data_dir)
        divided_category_datas: dict[str, list[MetaSample]] = {}
        for category, samples in category_datas.items():
            angle_category_datas: dict[str, list[MetaSample]] = {}
            for angle_i in range(1, 6):
                angle_substr = f"C{angle_i}"
                angle_indices = [
                    i
                    for i, sample in enumerate(samples)
                    if angle_substr in sample.image_path
                ]
                angle_category_datas[f"{category}_{angle_substr}"] = [
                    samples[i] for i in angle_indices
                ]
            assert len(samples) == sum(
                len(datas) for datas in angle_category_datas.values()
            ), (
                f"Data size mismatch when dividing by angle for category {category}:"
                f" {len(samples)} vs {sum(len(datas) for datas in angle_category_datas.values())}"
            )
            divided_category_datas.update(angle_category_datas)

        return divided_category_datas


class MVTecLOCO(CachedDataset):
    def __init__(
        self,
        path: Path = Path("~/hdd/mvtec_loco_anomaly_detection").expanduser(),
    ):
        super().__init__("mvtec_loco", path)

    @classmethod
    def load_from_data_dir(cls, data_dir: Path) -> dict[str, list[MetaSample]]:
        meta_file = data_dir / "meta.json"
        with open(meta_file, "r") as f:
            data = json.load(f)

        category_datas: dict[str, list[MetaSample]] = {}
        # 只使用 test 数据进行评估
        for category, samples_data in data["test"].items():
            samples: list[MetaSample] = []

            for sample in samples_data:
                img_path = data_dir / sample["img_path"]
                is_anomaly = sample["anomaly"] == 1

                if is_anomaly and sample["mask_path"]:
                    mask_path = data_dir / sample["mask_path"]
                    mask_path_str = str(mask_path)
                else:
                    mask_path_str = None

                samples.append(
                    MetaSample(
                        image_path=str(img_path),
                        mask_path=mask_path_str,
                        label=is_anomaly,
                    )
                )

            category_datas[category] = samples

        return category_datas


class MPDD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/MPDD").expanduser(),
    ):
        super().__init__("MPDD", path)


class BTech(MVTecLike):
    good_category = "ok"

    def __init__(
        self,
        path: Path = Path("~/hdd/BTech_Dataset_transformed").expanduser(),
    ):
        super().__init__("BTech", path)


class _3CAD(MVTecLike):
    def __init__(
        self,
        path: Path = Path("~/hdd/3CAD").expanduser(),
    ):
        super().__init__("3CAD", path)
