from pathlib import Path
import time

from matplotlib.pylab import ndarray
import numpy as np
import torch
from data.cached_dataset import MVTecAD
from torch.utils.data import DataLoader

from data.detection_dataset import MetaSample, TensorSampleBatch
from data.utils import (
    generate_empty_mask,
    generate_image,
    generate_mask,
    normalize_image,
)
from PIL import Image


if __name__ == "__main__":
    dataset = MVTecAD()
    image_size = (518, 518)
    categories = list(dataset.get_meta_dataset().category_datas.keys())
    subset = dataset.get_tensor_dataset(image_size).category_datas[categories[0]]
    dataloader = DataLoader(subset, collate_fn=subset.collate_fn)
    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()
    print(f"Time taken to iterate through dataset: {end_time - start_time} seconds")

    def collate_fn(samples: list[MetaSample]) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        images = []
        masks = []
        labels = []
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            # image = np.array(image)
            # image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            if sample.mask_path is None:
                mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
            else:
                mask = Image.open(sample.mask_path).convert("L")
            images.append(image)
            masks.append(mask)
            labels.append(sample.label)
        return list(zip(images, masks, labels))
    dataloader = DataLoader(
        dataset.get_meta_dataset().category_datas[categories[0]], collate_fn=collate_fn
    )
    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()
    print(
        f"Time taken to iterate through dataset with collate_fn: {end_time - start_time} seconds"
    )
