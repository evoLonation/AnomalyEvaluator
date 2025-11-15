from pathlib import Path
import numpy as np
from tqdm import tqdm
from data.detection_dataset import MetaDataset
from data import MVTecAD, RealIAD, VisA


def get_dataset_info(
    dataset: MetaDataset,
    save_dir: Path = Path("results/dataset_analysis"),
    patch_grid: tuple[int, int] = (37, 37),
):
    """
    对每个 category：
        有多少样本，多少异常样本，多少正常样本，异常样本比例是多少
        异常样本中：
            异常部分占整张图片面积的比例的分布情况，给出均值、中位数、最大值、最小值，绘制出分布图
            同理异常块在一张图中的数量的分布情况，给出均值、中位数、最大值、最小值，绘制出分布图
            patch级别的异常面积占比：计算异常部分占多少个patch
    结果保存为 json 文件和图片:
    save_dir/{name}/
        dataset_info.json
        anomaly_area_ratio_distribution.png
        anomaly_count_distribution.png
        anomaly_patch_count_distribution.png
        {category}/
            dataset_info.json
            anomaly_area_ratio_distribution.png
            anomaly_count_distribution.png
            anomaly_patch_count_distribution.png
    """
    import json
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy import ndimage

    def save_statistics_and_plots(
        save_dir: Path,
        category_name: str,
        area_ratios: list,
        anomaly_counts: list,
        anomaly_patch_counts: list,
        category_info: dict,
    ):
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "dataset_info.json", "w") as f:
            json.dump(category_info, f, indent=2)

        plt.figure(figsize=(10, 6))
        plt.hist(area_ratios, bins=100, edgecolor="black", alpha=0.7)
        plt.xlabel("Anomaly Area Ratio")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.title(f"{category_name} - Anomaly Area Ratio Distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            save_dir / "anomaly_area_ratio_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        max_count = int(max(anomaly_counts))
        min_count = int(min(anomaly_counts))
        plt.hist(
            anomaly_counts,
            bins=range(min_count, max_count + 2),
            edgecolor="black",
            alpha=0.7,
        )
        plt.xlabel("Number of Anomaly Regions")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.title(f"{category_name} - Anomaly Count Distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            save_dir / "anomaly_count_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_patch_counts, bins=100, edgecolor="black", alpha=0.7)
        plt.xlabel("Number of Anomaly Patches")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.title(f"{category_name} - Anomaly Patch Count Distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            save_dir / "anomaly_patch_count_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    dataset_save_dir = save_dir / dataset.name
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    assert not any(dataset_save_dir.iterdir()), f"{dataset_save_dir} is not empty!"

    dataset_info = {}
    all_area_ratios = []
    all_anomaly_counts = []
    all_anomaly_patch_counts = []

    for category, samples in tqdm(dataset.category_datas.items()):
        total_samples = len(samples)
        anomaly_samples = [s for s in samples if s.label]
        normal_samples = [s for s in samples if not s.label]
        num_anomaly = len(anomaly_samples)
        num_normal = len(normal_samples)
        anomaly_ratio = num_anomaly / total_samples if total_samples > 0 else 0

        area_ratios = []
        anomaly_counts = []
        anomaly_patch_counts = []

        def process_sample(sample):
            if sample.mask_path is None:
                return None, None, None
            mask = np.array(Image.open(sample.mask_path).convert("L"))
            binary_mask = mask > 0

            total_pixels = binary_mask.size
            anomaly_pixels = np.sum(binary_mask)
            area_ratio = anomaly_pixels / total_pixels

            label_result = ndimage.label(binary_mask)
            num_components = int(label_result[1])  # type: ignore

            total_patches = patch_grid[0] * patch_grid[1]
            patch_anomaly_count = area_ratio * total_patches

            return area_ratio, num_components, patch_anomaly_count

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(process_sample, sample): sample
                for sample in anomaly_samples
            }
            for future in tqdm(
                as_completed(futures),
                total=len(anomaly_samples),
                desc=f"Processing {category}",
            ):
                area_ratio, num_components, patch_count = future.result()
                if area_ratio is not None:
                    area_ratios.append(area_ratio)
                    anomaly_counts.append(num_components)
                    anomaly_patch_counts.append(patch_count)

        category_info = {
            "total_samples": total_samples,
            "anomaly_samples": num_anomaly,
            "normal_samples": num_normal,
            "anomaly_ratio": anomaly_ratio,
        }

        if area_ratios:
            category_info["area_ratio_stats"] = {
                "mean": float(np.mean(area_ratios)),
                "median": float(np.median(area_ratios)),
                "max": float(np.max(area_ratios)),
                "min": float(np.min(area_ratios)),
            }
            category_info["anomaly_count_stats"] = {
                "mean": float(np.mean(anomaly_counts)),
                "median": float(np.median(anomaly_counts)),
                "max": int(np.max(anomaly_counts)),
                "min": int(np.min(anomaly_counts)),
            }
            category_info["anomaly_patch_count_stats"] = {
                "mean": float(np.mean(anomaly_patch_counts)),
                "median": float(np.median(anomaly_patch_counts)),
                "max": float(np.max(anomaly_patch_counts)),
                "min": float(np.min(anomaly_patch_counts)),
            }
            all_area_ratios.extend(area_ratios)
            all_anomaly_counts.extend(anomaly_counts)
            all_anomaly_patch_counts.extend(anomaly_patch_counts)

            category_save_dir = dataset_save_dir / category
            save_statistics_and_plots(
                category_save_dir,
                category,
                area_ratios,
                anomaly_counts,
                anomaly_patch_counts,
                category_info,
            )

        dataset_info[category] = category_info

    total_dataset_info = {}
    if all_area_ratios:
        total_samples_count = sum(
            info["total_samples"] for info in dataset_info.values()
        )
        total_anomaly_count = sum(
            info["anomaly_samples"] for info in dataset_info.values()
        )
        total_normal_count = sum(
            info["normal_samples"] for info in dataset_info.values()
        )
        total_dataset_info = {
            "total_samples": total_samples_count,
            "anomaly_samples": total_anomaly_count,
            "normal_samples": total_normal_count,
            "anomaly_ratio": (
                total_anomaly_count / total_samples_count
                if total_samples_count > 0
                else 0
            ),
            "area_ratio_stats": {
                "mean": float(np.mean(all_area_ratios)),
                "median": float(np.median(all_area_ratios)),
                "max": float(np.max(all_area_ratios)),
                "min": float(np.min(all_area_ratios)),
            },
            "anomaly_count_stats": {
                "mean": float(np.mean(all_anomaly_counts)),
                "median": float(np.median(all_anomaly_counts)),
                "max": int(np.max(all_anomaly_counts)),
                "min": int(np.min(all_anomaly_counts)),
            },
            "anomaly_patch_count_stats": {
                "mean": float(np.mean(all_anomaly_patch_counts)),
                "median": float(np.median(all_anomaly_patch_counts)),
                "max": float(np.max(all_anomaly_patch_counts)),
                "min": float(np.min(all_anomaly_patch_counts)),
            },
        }

        save_statistics_and_plots(
            dataset_save_dir,
            f"{dataset.name} (All Categories)",
            all_area_ratios,
            all_anomaly_counts,
            all_anomaly_patch_counts,
            total_dataset_info,
        )

    final_dataset_info = {"__total__": total_dataset_info, **dataset_info}
    with open(dataset_save_dir / "dataset_info.json", "w") as f:
        json.dump(final_dataset_info, f, indent=2)


if __name__ == "__main__":
    get_dataset_info(RealIAD().get_meta_dataset())
    get_dataset_info(MVTecAD().get_meta_dataset())
    get_dataset_info(VisA().get_meta_dataset())
