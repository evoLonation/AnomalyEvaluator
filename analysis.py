from contextlib import redirect_stdout
from pathlib import Path
import pstats
from shutil import copy
from typing import cast

import torch

from data.cached_impl import RealIADDevidedByAngle
from data.detection_dataset import DetectionDataset
from data.utils import ImageSize
from evaluator.analysis import analyze_errors_by_csv, read_scores_csv
from evaluator.evaluation import evaluation_detection
from align.rect import AlignedDataset
from torch.utils.data import RandomSampler, Sampler
from evaluator.musc2 import MuScConfig2, MuScDetector2
import evaluator.reproducibility as repro


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    categories = [
        # "audiojack_C1",
        # "audiojack_C2",
        # "audiojack_C3",
        # "audiojack_C4",
        # "audiojack_C5",
        "button_battery_C1",
        "button_battery_C2",
        "button_battery_C3",
        "button_battery_C4",
        "button_battery_C5",
        # "end_cap_C1",
        # "end_cap_C2",
        # "end_cap_C3",
        # "end_cap_C4",
        # "end_cap_C5",
        # "mint_C1",
        # "mint_C2",
        # "mint_C3",
        # "mint_C4",
        # "mint_C5",
    ]
    result_dir = Path("results/musc_11_27")
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    for category in categories:
        run = "b4_MuSc2(r1)(dino)(train)RealIAD(angle)_42"
        meta_dataset = dataset.get_meta(category)
        print(f"Analyzing results for run: {run}, category: {category}")
        result = analyze_errors_by_csv(
            scores_csv=result_dir / f"{run}_scores/{category}.csv",
            dataset=meta_dataset,
        )
        total_number = len(meta_dataset)
        positive_number = len([True for x in meta_dataset if x.label])
        negative_number = total_number - positive_number
        print(f"Total sample number: {total_number}")
        print(
            f"      positive / negative number: {positive_number} / {negative_number}"
        )
        print(
            f"false nagetive / positive number: {len(result.fn_indices)} / {len(result.fp_indices)}"
        )
        fn_indices_list = result.fn_indices.tolist()
        fn_scores_list = result.fn_scores.tolist()
        sorted_fn_list = sorted(
            zip(fn_indices_list, fn_scores_list), key=lambda x: x[1]
        )
        fn_indices_list = [x[0] for x in sorted_fn_list]
        fn_scores_list = [x[1] for x in sorted_fn_list]
        fp_indices_list = result.fp_indices.tolist()
        fp_scores_list = result.fp_scores.tolist()
        sorted_fp_list = sorted(
            zip(fp_indices_list, fp_scores_list), key=lambda x: -x[1]
        )
        fp_indices_list = [x[0] for x in sorted_fp_list]
        fp_scores_list = [x[1] for x in sorted_fp_list]
        dst_dir = Path(f"results_analysis/error_images/{run}/{category}")
        dst_dir_fp = dst_dir / "fp"
        dst_dir_fp.mkdir(parents=True, exist_ok=True)
        dst_dir_fn = dst_dir / "fn"
        dst_dir_fn.mkdir(parents=True, exist_ok=True)
        for i, (idx, score) in enumerate(zip(fn_indices_list, fn_scores_list)):
            meta = meta_dataset[idx]
            dst_image_path = dst_dir_fn / (
                f"{i:04d}_s{score:.4f}_i" + Path(meta.image_path).suffix
            )
            dst_mask_path = dst_dir_fn / (
                f"{i:04d}_s{score:.4f}_mask" + Path(meta.mask_path).suffix
            )
            copy(meta.image_path, dst_image_path)
            copy(meta.mask_path, dst_mask_path)
        for i, (idx, score) in enumerate(zip(fp_indices_list, fp_scores_list)):
            meta = meta_dataset[idx]
            dst_image_path = dst_dir_fp / (
                f"{i:04d}_s{score:.4f}" + Path(meta.image_path).suffix
            )
            copy(meta.image_path, dst_image_path)
