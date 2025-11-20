from contextlib import redirect_stdout
from pathlib import Path
import pstats
from typing import cast

import torch

from data.cached_impl import RealIADDevidedByAngle
from data.detection_dataset import DetectionDataset
from data.utils import ImageSize
from evaluator.analysis import analyze_errors_by_csv, read_scores_csv
from evaluator.evaluation import evaluation_detection
from evaluator.align import AlignedDataset
from torch.utils.data import RandomSampler, Sampler
from evaluator.musc2 import MuScConfig2, MuScDetector2
import evaluator.reproducibility as repro


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    categories = [
        "audiojack_C1",
        "audiojack_C2",
        "audiojack_C3",
        "audiojack_C4",
        "audiojack_C5",
    ]
    result_dir = Path("results/musc_11_19")
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    for category in categories:
        runs = [
            "MuSc2_RealIAD(angle)",
            "MuSc2_RealIAD(angle)_seed43",
        ]
        fn_indices_list = []
        meta_dataset = dataset.get_meta(category)
        for run in runs:
            print(f"Analyzing results for run: {run}, category: {category}")
            result = analyze_errors_by_csv(
                scores_csv=result_dir / f"{run}_scores/{category}.csv",
                dataset=meta_dataset,
            )
            total_number = len(meta_dataset)
            positive_number = len([True for x in meta_dataset if x.label])
            negative_number = total_number - positive_number
            print(f"Total sample number: {len(dataset.get_meta(category))}")
            print(
                f"      positive / negative number: {positive_number} / {negative_number}"
            )
            print(
                f"false nagetive / positive number: {len(result.fn_indices)} / {len(result.fp_indices)}"
            )
            fn_indices_list.append(set(result.fn_indices.tolist()))
        common_fn_indices = set.intersection(*fn_indices_list)
        indices, _, scores = read_scores_csv(
            result_dir / f"{runs[0]}_scores/{category}.csv"
        )
        common_fn_scores = [scores[indices.index(i)] for i in common_fn_indices]
        print(f"Common false negative number: {len(common_fn_indices)}")
        sorted_fn_indices = [
            x[0]
            for x in sorted(
                zip(common_fn_indices, common_fn_scores), key=lambda x: x[1]
            )
        ]
