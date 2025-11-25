from contextlib import redirect_stdout
from pathlib import Path
import pstats
from shutil import copy
from typing import cast

import numpy as np
import torch

from data.cached_impl import RealIADDevidedByAngle
from data.detection_dataset import DetectionDataset
from data.utils import ImageSize
from evaluator.analysis import (
    analyze_errors_by_csv,
    compare_analyze,
    compare_analyze_to_plot,
    read_scores_csv,
)
from evaluator.evaluation import evaluation_detection
from align.rect import AlignedDataset
from torch.utils.data import RandomSampler, Sampler
from evaluator.musc2 import MuScConfig2, MuScDetector2
import evaluator.reproducibility as repro


if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    result_dir = Path("results/musc_11_24")
    dataset = RealIADDevidedByAngle()
    categories = dataset.get_categories()
    # categories = [
    #     # "audiojack_C1",
    #     # "audiojack_C2",
    #     # "audiojack_C3",
    #     # "audiojack_C4",
    #     # "audiojack_C5",
    #     "bottle_cap_C4"
    # ]
    runs = [
        "MuSc2(r1)(dino)_RealIAD(angle)_s42",
        "MuSc2(r1)(dino)(od0.02)_RealIAD(angle)_s42",
    ]
    analysis_name = "r1_od0.2"
    dataset.filter_categories(categories)
    for category in categories:
        meta_dataset = dataset.get_meta(category)
        indices_1, _, scores_1 = read_scores_csv(
            result_dir / f"{runs[0]}_scores/{category}.csv"
        )
        indices_2, _, scores_2 = read_scores_csv(
            result_dir / f"{runs[1]}_scores/{category}.csv"
        )
        scores_1 = [x[1] for x in sorted(zip(indices_1, scores_1), key=lambda x: x[0])]
        scores_2 = [x[1] for x in sorted(zip(indices_2, scores_2), key=lambda x: x[0])]
        pred_scores1 = np.array(scores_1)
        pred_scores2 = np.array(scores_2)
        true_labels = np.array(dataset.get_labels(category))
        plot_path = Path(f"analysis/compare_analyze/{analysis_name}/{category}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        compare_analyze_to_plot(
            pred_scores1=pred_scores1,
            pred_scores2=pred_scores2,
            true_labels=true_labels,
            save_path=plot_path,
        )
        result = compare_analyze(
            pred_scores1=pred_scores1,
            pred_scores2=pred_scores2,
            true_labels=true_labels,
            top_k=30,
        )
        continue
        diff_indices = result.m1_better_on_anomaly.indices
        diff_values = result.m1_better_on_anomaly.diff_values
        print(diff_indices)
        print(diff_values)
        save_dir = Path(f"analysis/compare_images/{analysis_name}/m1_better_anomaly/{category}")
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(diff_indices):
            meta = meta_dataset[int(idx)]
            src_path = Path(meta.image_path)
            dst_path = save_dir / (
                f"{i+1}_idx{idx}_diff{diff_values[i]:.4f}" + src_path.suffix
            )
            copy(src_path, dst_path)
