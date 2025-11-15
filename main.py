from contextlib import redirect_stdout
from pathlib import Path
import pstats
from typing import cast

import torch

# from beartype import BeartypeConf
# from detectors.MuSc import MuSc
# from detectors.AnomalyCLIP import AnomalyCLIP
# from detectors.AdaCLIP import AdaCLIP
# from detectors.AACLIP import AACLIP
# from pathlib import Path
# import cProfile

# import evaluator.test
from data.cached_impl import RealIADDevidedByAngle
from data.rotate import RandomRotatedDetectionDataset
from evaluator.musc2 import MuScConfig2, MuScDetector2
from data import DetectionDataset, MVTecAD, RealIAD, VisA
from evaluator.evaluation import evaluation_detection

# from evaluator.train import TrainConfig, test, train
from torch.utils.data import RandomSampler, Sampler
import evaluator.reproducibility as repro

# from evaluator.evaluation import evaluation_detection
# from evaluator.data import MPDD, BTech, MVTecAD, MVTecLOCO, RealIADDevidedByAngle, VisA, RealIAD, generate_all_samples_batch_dataset, generate_random_batch_dataset, generate_summary_view

if __name__ == "__main__":
    seed = 42
    repro.init(seed)
    detector = MuScDetector2(MuScConfig2())
    path = Path("results/musc2_oc")
    batch_size = 16
    # datasets = [MVTecAD(), VisA(), RealIAD(), RealIADDevidedByAngle()]
    datasets = [RealIADDevidedByAngle()]
    categories = ["audiojack_C1", "audiojack_C2", "audiojack_C3", "audiojack_C4", "audiojack_C5"]
    rotated = False
    for dataset in cast(list[DetectionDataset], datasets):
        if rotated:
            dataset = RandomRotatedDetectionDataset(dataset, seed=seed)
        evaluation_detection(
            path=path,
            detector=detector,
            dataset=dataset,
            batch_size=batch_size,
            sampler_getter=lambda _, d: RandomSampler(
                d, replacement=False, generator=torch.Generator().manual_seed(seed)
            ),
            category=categories,
            save_anomaly_map=True,
            save_anomaly_score=True,
        )


    # finally:
    #     profile.disable()
    #     profile.dump_stats("evaluation2.prof")
    #     with open("evaluation2.prof.txt", "w") as f:
    #         stats = pstats.Stats(profile, stream=f)
    #         stats.sort_stats(pstats.SortKey.TIME)
    #         stats.print_stats(20)
