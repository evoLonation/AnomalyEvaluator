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
from detectors.MuSc import MuSc, MuScTensor
from evaluator.musc2 import MuScConfig, MuScDetector2
from data import DetectionDataset, MVTecAD, RealIAD, VisA
from evaluator.evaluation import evaluation_detection

# from evaluator.train import TrainConfig, test, train
from torch.utils.data import RandomSampler, Sampler
import evaluator.reproducibility as repro

# from evaluator.evaluation import evaluation_detection
# from evaluator.data import MPDD, BTech, MVTecAD, MVTecLOCO, RealIADDevidedByAngle, VisA, RealIAD, generate_all_samples_batch_dataset, generate_random_batch_dataset, generate_summary_view

if __name__ == "__main__":
    # Initialize all baseline detectors
    # profile = cProfile.Profile()
    # profile.enable()
    # repro.init(42)
    # musc = MuScTensor(image_size=(518, 518), max_batch_size=16)
    # for dataset in cast(list[DetectionDataset], [MVTecAD(), VisA()]):
    #     category_samplers: dict[str, Sampler] = {
    #         k: RandomSampler(
    #             v, replacement=False, generator=torch.Generator().manual_seed(42)
    #         )
    #         for k, v in dataset.get_tensor_dataset(
    #             musc.image_size
    #         ).category_datas.items()
    #     }
    #     evaluation_detection(
    #         path=Path("results/musc_official_tensor"),
    #         detector=musc,
    #         dataset=dataset,
    #         batch_size=16,
    #         category_samplers=category_samplers,
    #     )
    repro.init(42)
    detector = MuScDetector2(MuScConfig())
    # for dataset in cast(list[DetectionDataset], [MVTecAD(), VisA()]):
    # for dataset in cast(list[DetectionDataset], [VisA()]):
    for dataset in cast(
        # list[DetectionDataset], [MVTecAD(), VisA(), RealIAD(), RealIADDevidedByAngle()]
        list[DetectionDataset], [MVTecAD()]
    ):
        dataset = RandomRotatedDetectionDataset(dataset, seed=42)
        # for dataset in cast(list[DetectionDataset], [RealIAD(), RealIADDevidedByAngle()]):
        evaluation_detection(
            path=Path("results/musc2_oc"),
            detector=detector,
            dataset=dataset,
            batch_size=16,
            sampler_getter=lambda _, d: RandomSampler(
                d, replacement=False, generator=torch.Generator().manual_seed(42)
            ),
            category="bottle",
            save_anomaly_map=True,
            save_anomaly_score=True,
        )

    # train(
    #     TrainConfig(
    #         enable_vvv=False,
    #         image_size=(518, 518),
    #         num_epochs=15,
    #     ),
    #     dir_suffix="no_vvv_518"
    # )
    # train(resume_dir=Path("results/train/11.08_14:37:10_518"))
    # test(Path("results/train/11.08_14:37:10_518"))
    # test(Path("results/train/11.05_00:19:10"))
    # train(resume_dir=Path("results/train/11.06_12:25:47"))
    # for epoch in [0, 1, 3, 5, 7, 10]:
    #     test(Path(f"results/train/11.06_12:25:47_promptlearning"), epoch_num=epoch)
    # test(
    #     Path(f"results/train/11.06_12:25:47_promptlearning"),
    #     enable_vvv=True,
    #     suffix="enable_vvv",
    # )
    # path = Path("./detection_evaluation")
    # clip = CLIPDetector()
    # print(clip.clip.state_dict().keys())
    # mvtec = MVTecAD()
    # evaluation_detection(path, clip, mvtec, batch_size=16)

    # finally:
    #     profile.disable()
    #     profile.dump_stats("evaluation2.prof")
    #     with open("evaluation2.prof.txt", "w") as f:
    #         stats = pstats.Stats(profile, stream=f)
    #         stats.sort_stats(pstats.SortKey.TIME)
    #         stats.print_stats(20)
