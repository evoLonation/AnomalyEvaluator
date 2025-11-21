from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import pstats
from typing import cast
import torch
from data.cached_impl import RealIADDevidedByAngle
from data.rotate import RandomRotatedDetectionDataset
from data.utils import ImageSize
from evaluator.align import AlignedDataset
from evaluator.analysis import get_all_error_images, get_error_dataset
from evaluator.musc2 import MuScConfig2, MuScDetector2
from data import DetectionDataset, MVTecAD, RealIAD, VisA
from evaluator.evaluation import evaluation_detection
from torch.utils.data import RandomSampler, Sampler
import evaluator.reproducibility as repro
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    seed: int = 42,
    suffix: str = "",
    high_resolution: bool = False,
    aligned: bool = False,
    save_result: bool = False,
    k_list: str = "",
    r_list: str = "",
    layers: str = "",
    topmin_min: float | None = None,
    topmin_max: float | None = None,
    is_dino: bool = False,
    patch_match: bool = False,
    borrow_indices: bool = False,
    r1_with_r5_indice: bool = False,
    consistent_features: bool = False,
):
    repro.init(seed)
    config = MuScConfig2()
    if patch_match:
        config.patch_match = True
    if is_dino:
        config.is_dino = True
    if high_resolution:
        config.input_image_size = ImageSize.square(1022)
        config.image_resize = 1024
    if borrow_indices:
        config.borrow_indices = True
    if r1_with_r5_indice:
        config.r1_with_r3_indice = True
    if consistent_features:
        config.consistent_feature = True
    if k_list:
        config.k_list = [int(k) for k in k_list.split(",")]
    if r_list:
        config.r_list = [int(r) for r in r_list.split(",")]
    if layers:
        config.feature_layers = [int(l) for l in layers.split(",")]
    if topmin_min is not None:
        config.topmin_min = topmin_min
    if topmin_max is not None:
        config.topmin_max = topmin_max
    detector = MuScDetector2(config)
    path = f"results/musc{suffix}"
    if aligned:
        path += "_aligned"
    if high_resolution:
        path += "_1022"
    path = Path(path)
    namer=lambda det, dset: f"{det.name}_{dset.get_name()}_s{seed}"
    batch_size = 16
    # datasets = [MVTecAD(), VisA(), RealIAD(), RealIADDevidedByAngle()]
    categories = None
    # categories = [
    #     "audiojack_C1",
    #     "audiojack_C2",
    #     "audiojack_C3",
    #     "audiojack_C4",
    #     "audiojack_C5",
    # ]
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    if aligned:
        dataset = AlignedDataset(dataset)
    log_path = path / (namer(detector, dataset)+".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as f:
        with redirect_stdout(f), redirect_stderr(f):
            evaluation_detection(
                path=path,
                detector=detector,
                dataset=dataset,
                batch_size=batch_size,
                sampler_getter=lambda c, d: RandomSampler(
                    d,
                    replacement=False,
                    generator=torch.Generator().manual_seed(seed),
                ),
                save_anomaly_score=save_result,
                namer=namer,
            )
    # dataset = RealIADDevidedByAngle()
    # for category in categories:
    #     get_all_error_images(
    #         scores_csv=Path(
    #             f"results/musc_aligned/MuSc2_RealIAD(angle)(aligned)_scores/{category}.csv"
    #         ),
    #         dataset=datasets[0].get_meta(category),
    #         save_dir=Path(f"results/musc_aligned/MuSc2_RealIAD(angle)(aligned)_errors/{category}"),
    #     )
    # dataset = get_error_dataset(
    #     datasets[0], [Path(f"results/musc_aligned/MuSc2_RealIAD(angle)(aligned)_scores/{c}.csv") for c in categories], categories
    # )
    # evaluation_detection(
    #     path=path,
    #     detector=detector,
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     sampler_getter=lambda _, d: RandomSampler(
    #         d, replacement=False, generator=torch.Generator().manual_seed(seed)
    #     ),
    # )


if __name__ == "__main__":
    app()
