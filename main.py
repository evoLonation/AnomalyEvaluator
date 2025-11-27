from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import pstats
from typing import Literal, cast
import torch
from data.cached_impl import RealIADDevidedByAngle
from data.rotate import RotatedDataset
from data.utils import ImageSize

# from evaluator.align import AlignedDataset
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
    suffix: str = "test",
    batch_size: int = 16,
    high_resolution: bool = False,
    aligned: bool = False,
    rotate: bool = False,
    save_result: bool = True,
    rs: str = "",
    layers: str = "",
    tmin: float | None = None,
    tmax: float | None = None,
    dino: bool = True,
    const: Literal["none", "train", "test"] = "none",
    shift: bool = False,
    shift_agg: bool = False,
    cpu_metrics: bool = False,
    log_file: bool = True,
    debug: bool = False,
):
    if debug:
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    repro.init(seed)
    config = MuScConfig2()
    if dino:
        config.is_dino = True
    if high_resolution:
        config.input_image_size = ImageSize.square(1022)
        config.image_resize = 1024
    if rs:
        config.r_list = [int(r) for r in rs.split(",")]
    if layers:
        config.feature_layers = [int(l) for l in layers.split(",")]
    if tmin is not None:
        config.topmin_min = tmin
    if tmax is not None:
        config.topmin_max = tmax
    if shift:
        config.shift_augmentation = True
    if shift_agg:
        config.shift_aggregation = True
    path = f"results/musc"
    if suffix:
        path += f"_{suffix}"
    path = Path(path)

    def namer(detector, dataset):
        name = ""
        if batch_size != 16:
            name += f"bs{batch_size}_"
        name += detector.name
        name += dataset.get_name()
        if high_resolution:
            name += "(hr)"
        if aligned:
            name += "(aligned)"
        name += f"_{repro.get_global_seed()}"
        return name

    categories = None
    # categories = [
    #     "audiojack_C1",
    #     "audiojack_C2",
    #     "audiojack_C3",
    #     "audiojack_C4",
    #     "audiojack_C5",
    # ]
    dataset = RealIADDevidedByAngle().filter_categories(categories)
    detector = MuScDetector2(
        config,
        const_features=(const != "none"),
        train_data=dataset.get_train_tensor if const == "train" else None,
    )
    if aligned:
        # dataset = AlignedDataset(dataset)
        pass
    if rotate:
        dataset = RotatedDataset(dataset, in_order=True)

    def evaluation():
        evaluation_detection(
            path=path,
            detector=detector,
            dataset=dataset,
            batch_size=batch_size,
            sampler_getter=lambda c, d: RandomSampler(
                d,
                replacement=False,
                generator=torch.Generator().manual_seed(repro.get_global_seed()),
            ),
            save_anomaly_score=save_result,
            namer=namer,
            cpu_metrics=cpu_metrics,
        )

    if log_file:
        log_path = path / (namer(detector, dataset) + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", buffering=1) as f:
            with redirect_stdout(f), redirect_stderr(f):
                evaluation()
    else:
        evaluation()


if __name__ == "__main__":
    app()
