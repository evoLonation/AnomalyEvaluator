from data.detection_dataset import DetectionDataset
from .cached_impl import (
    MVTecAD,
    VisA,
    RealIAD,
    RealIADDevidedByAngle,
    MVTecLOCO,
    MPDD,
    BTech,
    _3CAD,
)
from .reinad import ReinAD


if __name__ == "__main__":
    from .summary import generate_summary_view

    for dataset in list[DetectionDataset](
        [
            MVTecAD(),
            VisA(),
            RealIAD(),
            RealIADDevidedByAngle(),
            MVTecLOCO(),
            MPDD(),
            BTech(),
            _3CAD(),
            ReinAD(),
        ]
    ):
        # generate_summary_view(dataset.get_meta_dataset())
        # dataset.get_tensor_dataset(336)
        dataset.get_tensor_dataset(518)
        # dataset.get_tensor_dataset((336, 336))
        # dataset.get_tensor_dataset((518, 518))
