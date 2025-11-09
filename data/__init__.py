if __name__ == "__main__":
    from .cached_dataset import (
        CachedDataset,
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
    from .summary import generate_summary_view

    for dataset in list[CachedDataset](
        [
            MVTecAD(),
            VisA(),
            RealIAD(),
            RealIADDevidedByAngle(),
            MVTecLOCO(),
            MPDD(),
            BTech(),
            _3CAD(),
        ]
    ):
        generate_summary_view(dataset.get_meta_dataset())
        dataset.get_tensor_dataset((336, 336))
        dataset.get_tensor_dataset((518, 518))

    dataset = ReinAD()
    generate_summary_view(dataset.get_tensor_dataset(None))
