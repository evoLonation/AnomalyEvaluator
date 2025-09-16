from evaluation import *
from AnomalyCLIP import *
from Qwen2_5_VL import *

if __name__ == "__main__":
    # Initialize all baseline detectors
    detectors: list[Detector] = [
        AnomalyCLIP(),
        # Qwen2_5VL(),
    ]

    datasets: list[DetectionDataset] = [
        # RealIAD(path=Path("~/hdd/Real-IAD").expanduser(), sample_limit=500),
        MVTecAD(
            path=Path("~/hdd/mvtec_anomaly_detection").expanduser(), sample_limit=10
        )
    ]

    evaluation_detection_total(
        path=Path("./detection_evaluation"),
        detectors=detectors,
        datasets=datasets,
        checkpoint_n=100,
    )