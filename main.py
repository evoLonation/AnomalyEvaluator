from contextlib import redirect_stdout
import pstats
from evaluation2 import *
from AnomalyCLIP import *
import cProfile

if __name__ == "__main__":
    # Initialize all baseline detectors
    detectors: list[Detector] = [
        AnomalyCLIP(),
    ]

    datasets: list[DetectionDataset] = [
        MVTecAD(
            path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        )
    ]

    profile = cProfile.Profile()
    profile.enable()
    try:
        path = Path("./detection_evaluation")
        batch_size = 10
        mvtec = MVTecAD(
            path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        )
        anomaly_clip = AnomalyCLIP(type="visa")
        evaluation_detection(path, anomaly_clip, mvtec, batch_size=batch_size)
        anomaly_clip = AnomalyCLIP(type="mvtec")
        visa = VisA(
            path=Path("~/hdd/VisA").expanduser(),
        )
        evaluation_detection(path, anomaly_clip, visa, batch_size=batch_size)
        realiad = RealIAD(
            path=Path("~/hdd/RealIAD").expanduser(),
        )
        evaluation_detection(path, anomaly_clip, realiad, batch_size=batch_size)

    finally:
        profile.disable()
        profile.dump_stats("evaluation2.prof")

        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        with open("evaluation2.prof.txt", "w") as f:
            with redirect_stdout(f):
                stats.print_stats()
