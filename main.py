from contextlib import redirect_stdout
import pstats
from AdaCLIP import AdaCLIP
from evaluation import *
from AnomalyCLIP import *
import cProfile

if __name__ == "__main__":
    # Initialize all baseline detectors
    profile = cProfile.Profile()
    profile.enable()
    try:
        path = Path("./detection_evaluation")
        batch_size = 10
        # mvtec = MVTecAD(
        #     path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        # )
        # evaluation_detection(path, ada_clip, mvtec, batch_size=batch_size)
        # anomaly_clip = AnomalyCLIP(type="visa")
        # evaluation_detection(path, anomaly_clip, mvtec, batch_size=batch_size)
        # anomaly_clip = AnomalyCLIP(type="mvtec")
        # visa = VisA(
        #     path=Path("~/hdd/VisA").expanduser(),
        # )
        # evaluation_detection(path, anomaly_clip, visa, batch_size=batch_size)
        # realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        # evaluation_detection(path, anomaly_clip, realiad, batch_size=batch_size)
        mvtec = MVTecAD(
            path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        )
        ada_clip = AdaCLIP(type="visa", batch_size=batch_size)
        evaluation_detection(path, ada_clip, mvtec, batch_size=batch_size)
        ada_clip = AdaCLIP(type="mvtec", batch_size=batch_size)
        visa = VisA(
            path=Path("~/hdd/VisA").expanduser(),
        )
        evaluation_detection(path, ada_clip, visa, batch_size=batch_size)
        realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        evaluation_detection(path, ada_clip, realiad, batch_size=batch_size)

    finally:
        profile.disable()
        profile.dump_stats("evaluation2.prof")
        with open("evaluation2.prof.txt", "w") as f:
            stats = pstats.Stats(profile, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(20)
