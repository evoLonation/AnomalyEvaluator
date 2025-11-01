from contextlib import redirect_stdout
import pstats
from unicodedata import category
from detectors.MuSc import MuSc
from detectors.AnomalyCLIP import AnomalyCLIP
from detectors.AdaCLIP import AdaCLIP
from detectors.AACLIP import AACLIP
from evaluation import evaluation_detection
from data import MPDD, BTech, MVTecAD, MVTecLOCO, RealIADDevidedByAngle, VisA, RealIAD, generate_all_samples_batch_dataset, generate_random_batch_dataset, generate_summary_view
from pathlib import Path
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
        # anomaly_clip = AnomalyCLIP(type="visa")
        # evaluation_detection(path, anomaly_clip, mvtec, batch_size=batch_size)
        # anomaly_clip = AnomalyCLIP(type="mvtec")
        # visa = VisA(
        #     path=Path("~/hdd/VisA").expanduser(),
        # )
        # evaluation_detection(path, anomaly_clip, visa, batch_size=batch_size)
        # realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        # evaluation_detection(path, anomaly_clip, realiad, batch_size=batch_size)
        # mvtec = MVTecAD(
        #     path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        # )
        # ada_clip = AdaCLIP(type="visa", batch_size=batch_size)
        # evaluation_detection(path, ada_clip, mvtec, batch_size=batch_size)
        # ada_clip = AdaCLIP(type="mvtec", batch_size=batch_size)
        # visa = VisA(
        #     path=Path("~/hdd/VisA").expanduser(),
        # )
        # evaluation_detection(path, ada_clip, visa, batch_size=batch_size)
        # realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        # evaluation_detection(path, ada_clip, realiad, batch_size=batch_size)

        # AnomalyCLIP on MVTecLOCO
        # anomaly_clip = AnomalyCLIP(type="mvtec")
        # mvtec_loco = MVTecLOCO(
        #     path=Path("~/hdd/mvtec_loco_anomaly_detection").expanduser(),
        # )
        # evaluation_detection(
        #     path,
        #     anomaly_clip,
        #     mvtec_loco,
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )

        # AdaCLIP on MVTecLOCO
        # ada_clip = AdaCLIP(type="mvtec", batch_size=batch_size)
        # evaluation_detection(
        #     path,
        #     ada_clip,
        #     mvtec_loco,
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )

        # AACLIP on MVTec
        # aa_clip = AACLIP(batch_size=batch_size, type="visa", dataset="MVTec")
        # mvtec = MVTecAD(
        #     path=Path("~/hdd/mvtec_anomaly_detection").expanduser(),
        # )
        # evaluation_detection(path, aa_clip, mvtec, batch_size=batch_size)

        # AACLIP on VisA
        # aa_clip = AACLIP(batch_size=batch_size, type="mvtec", dataset="VisA")
        # visa = VisA(
        #     path=Path("~/hdd/VisA_pytorch/1cls").expanduser(),
        # )
        # evaluation_detection(path, aa_clip, visa, batch_size=batch_size)

        # AACLIP on RealIAD
        # aa_clip = AACLIP(batch_size=batch_size, type="visa", dataset="RealIAD")
        # realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        # evaluation_detection(path, aa_clip, realiad, batch_size=batch_size)

        # AACLIP on MVTecLOCO
        # aa_clip = AACLIP(batch_size=batch_size, type="mvtec", dataset="MVTecLOCO")
        # mvtec_loco = MVTecLOCO(
        #     path=Path("~/hdd/mvtec_loco_anomaly_detection").expanduser(),
        # )
        # evaluation_detection(
        #     path,
        #     aa_clip,
        #     mvtec_loco,
        #     category="breakfast_box",
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )

        # aa_clip = AACLIP(batch_size=batch_size, type="visa", dataset="RealIAD")
        # realiad = RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
        # evaluation_detection(
        #     path,
        #     aa_clip,
        #     realiad,
        #     category=["pcb2", "capsules", "pcb3"],
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )

        # realiad = RealIADDevidedByAngle()
        # realiad = MVTecAD()
        # realiad = VisA()
        # realiad = MVTecLOCO()
        # realiad = RealIAD()
        # categories = ["toothbrush", "toy", "mint", "switch", "plastic_nut"]
        # categories = [
        #     f"{cat}_C{angle_i}" for cat in categories for angle_i in range(1, 6)
        # ]
        # print(categories)
        # anomaly_clip = AnomalyCLIP(type="mvtec")
        # evaluation_detection(
        #     path,
        #     anomaly_clip,
        #     realiad,
        #     category=categories,
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )
        # del anomaly_clip
        # ada_clip = AdaCLIP(type="mvtec", batch_size=batch_size)
        # evaluation_detection(
        #     path,
        #     ada_clip,
        #     realiad,
        #     category=categories,
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )
        # del ada_clip
        # aa_clip = AACLIP(batch_size=batch_size, type="visa", dataset="RealIAD")
        # evaluation_detection(
        #     path,
        #     aa_clip,
        #     realiad,
        #     category=categories,
        #     save_anomaly_score=True,
        #     save_anomaly_map=True,
        #     batch_size=batch_size,
        # )
        # del aa_clip

        musc = MuSc()
        # visa = generate_all_samples_batch_dataset(VisA())
        # evaluation_detection(
        #     path,
        #     musc,
        #     visa,
        # )
        
        realiad = generate_random_batch_dataset(RealIAD(), batch_size=200)
        evaluation_detection(
            path,
            musc,
            realiad,
        )
        # del realiad
        # realiad_angle = generate_random_batch_dataset(RealIADDevidedByAngle(), batch_size=200)
        # evaluation_detection(
        #     path,
        #     musc,
        #     realiad_angle,
        # )
        # del realiad_angle
        # mvtec = generate_random_batch_dataset(MVTecAD(), batch_size=8)
        # evaluation_detection(
        #     path,
        #     musc,
        #     mvtec,
        # )
        # del mvtec


    finally:
        profile.disable()
        profile.dump_stats("evaluation2.prof")
        with open("evaluation2.prof.txt", "w") as f:
            stats = pstats.Stats(profile, stream=f)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(20)
