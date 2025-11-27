from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Type, TypeVar
import numpy as np
from jaxtyping import Float, Bool, Int
from numpy.typing import NDArray
from tqdm import tqdm
import torch
from torcheval.metrics import (
    BinaryAUROC, BinaryAUPRC,
)

from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .detector import DetectionResult, DetectionGroundTruth, Detector, TensorDetector


@dataclass
class DetectionMetrics:
    # precision: float
    # recall: float
    # f1_score: float
    auroc: float
    ap: float
    pixel_auroc: float
    pixel_aupro: float
    pixel_ap: float
    patch_distance: float

    def __str__(self):
        return (
            # f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1-Score: {self.f1_score:.4f}, "
            f"Image-AUROC: {self.auroc:.4f}, Image-AP: {self.ap:.4f}, "
            f"Pixel-AUROC: {self.pixel_auroc:.4f}, Pixel-AUPro: {self.pixel_aupro:.4f}, "
            f"Pixel-AP: {self.pixel_ap:.4f}, Patch-Distance: {self.patch_distance:.4f}"
        )


# 摘自 AnomalyCLIP/metrics.py
# expect_fpr: 期望的假正率，只取低于这个阈值的部分来计算AUC
# PRO = 正确检测的像素数 / 真实异常区域的像素数
def cal_pro_score(
    masks: Bool[np.ndarray, "N H W"],
    amaps: Float[np.ndarray, "N H W"],
    max_step=200,
    expect_fpr=0.3,
):
    from skimage import measure

    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []

    @dataclass
    class RegionProps:
        area: int
        coord_xs: Int[np.ndarray, "N"]
        coord_ys: Int[np.ndarray, "N"]

        def __init__(self, region):
            self.area = region.area
            coords = region.coords
            self.coord_xs, self.coord_ys = coords.T

    mask_regions = [
        [RegionProps(x) for x in measure.regionprops(measure.label(mask))]
        for mask in masks
    ]
    inverse_masks = 1 - masks
    inverse_masks_sum = inverse_masks.sum()
    for th in tqdm(np.arange(min_th, max_th, delta)):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, regions in zip(binary_amaps, mask_regions):
            for region in regions:
                tp_pixels = binary_amap[region.coord_xs, region.coord_ys].sum()
                pro.append(tp_pixels / region.area)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks_sum
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    assert isinstance(pro_auc, float)
    return pro_auc


def cal_pro_score_gpu(
    masks: Bool[torch.Tensor, "N H W"],
    amaps: Float[torch.Tensor, "N H W"],
    max_step: int = 200,
    expect_fpr: float = 0.3,
) -> float:
    assert torch.cuda.is_available()
    dev = torch.device("cuda")

    masks_t = masks.to(dev)
    amaps_t = amaps.to(dev)

    min_th, max_th = amaps_t.min(), amaps_t.max()
    ths = torch.linspace(min_th, max_th, max_step, device=dev)
    inverse_masks_t = ~masks_t
    inverse_masks_sum = inverse_masks_t.sum()

    from skimage import measure

    @dataclass
    class RegionProps:
        area: int
        coord_xs: Int[NDArray[np.int_], "N"]
        coord_ys: Int[NDArray[np.int_], "N"]

        def __init__(self, region):
            self.area = region.area
            coords = region.coords
            self.coord_xs, self.coord_ys = coords.T

    mask_regions = [
        [RegionProps(x) for x in measure.regionprops(measure.label(mask))]
        for mask in masks.cpu().numpy()
    ]

    # 用于存储每个块的计算结果
    pros_chunks = []
    fprs_chunks = []

    total_size = 2**30  # 1GB
    # total_size = 4 * total_size  # 4GB
    th_chunk_size = total_size // (
        masks_t.shape[0] * masks_t.shape[1] * masks_t.shape[2] * 4
    )  # 每个float32占4字节
    if th_chunk_size == 0:
        th_chunk_size = 1
    print(f"Threshold chunk size: {th_chunk_size}")
    # 核心优化：按块（chunk）循环处理阈值，避免一次性载入全部
    for th_chunk in tqdm(torch.split(ths, th_chunk_size)):
        # th_chunk 是一个小的阈值张量，例如包含 50 个阈值

        # 1. 对当前块进行并行阈值化
        #    中间张量 binary_amaps_t_chunk 的形状为 [N, len(th_chunk), H, W]
        #    其大小远小于之前的 [N, max_step, H, W]，从而节省了显存
        binary_amaps_t_chunk = amaps_t.unsqueeze(1) > th_chunk.view(1, -1, 1, 1)

        # 2. 对当前块并行计算 FPR
        fp_pixels_chunk = torch.logical_and(
            inverse_masks_t.unsqueeze(1), binary_amaps_t_chunk
        ).sum(dim=(0, 2, 3))
        fprs_chunk = fp_pixels_chunk / inverse_masks_sum
        fprs_chunks.append(fprs_chunk)

        # 3. 对当前块并行计算 PRO
        all_region_pros_chunk = []
        for i, regions in enumerate(mask_regions):
            if not regions:
                continue

            binary_amap_i_chunk = binary_amaps_t_chunk[i]
            for region in regions:
                coord_xs_t = torch.from_numpy(region.coord_xs).long().to(dev)
                coord_ys_t = torch.from_numpy(region.coord_ys).long().to(dev)

                tp_pixels_per_th_chunk = binary_amap_i_chunk[
                    :, coord_xs_t, coord_ys_t
                ].sum(dim=1)
                region_pro_chunk = tp_pixels_per_th_chunk / region.area
                all_region_pros_chunk.append(region_pro_chunk.unsqueeze(0))

        if not all_region_pros_chunk:
            pros_chunk = torch.zeros_like(th_chunk)
        else:
            pros_chunk = torch.cat(all_region_pros_chunk, dim=0).mean(dim=0)

        pros_chunks.append(pros_chunk)

    pros = torch.cat(pros_chunks)
    fprs = torch.cat(fprs_chunks)

    # 后处理部分与之前完全相同
    pros, fprs, ths = pros.cpu().numpy(), fprs.cpu().numpy(), ths.cpu().numpy()

    idxes = fprs < expect_fpr
    if not np.any(idxes):
        return 0.0

    fprs = fprs[idxes]
    pros = pros[idxes]

    if fprs.max() == fprs.min():
        return 0.0

    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros)
    assert isinstance(pro_auc, float)
    return pro_auc


class MetricsCalculatorInterface(ABC):
    @abstractmethod
    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        pass

    @abstractmethod
    def compute(self) -> DetectionMetrics:
        pass


class BaseMetricsCalculator(MetricsCalculatorInterface):
    def __init__(self, cpu: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_device = torch.device("cpu" if cpu else self.device)
        # self.precision_metric = BinaryPrecision()
        # self.recall_metric = BinaryRecall()
        # self.f1_metric = BinaryF1Score()
        self.auroc_metric = BinaryAUROC(device=self.device)
        self.ap_metric = BinaryAUPRC(device=self.device)
        self.pixel_auroc_metric = BinaryAUROC(device=self.pixel_device)
        self.pixel_ap_metric = BinaryAUPRC(device=self.pixel_device)
        # todo: 改成渐进式
        self.anomaly_maps: list[Float[torch.Tensor, "N H W"]] = []
        self.true_masks: list[Bool[torch.Tensor, "N H W"]] = []
        self.patch_distances_sum = 0.0
        self.patch_distances_num = 0

    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        preds.pred_scores = preds.pred_scores.to(self.device)
        preds.anomaly_maps = preds.anomaly_maps.to(self.pixel_device)
        gts.true_labels = gts.true_labels.to(self.device)
        gts.true_masks = gts.true_masks.to(self.pixel_device)

        pred_score = preds.pred_scores
        true_label = gts.true_labels.int()
        # self.precision_metric.update(pred_score, true_label)
        # self.recall_metric.update(pred_score, true_label)
        # self.f1_metric.update(pred_score, true_label)
        self.ap_metric.update(pred_score, true_label)
        self.auroc_metric.update(pred_score, true_label)

        pred_score_pixel = preds.anomaly_maps.flatten()
        true_mask_pixel = gts.true_masks.flatten().int()
        self.pixel_auroc_metric.update(pred_score_pixel, true_mask_pixel)
        self.pixel_ap_metric.update(pred_score_pixel, true_mask_pixel)
        self.anomaly_maps.append(preds.anomaly_maps)
        self.true_masks.append(gts.true_masks)
        self.patch_distances_sum += preds.patch_distances.sum().item()
        self.patch_distances_num += preds.patch_distances.numel()

    def compute(self) -> DetectionMetrics:
        # precision = self.precision_metric.compute().item()
        # recall = self.recall_metric.compute().item()
        # f1_score = self.f1_metric.compute().item()
        auroc = self.auroc_metric.compute().item()
        ap = self.ap_metric.compute().item()
        pixel_auroc = self.pixel_auroc_metric.compute().item()
        pixel_ap = self.pixel_ap_metric.compute().item()
        anomaly_maps = torch.concat(self.anomaly_maps, dim=0)
        true_masks = torch.concat(self.true_masks, dim=0)
        pixel_aupro = cal_pro_score_gpu(true_masks, anomaly_maps)
        return DetectionMetrics(
            # precision=precision,
            # recall=recall,
            # f1_score=f1_score,
            auroc=auroc,
            ap=ap,
            pixel_auroc=pixel_auroc,
            pixel_aupro=pixel_aupro,
            pixel_ap=pixel_ap,
            patch_distance=self.patch_distances_sum / self.patch_distances_num / 100,
        )


class AACLIPMetricsCalculator(MetricsCalculatorInterface):
    def __init__(self, domain: Literal["Industrial", "Medical"] = "Industrial"):
        self.anomaly_scores: list[float] = []
        self.true_labels: list[bool] = []
        self.anomaly_maps: list[Float[torch.Tensor, "N H W"]] = []
        self.true_masks: list[Bool[torch.Tensor, "N H W"]] = []
        self.pixel_auroc_metric = BinaryAUROC()
        self.pixel_ap_metric = BinaryAUPRC()
        self.domain = domain

    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        self.anomaly_scores.extend(preds.pred_scores.tolist())
        self.true_labels.extend(gts.true_labels.tolist())
        self.anomaly_maps.append(preds.anomaly_maps)
        self.true_masks.append(gts.true_masks)
        pred_score_pixel = torch.tensor(preds.anomaly_maps).flatten()
        true_mask_pixel = torch.tensor(gts.true_masks).flatten()
        self.pixel_auroc_metric.update(pred_score_pixel, true_mask_pixel)
        self.pixel_ap_metric.update(pred_score_pixel, true_mask_pixel)

    def compute(self) -> DetectionMetrics:
        from sklearn.metrics import roc_auc_score, average_precision_score

        # 参考 AA-CLIP/forward_utils.py 的处理方式
        # metrics_eval 函数
        image_preds: Float[np.ndarray, "N"] = np.array(self.anomaly_scores)
        pixel_preds: Float[np.ndarray, "N H W"] = np.concatenate(
            self.anomaly_maps, axis=0
        )
        # 两个归一化必须在拿到所有数据的pred后才能做，因此必须单独为AA-CLIP创建一个
        if pixel_preds.max() != 1:
            pixel_preds = (pixel_preds - pixel_preds.min()) / (
                pixel_preds.max() - pixel_preds.min()
            )
        if image_preds.max() != 1:
            image_preds = (image_preds - image_preds.min()) / (
                image_preds.max() - image_preds.min()
            )
        pmax_pred = pixel_preds.max(axis=(1, 2))
        if self.domain != "Medical":
            image_preds = pmax_pred * 0.5 + image_preds * 0.5
        else:
            image_preds = pmax_pred

        true_labels = torch.tensor(self.true_labels, dtype=torch.bool)
        true_masks = torch.concat(self.true_masks, dim=0)
        auroc = roc_auc_score(true_labels, image_preds)
        assert isinstance(auroc, float)
        ap = average_precision_score(true_labels, image_preds)
        assert isinstance(ap, float)
        pixel_auroc = self.pixel_auroc_metric.compute().item()
        pixel_aupro = cal_pro_score_gpu(true_masks, torch.tensor(pixel_preds))
        pixel_ap = self.pixel_ap_metric.compute().item()
        return DetectionMetrics(
            auroc=auroc,
            ap=ap,
            pixel_auroc=pixel_auroc,
            pixel_aupro=pixel_aupro,
            pixel_ap=pixel_ap,
            patch_distance=0.0,
        )


T_Detector = TypeVar("T_Detector", bound=Detector | TensorDetector)


class MetricsCalculator(MetricsCalculatorInterface):
    def __init__(self, detector_type: Type[T_Detector]):
        # if detector_type is AACLIP:
        #     self.calculator = AACLIPMetricsCalculator()
        # else:
        self.calculator = BaseMetricsCalculator()
        print(f"Using {type(self.calculator).__name__} for metrics calculation.")

    def update(self, preds: DetectionResult, gts: DetectionGroundTruth):
        self.calculator.update(preds, gts)

    def compute(self) -> DetectionMetrics:
        return self.calculator.compute()
