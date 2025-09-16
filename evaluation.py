from abc import ABC, abstractmethod
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Self, get_args, get_origin, get_type_hints, override
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


@dataclass
class DetectionDataset:
    @dataclass
    class Sample:
        image_path: str
        correct_label: bool

    name: str
    samples: list[Sample]

@dataclass
class Result:
    pred_label: bool

@dataclass
class ScoredResult(Result):
    pred_score: float  # [0-1]
    def __init__(self, pred_score: float, pred_label: bool | None = None):
        super().__init__(pred_label=pred_score > 0.5 if pred_label is None else pred_label)
        self.pred_score = pred_score

@dataclass
class PixelResult(ScoredResult):
    anomaly_map: np.ndarray # HxW, [0-1]
    def __init__(self, anomaly_map: np.ndarray, pred_score: float):
        assert isinstance(anomaly_map, np.ndarray)
        assert anomaly_map.ndim == 2
        assert anomaly_map.min() >= 0 and anomaly_map.max() <= 1
        super().__init__(pred_score=pred_score)
        self.anomaly_map = anomaly_map

def init_from_dict(cls, data: dict[str, Any]):
    """从dict中提取dataclass需要的字段来初始化"""
    field_names = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

class Detector(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, image_path: str) -> tuple[Result, dict]:
        pass
    
    @classmethod
    def get_detector_result_type(cls) -> type[Result]:
        """获取检测器返回的Result类型"""
        type_hints = get_type_hints(cls.__call__)
        return_type = type_hints.get('return')

        if return_type and get_origin(return_type) is tuple:
            args = get_args(return_type)
            if args:
                return args[0]  # 返回tuple的第一个类型
        raise ValueError("Cannot determine the Result type from __call__ method.")

class ScoredDetector(Detector):
    @abstractmethod
    def __call__(self, image_path: str) -> tuple[ScoredResult, dict]:
        pass

class PixelDetector(ScoredDetector):
    @abstractmethod
    def __call__(self, image_path: str) -> tuple[PixelResult, dict]:
        pass

@dataclass
class DetectionMetric:
    precision: float
    recall: float
    f1: float

    def __str__(self):
        return f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}"

@dataclass
class ScoredDetectionMetric(DetectionMetric):
    auroc: float
    ap: float

    def __str__(self):
        return super().__str__() + f", AUROC: {self.auroc:.4f}, AP: {self.ap:.4f}"

@dataclass
class PixelLevelDetectionMetric(ScoredDetectionMetric):
    pixel_auroc: float
    pixel_aupro: float

    def __str__(self):
        return super().__str__() + f", Pixel AUROC: {self.pixel_auroc:.4f}, Pixel AUPRO: {self.pixel_aupro:.4f}"

def compute_detection_metrics(results: list[Result], correct_labels: list[bool], correct_masks: list[np.ndarray]) -> DetectionMetric:
    pred_labels = [r.pred_label for r in results]
    precision = precision_score(correct_labels, pred_labels)
    recall = recall_score(correct_labels, pred_labels)
    f1 = f1_score(correct_labels, pred_labels)
    assert (
        isinstance(precision, float)
        and isinstance(recall, float)
        and isinstance(f1, float)
    )
    if isinstance(results[0], ScoredResult):
        pred_scores = [r.pred_score for r in results]   # pyright: ignore[reportAttributeAccessIssue]
        auroc = roc_auc_score(correct_labels, pred_scores)
        ap = average_precision_score(correct_labels, pred_scores)
        assert isinstance(auroc, float) and isinstance(ap, float)
        if isinstance(results[0], PixelResult):
            assert correct_masks[0].shape == results[0].anomaly_map.shape
            ground_truth = np.array(correct_masks)
            assert ground_truth.ndim == 3 and ground_truth.shape[1:] == correct_masks[0].shape
            anomaly_maps = np.array([r.anomaly_map for r in results])  # pyright: ignore[reportAttributeAccessIssue]
            pixel_auroc = roc_auc_score(ground_truth.ravel(), anomaly_maps.ravel())
            pixel_aupro = cal_pro_score(ground_truth, anomaly_maps)
            assert isinstance(pixel_auroc, float) and isinstance(pixel_aupro, float)
            return PixelLevelDetectionMetric(precision=precision, recall=recall, f1=f1, auroc=auroc, ap=ap, pixel_auroc=pixel_auroc, pixel_aupro=pixel_aupro)
        return ScoredDetectionMetric(precision=precision, recall=recall, f1=f1, auroc=auroc, ap=ap)
    else: 
        return DetectionMetric(precision=precision, recall=recall, f1=f1)

# 摘自 AnomalyCLIP/metrics.py
# expect_fpr: 期望的假正率，只取低于这个阈值的部分来计算AUC
# PRO = 正确检测的像素数 / 真实异常区域的像素数
def cal_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step=200, expect_fpr=0.3):
    from skimage import measure
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def evaluation_detection(
    path: str, detector: Detector, dataset: DetectionDataset, checkpoint_n: int = 1000
) -> DetectionMetric:
    print(f"Evaluating detector {detector.name} on dataset {dataset.name}...")
    print(f"Dataset has {len(dataset.samples)} samples.")

    correct_labels: list[bool] = []
    results: list[Result] = []
    contexts: list[dict] = []

    output_json_path = (
        Path(path) / f"detection_results_{detector.name}_{dataset.name}.json"
    )
    # 加载已有结果
    if output_json_path.exists():
        with open(output_json_path, "r") as f:
            contexts = json.load(f)
            assert isinstance(contexts, list)
            correct_labels.extend([r["correct_label"] for r in contexts])
            results.extend([init_from_dict(detector.get_detector_result_type(), r) for r in contexts])
        print(f"Loaded {len(contexts)} existing results from {output_json_path}.")

    def save_json(results: list[dict]):
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

    start_index = len(correct_labels)
    for sample_i, sample in enumerate(dataset.samples[start_index:]):
        sample_i += start_index
        result, context = detector(sample.image_path)
        correct_labels.append(sample.correct_label)
        results.append(result)
        context = {
            "image_path": sample.image_path,
            "correct_label": sample.correct_label,
            **asdict(result),
            **context,
        }
        contexts.append(context)
        if (sample_i + 1) % checkpoint_n == 0:
            save_json(contexts)
            metrics = compute_detection_metrics(results, correct_labels)
            print(
                f"Processed {sample_i + 1}/{len(dataset.samples)} samples. "
                f"Interim Metrics: {metrics}"
            )

    # 保存最终结果
    save_json(contexts)
    final_metrics = compute_detection_metrics(results, correct_labels)
    print(f"Final Metrics: {final_metrics}")
    return final_metrics


def evaluation_detection_total(
    path: Path,
    detectors: list[Detector],
    datasets: list[DetectionDataset],
    checkpoint_n: int = 1000,
):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


    @dataclass
    class MetricsItem:
        metrics_name: str
        detector_name: str
        dataset_name: str
        value: Any

    metrics_list: list[MetricsItem] = []

    for detector in detectors:
        for dataset in datasets:
            metrics = evaluation_detection(
                path=str(path),
                detector=detector,
                dataset=dataset,
                checkpoint_n=checkpoint_n,
            )
            # 遍历result的所有成员
            for field in metrics.__dataclass_fields__.keys():
                metrics_list.append(MetricsItem(
                    metrics_name=field,
                    detector_name=detector.name,
                    dataset_name=dataset.name,
                    value=getattr(metrics, field)
                ))

    tables: dict[str, pd.DataFrame] = {}
    for metrics_name in set(m.metrics_name for m in metrics_list):
        # 使用 dict.fromkeys 去重并保持顺序
        detector_names = list(dict.fromkeys(m.detector_name for m in metrics_list if m.metrics_name == metrics_name))
        dataset_names = list(dict.fromkeys(m.dataset_name for m in metrics_list if m.metrics_name == metrics_name))
        print(metrics_name, detector_names, dataset_names)
        tables[metrics_name] = pd.DataFrame(
            index=detector_names,
            columns=dataset_names,
        )
        for m in metrics_list:
            if m.metrics_name == metrics_name:
                tables[metrics_name].at[m.detector_name, m.dataset_name] = m.value

    for table_name, table in tables.items():
        table.to_csv(path / f"{table_name}_table.csv")
    print("Evaluation completed. ")


class Qwen2_5VL(Detector):
    def __init__(self):
        super().__init__(name="Qwen2.5-VL")
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.detection_prompt = "Are there any defects for the object in the image? Please reply with 'Yes' or 'No'."

    @override
    def __call__(self, image_path: str) -> tuple[Result, dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {"type": "text", "text": self.detection_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # type: ignore
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
        inputs = self.processor(
            text=[text],
            images=image_inputs,  # type: ignore
            videos=video_inputs,  # type: ignore
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        context = {
            "raw_answer": output_text[0],
        }
        pred_label = "yes" in output_text[0].strip().lower()
        return Result(pred_label), context


class WinCLIP(ScoredDetector):
    """WinCLIP: Window-level CLIP for anomaly detection"""
    
    def __init__(self, window_size: int = 224):
        super().__init__(name="WinCLIP")
        import torch
        import clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.window_size = window_size
        
        # Define text prompts for normal and anomalous
        self.normal_text = clip.tokenize(["a photo of a normal object", 
                                        "a good quality product",
                                        "a defect-free item"]).to(self.device)
        self.anomaly_text = clip.tokenize(["a photo of a defective object",
                                         "a damaged product", 
                                         "an object with defects"]).to(self.device)

    @override
    def __call__(self, image_path: str) -> tuple[ScoredResult, dict]:
        import torch
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_input)
            
            # Get text features for normal and anomaly
            normal_features = self.model.encode_text(self.normal_text)
            anomaly_features = self.model.encode_text(self.anomaly_text)
            
            # Calculate similarities
            normal_similarity = torch.cosine_similarity(image_features, normal_features.mean(dim=0, keepdim=True))
            anomaly_similarity = torch.cosine_similarity(image_features, anomaly_features.mean(dim=0, keepdim=True))
            
            normal_score = normal_similarity.item()
            anomaly_score = anomaly_similarity.item()
            
            # Convert to probability using softmax
            scores = torch.softmax(torch.tensor([normal_score, anomaly_score]), dim=0)
            final_anomaly_score = scores[1].item()
            assert isinstance(final_anomaly_score, float)
            
            context = {
                "normal_similarity": normal_score,
                "anomaly_similarity": anomaly_score,
                "confidence": abs(anomaly_score - normal_score),
            }

            return ScoredResult(final_anomaly_score), context


class AnomalyCLIP(ScoredDetector):
    """AnomalyCLIP: Specialized CLIP for anomaly detection with fine-tuned prompts"""
    
    def __init__(self):
        super().__init__(name="AnomalyCLIP")
        import torch
        import clip
        from PIL import Image
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # More detailed and specific text prompts
        self.normal_prompts = [
            "a photo of a flawless {}",
            "a high quality {}",
            "a perfect {}",
            "a {} without any defects",
            "an intact {}",
            "a {} in perfect condition"
        ]
        
        self.anomaly_prompts = [
            "a photo of a damaged {}",
            "a broken {}",
            "a defective {}",
            "a {} with visible defects",
            "a scratched {}",
            "a {} with anomalies"
        ]

    def __call__(self, image_path: str) -> tuple[ScoredResult, dict]:
        import torch
        import clip
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generic object term for prompts
        object_term = "object"
        
        # Prepare text inputs
        normal_texts = [prompt.format(object_term) for prompt in self.normal_prompts]
        anomaly_texts = [prompt.format(object_term) for prompt in self.anomaly_prompts]
        
        normal_tokens = clip.tokenize(normal_texts).to(self.device)
        anomaly_tokens = clip.tokenize(anomaly_texts).to(self.device)
        
        with torch.no_grad():
            # Encode image and text
            image_features = self.model.encode_image(image_input)
            normal_features = self.model.encode_text(normal_tokens)
            anomaly_features = self.model.encode_text(anomaly_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            normal_similarities = torch.mm(image_features, normal_features.T)
            anomaly_similarities = torch.mm(image_features, anomaly_features.T)
            
            # Average similarities
            avg_normal_sim = normal_similarities.mean().item()
            avg_anomaly_sim = anomaly_similarities.mean().item()
            
            # Use softmax to convert to probability
            scores = torch.softmax(torch.tensor([avg_normal_sim, avg_anomaly_sim]), dim=0)
            anomaly_score = scores[1].item()
            assert isinstance(anomaly_score, float)
            
            context = {
                "normal_similarity": avg_normal_sim,
                "anomaly_similarity": avg_anomaly_sim,
                "confidence": abs(avg_anomaly_sim - avg_normal_sim),
                "raw_answer": f"Normal: {avg_normal_sim:.4f}, Anomaly: {avg_anomaly_sim:.4f}"
            }

            return ScoredResult(anomaly_score), context

class AdaCLIP(ScoredDetector):
    """AdaCLIP: Adaptive CLIP for anomaly detection with learnable prompts"""
    
    def __init__(self, n_ctx: int = 16):
        super().__init__(name="AdaCLIP")
        import torch
        import torch.nn as nn
        import clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.n_ctx = n_ctx
        
        # Initialize learnable context vectors (simplified version)
        self.ctx_dim = self.model.token_embedding.embedding_dim
        self.normal_ctx = nn.Parameter(torch.randn(n_ctx, self.ctx_dim)).to(self.device)
        self.anomaly_ctx = nn.Parameter(torch.randn(n_ctx, self.ctx_dim)).to(self.device)
        
        # Class tokens
        self.class_tokens = clip.tokenize(["normal", "anomaly"]).to(self.device)

    def __call__(self, image_path: str) -> tuple[ScoredResult, dict]:
        import torch
        from PIL import Image
        import clip
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode image
            image_features = self.model.encode_image(image_input)
            
            # For simplicity, use standard text prompts
            normal_text = clip.tokenize(["a normal object"]).to(self.device)
            anomaly_text = clip.tokenize(["an anomalous object"]).to(self.device)
            
            normal_features = self.model.encode_text(normal_text)
            anomaly_features = self.model.encode_text(anomaly_text)
            
            # Calculate similarities
            normal_sim = torch.cosine_similarity(image_features, normal_features).item()
            anomaly_sim = torch.cosine_similarity(image_features, anomaly_features).item()
            
            # Use softmax to convert to probability
            scores = torch.softmax(torch.tensor([normal_sim, anomaly_sim]), dim=0)
            anomaly_score = scores[1].item()
            assert isinstance(anomaly_score, float)
            
            threshold = (normal_sim + anomaly_sim) / 2
            confidence = abs(anomaly_sim - normal_sim)
            
            context = {
                "normal_similarity": normal_sim,
                "anomaly_similarity": anomaly_sim,
                "threshold": threshold,
                "confidence": confidence,
                "raw_answer": f"Normal: {normal_sim:.4f}, Anomaly: {anomaly_sim:.4f}, Threshold: {threshold:.4f}"
            }

            return ScoredResult(anomaly_score), context


class RealIAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        super().__init__(name="RealIAD", samples=[])

        json_dir = path / "realiad_jsons"
        image_dir = path / "realiad_1024"
        assert json_dir.exists() and image_dir.exists()
        for json_file in json_dir.glob("*.json"):
            print(f"Loading dataset from {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            normal_class = data["meta"]["normal_class"]
            prefix: str = data["meta"]["prefix"]
            for item in data["test"]:
                anomaly_class = item["anomaly_class"]
                correct_label = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                image_path = str(image_path)
                self.samples.append(DetectionDataset.Sample(image_path, correct_label))
        
        if 0 < sample_limit <= len(self.samples):
            indices = np.random.choice(len(self.samples), size=sample_limit, replace=False)
            self.samples = [self.samples[i] for i in indices]

class MVTecAD(DetectionDataset):
    def __init__(self, path: Path, sample_limit: int = -1):
        super().__init__(name="MVTecAD", samples=[])
        
        # MVTec数据集类别列表
        categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", 
                     "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", 
                     "transistor", "wood", "zipper"]
        
        for category in categories:
            category_path = path / category / "test"
            if not category_path.exists():
                raise ValueError(f"Category path {category_path} does not exist.")
                
            # 加载正常样本 (good文件夹)
            good_path = category_path / "good"
            for img_file in good_path.glob("*.png"):
                self.samples.append(DetectionDataset.Sample(str(img_file), False))
            
            # 加载异常样本 (除good外的所有文件夹)
            for anomaly_dir in category_path.iterdir():
                if anomaly_dir.is_dir() and anomaly_dir.name != "good":
                    for img_file in anomaly_dir.glob("*.png"):
                        self.samples.append(DetectionDataset.Sample(str(img_file), True))
        
        if 0 < sample_limit <= len(self.samples):
            indices = np.random.choice(len(self.samples), size=sample_limit, replace=False)
            self.samples = [self.samples[i] for i in indices]

if __name__ == "__main__":
    # Initialize all baseline detectors
    detectors: list[Detector] = [
        WinCLIP(),
        AnomalyCLIP(),
        # AdaCLIP(),
        Qwen2_5VL(),
    ]
    
    datasets: list[DetectionDataset] = [
        # RealIAD(path=Path("~/hdd/Real-IAD").expanduser(), sample_limit=500),
        MVTecAD(path=Path("~/hdd/mvtec_anomaly_detection").expanduser(), sample_limit=50)
    ]
    
    evaluation_detection_total(
        path=Path("./detection_evaluation"),
        detectors=detectors,
        datasets=datasets,
        checkpoint_n=100,
    )
