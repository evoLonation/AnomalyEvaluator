from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, override
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


@dataclass
class DetectionDataset:
    @dataclass
    class Sample:
        image_path: str
        correct_label: bool

    name: str
    samples: list[Sample]


class Detector(ABC):
    def __init__(self, name: str):
        self.name = name

    # output: (pred_label, context)
    @abstractmethod
    def __call__(self, image_path: str) -> tuple[bool, dict]:
        pass


@dataclass
class DetectionResult:
    precision: float
    recall: float
    f1: float

    def __str__(self):
        return f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}"


def evaluation_detection(
    path: str, detector: Detector, dataset: DetectionDataset, checkpoint_n: int = 1000
) -> DetectionResult:
    print(f"Evaluating detector {detector.name} on dataset {dataset.name}...")
    print(f"Dataset has {len(dataset.samples)} samples.")

    correct_labels: list[bool] = []
    pred_labels: list[bool] = []
    results: list[dict] = []

    output_json_path = (
        Path(path) / f"detection_results_{detector.name}_{dataset.name}.json"
    )
    # 加载已有结果
    if output_json_path.exists():
        with open(output_json_path, "r") as f:
            results = json.load(f)
            assert isinstance(results, list)
            correct_labels.extend([r["correct_label"] for r in results])
            pred_labels.extend([r["pred_label"] for r in results])
        print(f"Loaded {len(results)} existing results from {output_json_path}.")

    def save_checkpoint(
        image_paths: list[str],
        correct_labels: list[bool],
        pred_labels: list[bool],
        contexts: list[dict],
    ):
        with open(output_json_path, "w") as f:
            json.dump(
                [
                    {"image_path": ip, "correct_label": cl, "pred_label": pl, **ctx}
                    for ip, cl, pl, ctx in zip(
                        image_paths, correct_labels, pred_labels, contexts
                    )
                ],
                f,
                indent=4,
            )

    def save_json(results: list[dict]):
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

    def compute_metrics(correct_labels, pred_labels) -> DetectionResult:
        precision = precision_score(correct_labels, pred_labels)
        recall = recall_score(correct_labels, pred_labels)
        f1 = f1_score(correct_labels, pred_labels)
        assert (
            isinstance(precision, float)
            and isinstance(recall, float)
            and isinstance(f1, float)
        )
        return DetectionResult(precision=precision, recall=recall, f1=f1)

    start_index = len(correct_labels)
    for sample_i, sample in enumerate(dataset.samples[start_index:]):
        sample_i += start_index
        pred_label, context = detector(sample.image_path)
        correct_labels.append(sample.correct_label)
        pred_labels.append(pred_label)
        result = {
            "image_path": sample.image_path,
            "correct_label": sample.correct_label,
            "pred_label": pred_label,
            **context,
        }
        results.append(result)
        if (sample_i + 1) % checkpoint_n == 0:
            save_json(results)
            interim_result = compute_metrics(correct_labels, pred_labels)
            print(
                f"Processed {sample_i + 1}/{len(dataset.samples)} samples. "
                f"Interim Result: {interim_result}"
            )

    # 保存最终结果
    save_json(results)
    final_result = compute_metrics(correct_labels, pred_labels)
    print(f"Result: {final_result}")
    return final_result


def evaluation_detection_total(
    path: Path,
    detectors: list[Detector],
    datasets: list[DetectionDataset],
    checkpoint_n: int = 1000,
):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # 每个指标单独一个表，每行是不同模型，每列是不同指标
    precision_table = pd.DataFrame(
        index=[detector.name for detector in detectors],
        columns=[dataset.name for dataset in datasets],
    )
    recall_table = pd.DataFrame(
        index=[detector.name for detector in detectors],
        columns=[dataset.name for dataset in datasets],
    )
    f1_table = pd.DataFrame(
        index=[detector.name for detector in detectors],
        columns=[dataset.name for dataset in datasets],
    )
    for detector in detectors:
        for dataset in datasets:
            result = evaluation_detection(
                path=str(path),
                detector=detector,
                dataset=dataset,
                checkpoint_n=checkpoint_n,
            )
            precision_table.loc[detector.name, dataset.name] = result.precision
            recall_table.loc[detector.name, dataset.name] = result.recall
            f1_table.loc[detector.name, dataset.name] = result.f1
    precision_table.to_csv(path / "precision_table.csv")
    recall_table.to_csv(path / "recall_table.csv")
    f1_table.to_csv(path / "f1_table.csv")
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
    def __call__(self, image_path: str) -> tuple[bool, dict]:
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
        return pred_label, context


class RealIAD(DetectionDataset):
    def __init__(self, path: Path):
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


if __name__ == "__main__":
    detectors: list[Detector] = [Qwen2_5VL()]
    datasets: list[DetectionDataset] = [
        RealIAD(path=Path("~/hdd/Real-IAD").expanduser())
    ]
    evaluation_detection_total(
        path=Path("./detection_evaluation"),
        detectors=detectors,
        datasets=datasets,
        checkpoint_n=100,
    )
