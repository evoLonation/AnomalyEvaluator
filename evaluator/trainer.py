from abc import abstractmethod
import json
from dataclasses_json import DataClassJsonMixin
import torch
import torch.nn as nn
from dataclasses import asdict, dataclass, field
from pathlib import Path
import datetime
import pytz
from tqdm import tqdm
from typing import Callable, Dict, Any, Tuple, overload, TypeVar, Generic
from data.utils import ImageSize
import evaluator.reproducibility as repro
from .checkpoint import StateMapper, TrainCheckpointState
import dacite


@dataclass
class BaseTrainConfig(DataClassJsonMixin):
    lr: float = 1e-3
    batch_size: int = 8
    image_resize: int = 512
    centercrop: ImageSize = field(default_factory=lambda: ImageSize.square(512))
    num_epochs: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ConfigT = TypeVar("ConfigT", bound=BaseTrainConfig)
ModelT = TypeVar("ModelT", bound=nn.Module)


# --- 全局状态类 (通用) ---
@dataclass
class GlobalTrainState(Generic[ConfigT]):
    config: ConfigT
    done: bool = False
    trained_epoch: int = 0
    # 使用字典存储各种 loss 历史，扩展性更强
    history: Dict[str, list] = field(default_factory=dict)

    @staticmethod
    def get_save_path(result_dir: Path) -> Path:
        return result_dir / "total_state.json"

    def save(self, result_dir: Path):
        self.get_save_path(result_dir).write_text(json.dumps(asdict(self), indent=4))

    @staticmethod
    def load(result_dir: Path, real_config_type: type) -> "GlobalTrainState[ConfigT]":
        return dacite.from_dict(
            data_class=GlobalTrainState[real_config_type],
            data=json.loads(GlobalTrainState.get_save_path(result_dir).read_text()),
        )


# --- 抽象训练器基类 ---
class BaseTrainer(Generic[ConfigT, ModelT]):
    base_dir: Path = Path("results/train_12_13")
    model_type: type
    config_type: type

    @overload
    def __init__(
        self,
        config: ConfigT,
        name: str | None = None,
    ): ...
    @overload
    def __init__(
        self,
        resume_name: str,
    ): ...

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        if "config" in kwargs or (
            len(args) > 0 and isinstance(args[0], BaseTrainConfig)
        ):
            config = kwargs.get("config", args[0] if len(args) > 0 else None)
            name = kwargs.get("name", args[1] if len(args) > 1 else None)
            self._new(config=config, name=name)
        else:
            resume_name = kwargs.get("resume_name", args[0] if len(args) > 0 else None)
            self._new(resume_name=resume_name)

    def _new(
        self,
        config: ConfigT | None = None,
        name: str | None = None,
        resume_name: str | None = None,
    ):
        if config is not None:
            self.state: GlobalTrainState[ConfigT] = GlobalTrainState(config)
            self.result_dir = self.gen_result_dir(name)
            # 检查是否已经存在，如果存在且已经训练了至少一个回合则报错
            if self.state.get_save_path(self.result_dir).exists():
                loaded_state = GlobalTrainState.load(self.result_dir, self.config_type)
                if loaded_state.trained_epoch > 0:
                    raise ValueError(
                        f"Result directory {self.result_dir} already exists and has trained epochs."
                    )
                else:
                    print(
                        f"Warning: Result directory {self.result_dir} already exists but has no trained epochs. Overwriting."
                    )
            self.result_dir.mkdir(parents=True, exist_ok=True)
            self.state.save(self.result_dir)
            self.resume_dir = None
        else:
            assert resume_name is not None
            self.resume_dir = self.gen_result_dir(resume_name)
            self.state: GlobalTrainState[ConfigT] = GlobalTrainState.load(
                self.resume_dir, self.config_type
            )
            self.result_dir = self.resume_dir
            self.result_dir.mkdir(parents=True, exist_ok=True)

        self.config = self.state.config
        repro.init(self.config.seed)

        # 将由子类初始化
        self.model: ModelT | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def get_result_dir(self) -> Path:
        return self.result_dir

    @classmethod
    def gen_result_dir(cls, name: str | None) -> Path:
        if name is None:
            now = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
            name = now.strftime("%m.%d_%H:%M:%S")
        return cls.base_dir / name

    @classmethod
    @abstractmethod
    def setup_model(cls, config: ConfigT) -> ModelT:
        """子类必须实现：返回模型实例"""
        ...

    @classmethod
    @abstractmethod
    def setup_optimizer(cls, config: ConfigT, model: ModelT) -> torch.optim.Optimizer:
        """子类必须实现：返回优化器"""
        ...

    @abstractmethod
    def setup_other_components(self):
        """子类可选实现：设置其他组件，如数据集、数据加载器等"""
        ...

    @abstractmethod
    def train_one_epoch(self, epoch: int, model: ModelT) -> Dict[str, float]:
        """子类必须实现：执行一个Epoch的训练，返回 log metrics 字典"""
        ...

    @classmethod
    def get_trained_model(
        cls, name: str, epoch: int, mapper: StateMapper | None = None
    ) -> ModelT:
        """工具方法：加载指定训练结果的模型"""
        result_dir = cls.gen_result_dir(name)
        config: ConfigT = GlobalTrainState.load(result_dir, cls.config_type).config
        model = cls.setup_model(config)
        TrainCheckpointState.load_ckpt(
            result_dir,
            epoch,
            model=model,
            strict=False,
            model_mapper=mapper if mapper is not None else lambda x: x,
        )
        return model

    def optimize_step(self, loss: torch.Tensor):
        """工具方法：执行标准优化步骤"""
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _clear_epoch_loss_history(self):
        if hasattr(self, "epoch_loss_history"):
            del self.epoch_loss_history
            del self.last_loss_idx

    def record_loss(self, loss_dict: dict[str, float]):
        if not hasattr(self, "epoch_loss_history"):
            self.epoch_loss_history = {}
            self.last_loss_idx = 0
            for key in loss_dict:
                self.epoch_loss_history[key] = []
        for key in loss_dict:
            self.epoch_loss_history[key].append(loss_dict[key])

    def print_loss(self):
        for key, values in self.epoch_loss_history.items():
            avg_value = sum(values[self.last_loss_idx :]) / len(
                values[self.last_loss_idx :]
            )
            print(f"{key:15}: {avg_value:.4f}")
        self.last_loss_idx = len(next(iter(self.epoch_loss_history.values())))

    def compute_total_avg_loss(self) -> dict[str, float]:
        avg_losses = {}
        for key, values in self.epoch_loss_history.items():
            avg_value = sum(values) / len(values)
            avg_losses[key] = avg_value
        return avg_losses

    def run(self):
        """通用的训练主循环"""
        self.model = self.setup_model(self.config)
        self.optimizer = self.setup_optimizer(self.config, self.model)
        self.setup_other_components()

        # 断点加载逻辑
        if self.resume_dir is not None:
            start_epoch = self.state.trained_epoch
            TrainCheckpointState.load_ckpt(
                self.resume_dir, start_epoch, self.model, self.optimizer, {}
            )
        else:
            start_epoch = 0
            TrainCheckpointState.save_ckpt(
                self.result_dir, 0, self.model, self.optimizer, {}
            )

        # Epoch 循环
        for epoch in tqdm(
            range(start_epoch + 1, self.config.num_epochs + 1),
            initial=start_epoch,
            total=self.config.num_epochs,
            desc="Epoch",
        ):

            # 执行用户定义的具体训练逻辑
            self._clear_epoch_loss_history()
            metrics = self.train_one_epoch(epoch, self.model)

            # 打印日志
            log_str = f"Epoch {epoch}"
            for k, v in metrics.items():
                log_str += f", {k}: {v:.4f}"
            tqdm.write(log_str)

            # 更新状态和保存
            with repro.RNGStateChecker():
                TrainCheckpointState.save_ckpt(
                    self.result_dir, epoch, self.model, self.optimizer, {}
                )
                self.state.trained_epoch = epoch
                for k, v in metrics.items():
                    if k not in self.state.history:
                        self.state.history[k] = []
                    self.state.history[k].append(v)

                if epoch == self.config.num_epochs:
                    self.state.done = True
                self.state.save(self.result_dir)
