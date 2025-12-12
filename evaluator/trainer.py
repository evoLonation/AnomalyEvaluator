from abc import abstractmethod
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
import dataclasses_json
from pathlib import Path
import datetime
import pytz
from tqdm import tqdm
from typing import Dict, Any, Tuple, overload
from data.utils import ImageSize
import evaluator.reproducibility as repro
from .checkpoint import TrainCheckpointState


@dataclass
class TrainConfig(DataClassJsonMixin):
    lr: float = 1e-3
    batch_size: int = 8
    image_resize: int = 512
    centercrop: ImageSize = field(default_factory=lambda: ImageSize.square(512))
    num_epochs: int = 30
    seed: int = 42
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metadata=dataclasses_json.config(
            encoder=lambda d: str(d),
            decoder=lambda s: torch.device(s),
        ),
    )


# --- 全局状态类 (通用) ---
@dataclass
class GlobalTrainState(DataClassJsonMixin):
    config: TrainConfig
    done: bool = False
    trained_epoch: int = 0
    # 使用字典存储各种 loss 历史，扩展性更强
    history: Dict[str, list] = field(default_factory=dict)

    def save(self, result_dir: Path):
        (result_dir / "total_state.json").write_text(self.to_json(indent=4))

    @staticmethod
    def load(result_dir: Path) -> "GlobalTrainState":
        return GlobalTrainState.from_json((result_dir / "total_state.json").read_text())


# --- 抽象训练器基类 ---
class BaseTrainer:
    base_dir: Path = Path("results/train")

    @overload
    def __init__(
        self,
        config: TrainConfig,
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
        if "config" in kwargs or (len(args) > 0 and isinstance(args[0], TrainConfig)):
            config = kwargs.get("config", args[0] if len(args) > 0 else None)
            name = kwargs.get("name", args[1] if len(args) > 1 else None)
            self._new(config=config, name=name)
        else:
            resume_name = kwargs.get("resume_name", args[0] if len(args) > 0 else None)
            self._new(resume_name=resume_name)

    def _new(
        self,
        config: TrainConfig | None = None,
        name: str | None = None,
        resume_name: str | None = None,
    ):
        if config is not None:
            self.state = GlobalTrainState(config)
            self.result_dir = self._get_new_result_dir(name)
            self.resume_dir = None
        else:
            assert resume_name is not None
            self.resume_dir = self._get_resume_result_dir(resume_name)
            self.state = GlobalTrainState.load(self.resume_dir)
            self.result_dir = self.resume_dir

        self.config = self.state.config
        self.result_dir.mkdir(parents=True, exist_ok=True)
        repro.init(self.config.seed)

        # 将由子类初始化
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    @classmethod
    def _get_new_result_dir(cls, name: str | None) -> Path:
        if name is None:
            now = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
            name = now.strftime("%m.%d_%H:%M:%S")
        return cls.base_dir / name

    @classmethod
    def _get_resume_result_dir(cls, name: str) -> Path:
        return cls.base_dir / name

    @classmethod
    @abstractmethod
    def setup_model(cls, config: TrainConfig) -> nn.Module:
        """子类必须实现：返回模型实例"""
        ...

    @classmethod
    @abstractmethod
    def setup_optimizer(
        cls, config: TrainConfig, model: nn.Module
    ) -> torch.optim.Optimizer:
        """子类必须实现：返回优化器"""
        ...

    @abstractmethod
    def setup_other_components(self):
        """子类可选实现：设置其他组件，如数据集、数据加载器等"""
        ...

    @abstractmethod
    def train_one_epoch(self, epoch: int, model: nn.Module) -> Dict[str, float]:
        """子类必须实现：执行一个Epoch的训练，返回 log metrics 字典"""
        ...

    @classmethod
    def get_trained_model(cls, name: str, epoch: int) -> nn.Module:
        """工具方法：加载指定训练结果的模型"""
        result_dir = cls._get_resume_result_dir(name)
        config = GlobalTrainState.load(result_dir).config
        model = cls.setup_model(config)
        TrainCheckpointState.load_ckpt(result_dir, epoch, model=model, strict=False)
        return model

    def optimize_step(self, loss: torch.Tensor):
        """工具方法：执行标准优化步骤"""
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        """通用的训练主循环"""
        self.model = self.setup_model(self.config)
        self.optimizer = self.setup_optimizer(self.config, self.model)
        self.setup_other_components()

        # 断点加载逻辑
        start_epoch = 0
        with repro.RNGStateChecker():
            if self.resume_dir is not None:
                start_epoch = self.state.trained_epoch
                TrainCheckpointState.load_ckpt(
                    self.resume_dir, start_epoch, self.model, self.optimizer, {}
                )
            else:
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
