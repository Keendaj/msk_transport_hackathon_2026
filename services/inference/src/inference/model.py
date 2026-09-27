"""Модель прогноза задержки: сеть GRU, заглушка и загрузка весов."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from inference.features import FEATURES, Fleet, build_sequence
from inference.schedule import Schedule
from inference.schemas import Prediction

log = logging.getLogger(__name__)


class DelayPredictorGRU(nn.Module):
    """Сеть из ноутбука обучения: GRU по последовательности и два полносвязных слоя.

    Args:
        input_size: Число признаков на шаге.
        hidden_size: Размер скрытого состояния GRU.
        num_layers: Число слоёв GRU.
        fc_size: Размер скрытого полносвязного слоя.
        dropout_rate: Доля dropout между слоями GRU и в полносвязном блоке.
    """

    def __init__(
        self,
        input_size: int = len(FEATURES),
        hidden_size: int = 128,
        num_layers: int = 3,
        fc_size: int = 32,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прогноз по выходу GRU на последнем шаге.

        Args:
            x: Батч последовательностей формы (batch, шаги, ``input_size``).

        Returns:
            Прогноз задержки в секундах, форма (batch,).
        """
        out, _ = self.gru(x)  # начальное состояние — нули, как в ноутбуке
        return self.fc_block(out[:, -1, :]).squeeze(-1)


class Model:
    """Заглушка: прогноз задержки всегда 0.

    Задаёт интерфейс модели для ``Predictor``.

    Attributes:
        version: Версия модели, пишется в прогноз.
    """

    version = "stub"

    def predict(self, steps: npt.NDArray[np.float32]) -> float:
        """Прогноз задержки на целевой остановке, секунды.

        Args:
            steps: Матрица шаги × признаки, столбцы в порядке ``FEATURES``.
        """
        return 0.0


class TorchModel(Model):
    """DelayPredictorGRU с весами из state_dict, сохранённого torch.save.

    Версия модели — имя файла весов.

    Args:
        path: Файл весов.

    Raises:
        ValueError: Сеть обучена на другом числе признаков.
    """

    def __init__(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Размеры берутся из весов, чтобы переобученная сеть другого размера грузилась без правок
        input_size = state["gru.weight_ih_l0"].shape[1]
        if input_size != len(FEATURES):
            raise ValueError(f"Model expects {input_size} features, service builds {len(FEATURES)}")
        self._net = DelayPredictorGRU(
            input_size=input_size,
            hidden_size=state["gru.weight_hh_l0"].shape[1],
            num_layers=sum(key.startswith("gru.weight_ih_l") for key in state),
            fc_size=state["fc_block.0.weight"].shape[0],
        )
        self._net.load_state_dict(state)
        self._net.eval()
        self.version = path.name

    def predict(self, steps: npt.NDArray[np.float32]) -> float:
        """Прогноз задержки на целевой остановке, секунды."""
        with torch.inference_mode():
            return float(self._net(torch.from_numpy(steps).unsqueeze(0))[0])


def load_model(path: Path | None) -> Model:
    """Загружает веса из ``path``, без пути возвращает заглушку."""
    if path is None:
        log.warning("INFERENCE_MODEL_PATH is not set, using the stub model")
        return Model()
    # Сеть маленькая, прогнозы по одному: лишние потоки только тратят CPU
    torch.set_num_threads(1)
    model = TorchModel(path)
    log.info(f"Model {model.version} loaded from {path}")
    return model


class Predictor:
    """Модель вместе с состоянием потока: расписанием и точками всех ТС.

    Args:
        model: Модель прогноза.
        schedule: Плановое расписание.
        fleet: Точки всех ТС, по умолчанию пустые.

    Attributes:
        model: Модель прогноза.
        schedule: Плановое расписание.
        fleet: Точки всех ТС, их пополняет ``TelemetryConsumer``.
    """

    def __init__(self, model: Model, schedule: Schedule, fleet: Fleet | None = None) -> None:
        self.model = model
        self.schedule = schedule
        self.fleet = fleet if fleet is not None else Fleet()

    def predict(self, unit_id: int, at: datetime) -> Prediction:
        """Прогноз задержки на целевой остановке на момент at, или MissingData.

        Raises:
            MissingData: Для прогноза не хватает данных, см. ``build_sequence``.
        """
        sequence = build_sequence(self.fleet, self.schedule, unit_id, at)
        return Prediction(
            unit_id=unit_id,
            at=at,
            score=self.model.predict(sequence.steps),
            model_version=self.model.version,
            target_stop_id=sequence.target.stop_id,
            target_planned_at=sequence.target.planned_at,
            cur_dev_s=sequence.cur_dev_s,
        )
