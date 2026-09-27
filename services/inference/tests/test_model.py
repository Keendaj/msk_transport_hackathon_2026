import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.features import FEATURES, SEQ_LEN, MissingData, TrackPoint
from inference.model import DelayPredictorGRU, Model, Predictor, load_model
from inference.schedule import Schedule, load_schedule

MODEL_V1 = Path(__file__).parents[3] / "models" / "v1"
T0 = datetime(2026, 1, 6, 10, 0, tzinfo=UTC)
LON, LAT = 37.6, 55.75


def steps(cur_dev: float, speed: float, dist_from: float, dist_to: float, neighbors: int = 0):
    x = np.zeros((SEQ_LEN, len(FEATURES)), np.float32)
    x[:, 0], x[:, 1], x[:, 3] = cur_dev, speed, 90
    x[:, 2] = np.linspace(dist_from, dist_to, SEQ_LEN)
    if neighbors:
        x[:, 4], x[:, 5] = speed, neighbors
    return x


def save_net(path: Path, **sizes: int) -> DelayPredictorGRU:
    torch.manual_seed(0)
    net = DelayPredictorGRU(**sizes)
    torch.save(net.state_dict(), path)
    return net.eval()


def test_stub_model_without_path() -> None:
    model = load_model(None)
    assert type(model) is Model and model.version == "stub"
    assert model.predict(steps(120, 20, 3000, 1500)) == 0.0


@pytest.mark.skipif(not MODEL_V1.exists(), reason="models/v1 is not in the checkout")
@pytest.mark.parametrize(
    ("x", "expected"),
    [
        # Посчитано классом DelayPredictorGRU из ноутбука обучения
        (steps(120, 20, 3000, 1500), 113.1147),
        (steps(0, 20, 3000, 1500), 0.6706),
        (steps(120, 5, 3000, 2800, neighbors=3), 184.135),
    ],
)
def test_model_v1_matches_notebook(x: np.ndarray, expected: float) -> None:
    model = load_model(MODEL_V1)
    assert model.version == "v1"
    assert model.predict(x) == pytest.approx(expected, abs=1e-3)


def test_loads_network_of_other_size(tmp_path: Path) -> None:
    net = save_net(tmp_path / "small", hidden_size=16, num_layers=2, fc_size=8)
    x = np.random.default_rng(0).normal(size=(SEQ_LEN, len(FEATURES))).astype(np.float32)
    with torch.no_grad():
        expected = float(net(torch.from_numpy(x)[None])[0])
    assert load_model(tmp_path / "small").predict(x) == pytest.approx(expected, abs=1e-5)


def test_rejects_other_feature_count(tmp_path: Path) -> None:
    save_net(tmp_path / "wide", input_size=9)
    with pytest.raises(ValueError, match="expects 9 features"):
        load_model(tmp_path / "wide")


class RecordingModel(Model):
    version = "recording"

    def predict(self, steps: np.ndarray) -> float:
        self.steps = steps
        return 42.0


def test_predictor(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.csv"
    schedule.write_text(
        "tt_action_item_id,time_begin,order_date,tr_id,geom\n"
        f"77,2026-01-06 10:12:00,2026-01-06,700,POINT ({LON} {LAT + 0.01})\n"
    )
    units = tmp_path / "units.csv"
    units.write_text("unit_id,tr_id\n1,700\n")
    model = RecordingModel()
    predictor = Predictor(model, load_schedule(schedule, units))
    for i in range(SEQ_LEN):
        predictor.fleet.add(TrackPoint(1, T0.timestamp() - 15 * i, LON, LAT, 20.0, 0.0))

    prediction = predictor.predict(1, T0)

    assert model.steps.shape == (SEQ_LEN, len(FEATURES))
    assert (prediction.unit_id, prediction.at, prediction.score) == (1, T0, 42.0)
    assert (prediction.model_version, prediction.cur_dev_s) == ("recording", 0.0)
    assert prediction.target_stop_id == 77
    assert prediction.target_planned_at == datetime(2026, 1, 6, 10, 12, tzinfo=UTC)
    assert math.isclose(model.steps[-1, 2], 0.01 * 6367000 * math.pi / 180, rel_tol=1e-6)


def test_predictor_without_data() -> None:
    with pytest.raises(MissingData):
        Predictor(Model(), Schedule({}, {})).predict(1, T0)
