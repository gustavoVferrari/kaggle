import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from functions.predict_model import make_prediction, make_prediction_reg


class DummyClassifier:
    def predict(self, input_data):
        return np.array([0, 1, 1])

    def predict_proba(self, input_data):
        return np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )


class DummyRegressor:
    def predict(self, input_data):
        return np.array([10.5, 20.0, 30.25])


def test_make_prediction_returns_predictions_and_positive_probabilities():
    input_data = pd.DataFrame({"feature": [1, 2, 3]}, index=["a", "b", "c"])

    predictions, probabilities = make_prediction(DummyClassifier(), input_data)

    expected_predictions = pd.DataFrame(
        {"prediction": [0, 1, 1]},
        index=["a", "b", "c"],
    )
    expected_probabilities = pd.DataFrame(
        {"probability": [0.2, 0.7, 0.9]},
        index=["a", "b", "c"],
    )
    assert_frame_equal(predictions, expected_predictions, check_dtype=False)
    assert_frame_equal(probabilities, expected_probabilities)


def test_make_prediction_reg_returns_prediction_dataframe():
    input_data = pd.DataFrame({"feature": [1, 2, 3]}, index=["a", "b", "c"])

    predictions = make_prediction_reg(DummyRegressor(), input_data)

    expected = pd.DataFrame(
        {"prediction": [10.5, 20.0, 30.25]},
        index=["a", "b", "c"],
    )
    assert_frame_equal(predictions, expected)
