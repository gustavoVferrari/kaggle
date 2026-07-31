import pandas as pd
from pandas.testing import assert_frame_equal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.make_dataset import clean_data, load_data, save_data, split_data


def test_clean_data_returns_input_dataframe():
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

    result = clean_data(df)

    assert result is df


def test_split_data_separates_features_and_target():
    df = pd.DataFrame(
        {
            "feature_a": range(10),
            "feature_b": range(10, 20),
            "target": [0, 1] * 5,
        }
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="target",
        test_size=0.3,
        random_state=42,
    )

    assert list(X_train.columns) == ["feature_a", "feature_b"]
    assert list(X_test.columns) == ["feature_a", "feature_b"]
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3


def test_save_and_load_data_roundtrip(tmp_path):
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    save_data(tmp_path, "sample", df)
    result = load_data(tmp_path / "sample.parquet")

    assert_frame_equal(result, df)
