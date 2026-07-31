import json
import pandas as pd
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import to_jsonl


def test_to_jsonl_converts_dict_to_json_line():
    result = to_jsonl({"name": "Joao", "score": 10}, ensure_ascii=False)

    assert json.loads(result) == {"name": "Joao", "score": 10}


def test_to_jsonl_converts_dataframe_to_multiple_lines():
    df = pd.DataFrame(
        [
            {"name": "Ana", "score": 1},
            {"name": "Bia", "score": 2},
        ]
    )

    result = to_jsonl(df)

    lines = result.splitlines()
    assert [json.loads(line) for line in lines] == [
        {"name": "Ana", "score": 1},
        {"name": "Bia", "score": 2},
    ]


def test_to_jsonl_writes_and_appends_file(tmp_path):
    output_path = tmp_path / "records.jsonl"

    to_jsonl({"step": 1}, output_path)
    to_jsonl({"step": 2}, output_path, mode="append")

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        '{"step": 1}',
        '{"step": 2}',
    ]


def test_to_jsonl_rejects_invalid_mode():
    with pytest.raises(ValueError):
        to_jsonl({"step": 1}, mode="invalid")


def test_to_jsonl_rejects_unsupported_type():
    with pytest.raises(TypeError):
        to_jsonl(object())
