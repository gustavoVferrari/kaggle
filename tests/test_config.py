import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.config import find_repo_root, resolve_init_path


def test_find_repo_root_from_nested_path(tmp_path):
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert find_repo_root(nested) == str(repo_root)


def test_resolve_init_path_converts_relative_path_to_repo_absolute(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    config = {"init_path": "Classification/Titanic"}

    resolved = resolve_init_path(config, repo_root)

    expected = os.path.abspath(os.path.join(repo_root, "Classification/Titanic"))
    assert resolved["init_path"] == expected


def test_resolve_init_path_keeps_absolute_path(tmp_path):
    absolute_path = os.path.abspath(tmp_path / "project")
    config = {"init_path": absolute_path}

    resolved = resolve_init_path(config, tmp_path)

    assert resolved["init_path"] == absolute_path
