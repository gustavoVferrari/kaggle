import os
from pathlib import Path


def find_repo_root(start_path: str | os.PathLike | None = None) -> str:
    """Return the nearest parent directory containing .git."""
    path = Path(start_path or __file__).resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return str(candidate)

    return str(path)


def resolve_init_path(config: dict, repo_root: str | os.PathLike | None = None) -> dict:
    """Resolve config['init_path'] against the repository root when relative."""
    init_path = config.get("init_path")
    if not init_path or os.path.isabs(init_path):
        return config

    root = find_repo_root(repo_root)
    config["init_path"] = os.path.abspath(os.path.join(root, init_path))
    return config
