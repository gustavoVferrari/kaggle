import logging
import os
import sys
from datetime import datetime

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Regression.house_prices.src.utils.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


try:
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    EVIDENTLY_API = "new"
except ModuleNotFoundError as exc:
    if exc.name != "evidently":
        raise
    raise ModuleNotFoundError(
        "O pacote 'evidently' nao esta instalado. Instale com: pip install evidently"
    ) from exc
except ImportError:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    EVIDENTLY_API = "legacy"


def build_path(config: dict, *parts: str) -> str:
    return os.path.join(config["init_path"], *parts)


def get_monitoring_columns(reference: pd.DataFrame, current: pd.DataFrame, config_pipe: dict) -> list:
    preprocessing_config = config_pipe["features"]["preprocessing"]
    target_columns = set(config_pipe["features"].get("target", []))
    dropped_columns = set(preprocessing_config.get("cols_2_drop", []))

    configured_columns = (
        preprocessing_config.get("cat_var", [])
        + preprocessing_config.get("num_dis_1", [])
        + preprocessing_config.get("num_con_1", [])
    )

    common_columns = set(reference.columns).intersection(current.columns)
    selected_columns = [
        column
        for column in configured_columns
        if column in common_columns
        and column not in target_columns
        and column not in dropped_columns
    ]

    if selected_columns:
        return selected_columns

    logger.warning(
        "Nenhuma coluna configurada foi encontrada nos dois datasets. Usando todas as colunas em comum."
    )
    return sorted(common_columns - target_columns - dropped_columns)


def save_report(report, result, html_path: str, json_path: str) -> None:
    if hasattr(result, "save_html"):
        result.save_html(html_path)
    elif hasattr(report, "save_html"):
        report.save_html(html_path)
    else:
        report.save(html_path)

    if hasattr(result, "json"):
        with open(json_path, "w", encoding="utf-8") as file:
            file.write(result.json())
    elif hasattr(report, "json"):
        with open(json_path, "w", encoding="utf-8") as file:
            file.write(report.json())
    elif hasattr(report, "save"):
        report.save(json_path)


def run_data_drift_monitoring(config: dict, config_pipe: dict) -> tuple[str, str]:
    logger.info("Iniciando monitoramento Evidently: api=%s", EVIDENTLY_API)

    reference_path = build_path(
        config,
        config["data"]["processed"],
        "train_features.parquet"
    )
    current_path = build_path(
        config,
        config["data"]["processed"],
        "test_features.parquet"
    )

    logger.info("Carregando dataset de referencia: %s", reference_path)
    reference = pd.read_parquet(reference_path)
    logger.info("Carregando dataset atual: %s", current_path)
    current = pd.read_parquet(current_path)

    monitoring_columns = get_monitoring_columns(reference, current, config_pipe)
    reference = reference[monitoring_columns].copy()
    current = current[monitoring_columns].copy()
    logger.info(
        "Datasets preparados: reference=%s, current=%s, columns=%s",
        reference.shape,
        current.shape,
        len(monitoring_columns)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = build_path(config, "reports", "evidently")
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"data_drift_{timestamp}.html")
    json_path = os.path.join(output_dir, f"data_drift_{timestamp}.json")

    if EVIDENTLY_API == "new":
        report = Report([DataDriftPreset(), DataSummaryPreset()])
        result = report.run(current, reference)
    else:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report

    save_report(report, result, html_path, json_path)
    logger.info("Relatorio Evidently salvo em: %s", html_path)
    logger.info("Snapshot Evidently salvo em: %s", json_path)

    return html_path, json_path


def main() -> None:
    logger.info("Carregando configuracoes: config, config_pipe.")
    config, config_pipe = load_config(load_all=["config", "config_pipe"])
    logger.info("Configuracoes carregadas com sucesso.")
    run_data_drift_monitoring(config, config_pipe)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no monitoramento Evidently")
        raise
