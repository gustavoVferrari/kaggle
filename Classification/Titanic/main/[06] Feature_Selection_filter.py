import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Classification.Titanic.src.utils.config import load_config
from functions.feature_selection import FeatureSelectionOrchestrator
from utils.plots import Pearson_correlation, Bar_plot
from utils.utils import to_jsonl
from utils.logs import setup_logging


def Main_Feature_Selection(pipeline_name: str, config: dict):
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)

    try:
        logger.info("Starting feature selection filter pipeline: %s", pipeline_name)

        x_train_path = os.path.join(
            config['init_path'],
            config['data']['feature_eng'],
            f"X_train_feat_eng_{pipeline_name}.parquet",
        )

        y_train_path = os.path.join(
            config['init_path'],
            config['data']['feature_eng'],
            f"y_train_feat_eng_{pipeline_name}.parquet",
        )

        logger.info("Loading training features from: %s", x_train_path)
        X_train = pd.read_parquet(x_train_path)
        logger.info("Training features loaded with shape: %s", X_train.shape)

        logger.info("Loading training target from: %s", y_train_path)
        y_train = pd.read_parquet(y_train_path)
        logger.info("Training target loaded with shape: %s", y_train.shape)

        logger.info("Initializing feature selection orchestrator")
        feature_selection = FeatureSelectionOrchestrator()

        logger.info("Running ANOVA feature selection")
        Anova = feature_selection.apply("Anova", X_train, y_train)
        logger.info("ANOVA completed with %s scored features", len(Anova))

        logger.info("Running mutual information feature selection")
        mi = feature_selection.apply("MutualInformationClassif", X_train, y_train)
        logger.info("Mutual information completed with %s scored features", len(mi))

        logger.info("Running Pearson correlation analysis")
        corr = feature_selection.apply("PearsonCorrelation", X_train, y_train)
        logger.info("Pearson correlation completed with shape: %s", corr.shape)

        logger.info("Running smart correlated feature selection")
        smart_corr = feature_selection.apply("SmartCorrelatedSelection", X_train, y_train)
        logger.info(
            "Smart correlated selection completed | correlated features: %s | features to drop: %s",
            len(smart_corr.get("corr_feature", [])),
            len(smart_corr.get("corr_2_drop", [])),
        )

        path_sc = os.path.join(
            config['init_path'],
            config['reports']['tables'],
            f"corr_features_{pipeline_name}.jsonl",
        )

        logger.info("Saving correlated feature selection table to: %s", path_sc)
        to_jsonl(smart_corr, path_sc, mode='append')

        path_ = os.path.join(
            config['init_path'],
            config['reports']['plots'],
        )

        logger.info("Saving Pearson correlation plot to: %s", path_)
        Pearson_correlation(corr, title=f"corr_{pipeline_name}", path=path_)

        logger.info("Saving ANOVA plot to: %s", path_)
        Bar_plot(Anova, title=f"Anova_{pipeline_name}", path=path_)

        logger.info("Saving mutual information plot to: %s", path_)
        Bar_plot(mi, title=f"Mutual_information_{pipeline_name}", path=path_)

        logger.info("Feature selection filter pipeline completed: %s", pipeline_name)
    except Exception:
        logger.exception("Feature selection filter pipeline failed: %s", pipeline_name)
        raise


def main():
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)

    try:
        logger.info("Starting feature selection filter process")
        config = load_config(['config'])
        logger.info("Configuration loaded")

        Main_Feature_Selection(pipeline_name="Pipeline3", config=config)
        Main_Feature_Selection(pipeline_name="Pipeline2", config=config)
        Main_Feature_Selection(pipeline_name="Pipeline1", config=config)

        logger.info("Feature selection filter process completed")
    except Exception:
        logger.exception("Feature selection filter process failed")
        raise


if __name__ == "__main__":
    main()
