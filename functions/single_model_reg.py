from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform


def _prefix_params(step_name, params):
    if isinstance(params, list):
        return [{f"{step_name}__{k}": v for k, v in grid.items()} for grid in params]
    return {f"{step_name}__{k}": v for k, v in params.items()}


def _build_config(model, model_name, param_grid=None, param_distributions=None, step_name=None):
    step_name = step_name or model.__class__.__name__.lower()
    config = {
        "model": model,
        "model_name": model_name,
    }

    if param_grid is not None:
        config["param_grid"] = param_grid
        config["models_gs"] = {
            step_name: [
                model,
                _prefix_params(step_name, param_grid),
            ]
        }

    if param_distributions is not None:
        config["param_distributions"] = param_distributions

    return config


def apply_lightGBM_regressor():
    """Apply LGBMRegressor model configuration."""
    model = LGBMRegressor(random_state=23)
    param_grid = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [5, 15, 20],
        "max_depth": [-1, 5, 10],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.6, 0.8, 1.0],
        # "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }
    param_distributions = {
        "n_estimators": randint(50, 500),
        "learning_rate": uniform(0.01, 0.2),
        "num_leaves": randint(15, 120),
        "max_depth": randint(-1, 30),
        "min_child_samples": randint(5, 60),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    }
    return _build_config(
        model=model,
        model_name="LGBMRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_XGB_regressor():
    """Apply XGBRegressor model configuration."""
    model = XGBRegressor(
        random_state=23,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [100, 150, 200, 250, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    param_distributions = {
        "n_estimators": randint(100, 600),
        "learning_rate": uniform(0.01, 0.2),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 10.0),
    }
    return _build_config(
        model=model,
        model_name="XGBRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_svm_regressor():
    """Apply SVR model configuration."""
    model = SVR()
    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2],
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "epsilon": [0.01, 0.1, 0.2],
        },
    ]
    param_distributions = [
        {
            "kernel": ["linear"],
            "C": loguniform(1e-3, 1e3),
            "epsilon": loguniform(1e-4, 1e-1),
        },
        {
            "kernel": ["rbf"],
            "C": loguniform(1e-3, 1e3),
            "gamma": loguniform(1e-4, 1),
            "epsilon": loguniform(1e-4, 1e-1),
        },
    ]
    return _build_config(
        model=model,
        model_name="SVR",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_knn_regressor():
    """Apply KNeighborsRegressor model configuration."""
    model = KNeighborsRegressor()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13],
        "weights": ["uniform", "distance"],
        "metric": ["manhattan", "euclidean", "minkowski"],
        "leaf_size": [20, 30, 40],
    }
    param_distributions = {
        "n_neighbors": randint(3, 25),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "leaf_size": randint(10, 60),
    }
    return _build_config(
        model=model,
        model_name="KNeighborsRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_ridge_regression():
    """Apply Ridge regression model configuration."""
    model = Ridge(random_state=23)
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
    }
    param_distributions = {
        "alpha": loguniform(1e-3, 1e3),
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
    }
    return _build_config(
        model=model,
        model_name="RidgeRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_random_forest():
    """Apply RandomForestRegressor model configuration."""
    model = RandomForestRegressor(random_state=23)
    param_grid = {
        "n_estimators": [300, 400, 500],
        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "max_depth": [4, 5, 7, 10],
        "min_samples_split": [2, 5, 9],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True], 
    }
    param_distributions = {
        "n_estimators": randint(100, 600),
        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "max_depth": randint(3, 15),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }
    return _build_config(
        model=model,
        model_name="RandomForestRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_mlp():
    """Apply MLPRegressor model configuration."""
    model = MLPRegressor(random_state=23, max_iter=1000, early_stopping=True)
    param_grid = {
        "hidden_layer_sizes": [(10,), (20,), (50,), (10, 5), (20, 10)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.01, 0.001],
        "alpha": [0.001, 0.0001],
    }
    param_distributions = {
        "hidden_layer_sizes": [(10,), (20,), (50,), (10, 5), (20, 10)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": loguniform(1e-4, 1e-1),
        "alpha": loguniform(1e-4, 1e-1),
    }
    return _build_config(
        model=model,
        model_name="MLPRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_adaboost():
    """Apply AdaBoostRegressor model configuration."""
    model = AdaBoostRegressor(random_state=23)
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"],
    }
    param_distributions = {
        "n_estimators": randint(50, 500),
        "learning_rate": loguniform(1e-3, 1.0),
        "loss": ["linear", "square", "exponential"],
    }
    return _build_config(
        model=model,
        model_name="AdaBoostRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_gradboost():
    """Apply GradientBoostingRegressor model configuration."""
    model = GradientBoostingRegressor(random_state=23)
    param_grid = {
        "n_estimators": [100, 150, 200, 250],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "max_depth": [2, 3, 5],
        "min_samples_leaf": [1, 2, 4],
    }
    param_distributions = {
        "n_estimators": randint(100, 400),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.5, 0.5),
        "max_depth": randint(2, 8),
        "min_samples_leaf": randint(1, 10),
    }
    return _build_config(
        model=model,
        model_name="GradientBoostingRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


def apply_hist_gradboost():
    """Apply HistGradientBoostingRegressor model configuration."""
    model = HistGradientBoostingRegressor(random_state=23)
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_iter": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 30],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    param_distributions = {
        "learning_rate": uniform(0.01, 0.2),
        "max_iter": randint(100, 500),
        "max_depth": [None, 5, 10, 20],
        "max_leaf_nodes": randint(15, 120),
        "min_samples_leaf": randint(5, 40),
        "l2_regularization": uniform(0.0, 1.0),
    }
    return _build_config(
        model=model,
        model_name="HistGradientBoostingRegressor",
        param_grid=param_grid,
        param_distributions=param_distributions,
    )


class SingleModelOrchestrator:
    def __init__(self, seed_=23):
        self.seed_ = seed_
        self.methods = {
            "RidgeRegressor": apply_ridge_regression,
            "KNeighborsRegressor": apply_knn_regressor,
            "SVRRegressor": apply_svm_regressor,
            "RandomForestRegressor": apply_random_forest,
            "MLPRegressor": apply_mlp,
            "AdaBoostRegressor": apply_adaboost,
            "GradientBoostingRegressor": apply_gradboost,
            "HistGradientBoostingRegressor": apply_hist_gradboost,
            "XGBRegressor": apply_XGB_regressor,
            "LGBMRegressor": apply_lightGBM_regressor,

        }

        self.model_methods = list(self.methods.keys())

    def apply(self, method_name, **kwargs):
        """
        Retorna a configuracao do modelo escolhido.

        Args:
            method_name (str): O nome do modelo a ser aplicado.
            **kwargs: parametros adicionais.

        Returns:
            dict: Dicionario contendo models_gs, model e model_name.
        """
        if method_name not in self.methods:
            raise ValueError(
                f"Modelo '{method_name}' nao reconhecido. Escolha entre: {self.model_methods}"
            )

        model_func = self.methods[method_name]
        return model_func()


if __name__ == "__main__":
    print("Single model regression loaded")
