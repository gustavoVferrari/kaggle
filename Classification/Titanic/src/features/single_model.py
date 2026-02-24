from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier



def apply_lightGBM_classifier(seed_):
    """Apply LGBMClassifiermodel configuration."""
    return dict(
        models_gs={
            "lgb": [
                LGBMClassifier(),
                 {
                     'lgbmclassifier__n_estimators':[150, 200, 250, 300],
                     'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1, 0.2], 
                     'lgbmclassifier__max_depth': [3, 5, 7]
                     }
            ]
        },
        model=LGBMClassifier(),
        model_name='lgb'
    )

def apply_XGB_classifier(seed_):
    """Apply XGBClassifier model configuration."""
    return dict(
        models_gs={
            "xgb": [
                XGBClassifier(random_state=seed_),
                 {
                     'xgbclassifier__n_estimators':[100, 150, 200, 250],
                     'xgbclassifier__learning_rate': [0.01, 0.1, 0.2], 
                     'xgbclassifier__max_depth': [3, 5, 7],
                     'xgbclassifier__subsample':[0.6, 0.8, 1.0]
                     }
            ]
        },
        model=XGBClassifier(random_state=seed_),
        model_name='xgb'
    )

def apply_svm_classifier(seed_):
    """Apply SVM model configuration."""
    return dict(
        models_gs={
            "svc_clf": [
                SVC(probability=True, random_state=seed_),
                 {
                     'svc__kernel':['poly', 'rbf'],
                     'svc__C': [0.1, 1, 10]
                     }
            ]
        },
        model=SVC(probability=True, random_state=seed_),
        model_name='svc'
    )


def apply_knn_classifier(seed_=None):
    """Apply knn model configuration."""
    return dict(
        models_gs={
            "knn_clf": [
                KNeighborsClassifier(),
                {
                    'kneighborsclassifier__n_neighbors':[3,5,7,9,11,13],
                    'kneighborsclassifier__weights': ['uniform', 'distance'], 
                    'kneighborsclassifier__metric': ['manhattan', 'cosine', 'haversine', 'minkowski']
                    }
            ]
        },
        model=KNeighborsClassifier(),
        model_name='knn'
    )
 
 
def apply_logistic_regression(seed_):
    """Apply Logistic Regression model configuration."""
    return dict(
        models_gs={
            "logistic_reg": [
                LogisticRegression(max_iter=1000, penalty="l2"),
                {
                    "logisticregression__solver": ["saga", "lbfgs", "liblinear"],
                    "logisticregression__C": [0.1, 1, 100],
                }
            ]
        },
        model=LogisticRegression(random_state=seed_),
        model_name='lr'
    )

def apply_random_forest(seed_):
    """Apply Random Forest model configuration."""
    return dict(
        models_gs={
            "random_forest": [
                RandomForestClassifier(random_state=seed_),
                {
                    'randomforestclassifier__n_estimators': [100, 150, 200, 250],
                    'randomforestclassifier__criterion': ['gini', 'entropy'], 
                    'randomforestclassifier__max_depth': [None, 4, 5, 7],
                    'randomforestclassifier__min_samples_split': [2, 4, 6]
                }
            ]
        },
        model=RandomForestClassifier(random_state=seed_),
        model_name='rf'
    )

def apply_mlp(seed_):
    """Apply MLP Classifier model configuration."""
    return dict(
        models_gs={
            "ml": [
                MLPClassifier(random_state=seed_),        
                {
                    'mlpclassifier__hidden_layer_sizes': [10, 20, 30],
                    'mlpclassifier__activation': ['relu', 'tanh'],
                    'mlpclassifier__learning_rate_init': [0.01, 0.001]
                }
            ]
        },
        model=MLPClassifier(random_state=seed_),
        model_name='MLP_clf'
    )

def apply_adaboost(seed_):
    """Apply AdaBoost model configuration."""
    return dict(
        models_gs={
            "ab": [
                AdaBoostClassifier(random_state=seed_),        
                {
                    'adaboostclassifier__n_estimators': [150, 200, 250, 300],
                    'adaboostclassifier__learning_rate': [0.01, 0.1, 0.001]
                }
            ]
        },
        model=AdaBoostClassifier(random_state=seed_),
        model_name='adaboost'
    )


def apply_gradboost(seed_):
    """Apply AdaBoost model configuration."""
    return dict(
        models_gs={
            "ab": [
                GradientBoostingClassifier(random_state=seed_),        
                {
                    'gradientboostingclassifier__n_estimators':[100, 150, 200, 250],
                    'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.001]
                }
            ]
        },
        model=AdaBoostClassifier(random_state=seed_),
        model_name='gradboost'
    )

class ModelOrchestrator:
    def __init__(self, seed_=42):
        self.seed_ = seed_
        self.methods = {
            "LogisticRegression": apply_logistic_regression,
            "KnnClassifier": apply_knn_classifier,
            "SvmClassifier": apply_svm_classifier,
            "RandomForest": apply_random_forest,
            "MLP": apply_mlp,
            "AdaBoost": apply_adaboost,
            "GradBoost":apply_gradboost,
            "XGradBoost":apply_XGB_classifier,
            "LGBM":apply_lightGBM_classifier
        }
        
        self.model_methods = list(self.methods.keys())
        
    def apply(self, method_name):
        """
        Retorna a configuração do modelo escolhido.
        
        Args:
            method_name (str): O nome do modelo a ser aplicado.
            
        Returns:
            dict: Dicionário contendo models_gs, model e model_name.
        """
        if method_name not in self.methods:
            raise ValueError(f"Modelo '{method_name}' não reconhecido. Escolha entre: {self.model_methods}")
        
        # Obtém a função correspondente e a executa
        model_func = self.methods[method_name]
        return model_func(self.seed_)