from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    )
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform

def apply_lightGBM_classifier():
    """Apply LGBMClassifiermodel configuration."""
    return dict(
        param_grid={
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10, 20],
            "min_child_samples": [10, 20, 30],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        param_distributions={
            "n_estimators": randint(50, 300),
            "learning_rate": uniform(0.01, 0.2),
            "num_leaves": randint(15, 100),
            "max_depth": randint(-1, 30),
            "min_child_samples": randint(5, 50),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5)
        },
        model=LGBMClassifier(),
        model_name='LGBMClassifier'
    )

def apply_XGB_classifier():
    """Apply XGBClassifier model configuration."""
    return dict(       
        model=XGBClassifier(random_state=23),
        model_name='XGBClassifier'
    )

def apply_svm_classifier():
    """Apply SVM model configuration."""
    
    return dict(
        param_grid = [
            {"kernel": ["linear"],
            "C": [0.1, 1, 10, 100]}
            ,
            {"kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1]}              
        ], 
        param_distributions =  [
            {"kernel": ["linear"],
             "C": loguniform(1e-3, 1e3)},
            {"kernel": ["rbf"],
             "C": loguniform(1e-3, 1e3),
             "gamma": loguniform(1e-4, 1)}
            ],     
        model=SVC(probability=True, random_state=23),
        model_name='SVC')

def apply_knn_classifier():
    """Apply knn model configuration."""
    return dict(
        param_grid = {
            "n_neighbors":[3,5,7,9,11,13],
            "weights":['uniform', 'distance'], 
            "metric": ['manhattan', 'cosine', 'minkowski']           
            },
        param_distributions = {
            "n_neighbors": randint(3, 20),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
            },        
        model=KNeighborsClassifier(),
        model_name='KNeighborsClassifier'
    ) 
 
def apply_logistic_regression():
    """Apply Logistic Regression model configuration."""
    
    return dict(
        param_grid = {
            "penalty":['l2'],
            "C":[10, 1, 0.1], 
            "max_iter":[200, 300, 1000], 
            "solver":['liblinear', 'saga', 'lbfgs']    
            }, 
        param_distributions = {
            "penalty":['l2'],
            "max_iter":randint(100, 1500),
            "C":loguniform(1e-3, 1e2),
            "solver":['liblinear', 'saga', 'lbfgs'] 
        },      
        model=LogisticRegression(random_state=23),
        model_name='LogisticRegression'
    )

def apply_random_forest():
    """Apply Random Forest model configuration."""
    
    return dict(
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 7, 10],
            "min_samples_split": [4, 5, 10],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }, 
        param_distributions = {
            "n_estimators": randint(100, 600),
            "max_depth": randint(3, 10),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },          
        model=RandomForestClassifier(random_state=23),
        model_name='RandomForestClassifier'
    )

def apply_mlp():
    """Apply MLP Classifier model configuration."""
    return dict(
        param_grid = {
            "hidden_layer_sizes":[10, 30, 50],
            "activation": ['relu', 'tanh'],
            "learning_rate_init": [0.01, 0.001],
            'alpha': [0.001, 0.0001]            
        },
        param_distributions = {
            "hidden_layer_sizes": [(10,), (20,), (50,),
        (10, 5), (20, 10)],
            "activation": ['relu', 'tanh'],
            "learning_rate_init": loguniform(1e-4, 1e-1),
            "alpha": loguniform(1e-4, 1e-1)           
        },
        model=MLPClassifier(
            random_state=23, 
            max_iter=1000, 
            early_stopping=True),
        model_name='MLPClassifier'
    )

def apply_adaboost():
    """Apply AdaBoost model configuration."""
    return dict(
        param_grid={
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.1, 1],
            "algorithm": ["SAMME", "SAMME.R"]
        },
        param_distributions={
            "n_estimators": randint(50, 500),
            "learning_rate": loguniform(1e-3, 1),
            "algorithm": ["SAMME", "SAMME.R"]
        },       
        model=AdaBoostClassifier(random_state=23),
        model_name='AdaBoostClassifier'
    )

def apply_gradboost():
    """Apply AdaBoost model configuration."""
    return dict(        
        model=GradientBoostingClassifier(random_state=23),
        model_name='GradientBoostingClassifier'
    )

class SingleModelOrchestrator:
    def __init__(self):
                
        self.methods = {
            "LogisticRegression": apply_logistic_regression,
            "KNeighborsClassifier": apply_knn_classifier,
            "SVC": apply_svm_classifier,
            "RandomForestClassifier": apply_random_forest,
            "MLPClassifier": apply_mlp,
            "AdaBoostClassifier": apply_adaboost,
            "GradientBoostingClassifier":apply_gradboost,
            "XGBClassifier":apply_XGB_classifier,
            "LGBMClassifier":apply_lightGBM_classifier
        }
        
        self.model_methods = list(self.methods.keys())
        
    def apply(self, method_name, **kwargs):
        """
        Retorna a configuração do modelo escolhido.
        
        Args:
            method_name (str): O nome do modelo a ser aplicado.
            **kwargs: parâmetros adicionais (ex: input_dim para ANN)
            
        Returns:
            dict: Dicionário contendo models_gs, model e model_name.
        """
        if method_name not in self.methods:
            raise ValueError(f"Modelo '{method_name}' não reconhecido. Escolha entre: {self.model_methods}")
        
        # Obtém a função correspondente e a executa
        model_func = self.methods[method_name]
        return model_func()
    
    
if __name__ == "__main__":
    print("Single model loaded")