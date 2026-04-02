import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from feature_engine.imputation import MeanMedianImputer, ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder
from feature_engine.transformation import LogCpTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.discretisation import EqualWidthDiscretiser,ArbitraryDiscretiser


def apply_preprocessing(numerical_con: list, numerical_dis: list, categorical_var: list):

    """Apply Basic Preprocessing"""
    
    # numerical con
    median = make_pipeline(
        MeanMedianImputer(
            imputation_method='median',
            variables=numerical_con))    
    
    # numerical dis
    zero_inputer = make_pipeline(
        ArbitraryNumberImputer(
            arbitrary_number=0,
            variables=numerical_dis))
    
    # cat
    cat_imputer = make_pipeline(
        CategoricalImputer(
            imputation_method='missing',
            fill_value='missing', 
            variables=categorical_var))


    
    numerical_pipe_con = make_pipeline(median, )
    numerical_pipe_dis = make_pipeline(zero_inputer, )
    categorical_pipe = make_pipeline(cat_imputer)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_pipe_con", numerical_pipe_con, numerical_con), 
            ("numerical_pipe_dis", numerical_pipe_dis, numerical_dis), 
            ("categorical_pipe", categorical_pipe, categorical_var),
        ])   
    
    pipe = make_pipeline(preprocessor.set_output(transform="pandas"))
    
    return pipe


def apply_preprocessing_pipeline_1(numerical_con: list, numerical_dis: list, categorical_var: list):

    """Apply Preprocessing Pipeline 1 configuration."""
    
    # numerical con
    median = make_pipeline(
        MeanMedianImputer(
            imputation_method='median',
            variables=numerical_con))
    
    # numerical dis
    zero_inputer = make_pipeline(
        ArbitraryNumberImputer(
            arbitrary_number=0,
            variables=numerical_dis))
    
    # nominal var
    cat_imputer = make_pipeline(
        CategoricalImputer(
            imputation_method='missing',
            fill_value='missing', 
            variables=categorical_var))

    encoder = make_pipeline(
        OneHotEncoder(
            variables=categorical_var, 
            drop_last=True))

    rare_label = make_pipeline(
        RareLabelEncoder(
            variables=categorical_var,
            tol=.01,
            n_categories=2))
    
    numerical_pipe_con = make_pipeline(median, StandardScaler().set_output(transform="pandas"))
    numerical_pipe_dis = make_pipeline(zero_inputer, MinMaxScaler().set_output(transform="pandas"))
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_pipe_con", numerical_pipe_con, numerical_con), 
            ("numerical_pipe_dis", numerical_pipe_dis, numerical_dis), 
            ("categorical_pipe", categorical_pipe, categorical_var),
        ])   
    
    pipe = make_pipeline(preprocessor.set_output(transform="pandas"))
    
    return pipe


def apply_preprocessing_pipeline_2(numerical_con: list, numerical_dis: list, categorical_var: list):
    """Apply Preprocessing Pipeline 2 configuration."""
    
     # numerical con
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con))
    
    # numerical dis
    zero_inputer = make_pipeline(
        ArbitraryNumberImputer(
        arbitrary_number = 0,
        variables = numerical_dis))
    
    # nominal var
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = categorical_var
        ))

    encoder = make_pipeline(
        OneHotEncoder(
        variables = categorical_var, 
        drop_last = True
        ))

    rare_label = make_pipeline(
        RareLabelEncoder(
        variables=categorical_var,
        tol=.15,
        n_categories=2
    ))
    
    log_transform = make_pipeline(
        LogCpTransformer(variables=['Fare']))
    
    median_fare = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = ['Fare']))
    
    
    numerical_pipe_con = make_pipeline(median, StandardScaler().set_output(transform="pandas"))
    numerical_pipe_dis = make_pipeline(zero_inputer, MinMaxScaler().set_output(transform="pandas"))
    numerical_transf = make_pipeline(median_fare, log_transform)
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    
    preprocessor  = ColumnTransformer(
    transformers = [
    ("numerical_pipe_con", numerical_pipe_con, numerical_con), 
    ("numerical_pipe_dis", numerical_pipe_dis, numerical_dis), 
    ("numerical_pipe_trans", numerical_transf, ['Fare']), 
    ("categorical_pipe", categorical_pipe, categorical_var),
    ]
    )   
    
    pipe = make_pipeline(
        preprocessor.set_output(transform="pandas"),
        
        )
    
    return pipe

def apply_preprocessing_pipeline_3(numerical_con:list,numerical_dis:list,categorical_var:list):    
    """Apply Preprocessing Pipeline 3 configuration."""
    
    # numerical con
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con))    
    
    user_dict = {
        'Fare': [0, 20, 50, 100, np.inf],
        'Age':[0, 12, 60, np.inf]
        }
    discretiser = make_pipeline(ArbitraryDiscretiser(
        binning_dict=user_dict, 
        return_object=False,
        return_boundaries=False    
        ))    
    
    # numerical dis
    zero_inputer = make_pipeline(
        ArbitraryNumberImputer(
        arbitrary_number = 0,
        variables = numerical_dis))
    
    user_dict_disc = {
        'FamilySize': [0, 5, np.inf],
        }
    discretiser_dis = make_pipeline(ArbitraryDiscretiser(
        binning_dict=user_dict_disc, 
        return_object=False,
        return_boundaries=False    
        )) 
    
    # nominal var
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = categorical_var
        ))

    # encoder = make_pipeline(
    #     OneHotEncoder(
    #     variables = categorical_var, 
    #     drop_last = True
    #     ))
    
    encoder = make_pipeline(
        OrdinalEncoder(
            variables = categorical_var
        ))

    rare_label = make_pipeline(
        RareLabelEncoder(
        variables=categorical_var,
        tol=.01,
        n_categories=2
    ))
    
    
    numerical_pipe_con = make_pipeline(median, discretiser)
    numerical_pipe_dis = make_pipeline(zero_inputer, discretiser_dis)
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    
    preprocessor  = ColumnTransformer(
    transformers = [
    ("numerical_pipe_con", numerical_pipe_con, numerical_con), 
    ("numerical_pipe_dis", numerical_pipe_dis, numerical_dis), 
    ("categorical_pipe", categorical_pipe, categorical_var),
    ]
    )   
    
    pipe = make_pipeline(
        preprocessor.set_output(transform="pandas"),
        StandardScaler().set_output(transform="pandas")
        
        )
    
    return pipe


class PreprocessingOrchestrator:
    def __init__(self, numerical_con: list, numerical_dis: list, categorical_var: list):
        """
        Inicializa o orquestrador de pipelines de preprocessamento.
        
        Args:
            numerical_con (list): Lista de variáveis numéricas contínuas.
            numerical_dis (list): Lista de variáveis numéricas discretas.
            categorical_var (list): Lista de variáveis categóricas.
        """
        self.numerical_con = numerical_con
        self.numerical_dis = numerical_dis
        self.categorical_var = categorical_var
        
        self.methods = {
            "preprocessing":apply_preprocessing,
            "Pipeline1": apply_preprocessing_pipeline_1,
            "Pipeline2": apply_preprocessing_pipeline_2,
            "Pipeline3": apply_preprocessing_pipeline_3,
        }
        
        self.preprocessing_methods = list(self.methods.keys())
        
    def apply(self, method_name):
        """
        Retorna o pipeline de preprocessamento escolhido.
        
        Args:
            method_name (str): O nome do pipeline a ser aplicado.
            
        Returns:
            Pipeline: Pipeline de preprocessamento configurado.
        """
        if method_name not in self.methods:
            raise ValueError(
                f"Pipeline '{method_name}' não reconhecido. "
                f"Escolha entre: {self.preprocessing_methods}"
            )
        
        # Obtém a função correspondente e a executa
        pipeline_func = self.methods[method_name]
        return pipeline_func(self.numerical_con, self.numerical_dis, self.categorical_var)
    
if __name__ == "__main__":
   print("Feature eng. carregado")