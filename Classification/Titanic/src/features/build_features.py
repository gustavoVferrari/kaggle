import pandas as pd
from sklearn.pipeline import make_pipeline
from feature_engine.imputation import MeanMedianImputer, ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.discretisation import EqualWidthDiscretiser


def FitFeatureEngineering(X:pd.DataFrame, y:pd.DataFrame, pipeline):
    pipeline_fit = pipeline.fit(X, y)
    
    return pipeline_fit

def ApplyFeatureEngineering(pipeline_fit, X:pd.DataFrame):
    X_transformed = pipeline_fit.transform(X)
    
    return X_transformed

def PreprocessingPipeline_1(numerical_con:list,numerical_dis:list,categorical_var:list):    
    
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
        tol=.1,
        n_categories=2
    ))
    
    
    numerical_pipe_con = make_pipeline(median, StandardScaler().set_output(transform="pandas"))
    numerical_pipe_dis = make_pipeline(zero_inputer, MinMaxScaler().set_output(transform="pandas"))
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
        
        )
    
    return pipe


def PreprocessingPipeline_2(numerical_con:list,numerical_dis:list,categorical_var:list):    
    
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
        tol=.1,
        n_categories=2
    ))
    
    
    numerical_pipe_con = make_pipeline(median, StandardScaler().set_output(transform="pandas"))
    numerical_pipe_dis = make_pipeline(zero_inputer, MinMaxScaler().set_output(transform="pandas"))
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
        
        )
    
    return pipe


def PreprocessingPipeline_3(numerical_con:list,numerical_dis:list,categorical_var:list):    
    
    # numerical con
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con))    
    
    zero_inputer_eq = make_pipeline(
        ArbitraryNumberImputer(
        arbitrary_number = 0,
        variables = ['Fare', 'SibSp', 'FamilySize']))
    
    equalwidth_1 = make_pipeline(EqualWidthDiscretiser(
        variables=['Fare'],
        bins=5
        ))
    
    equalwidth_2 = make_pipeline(EqualWidthDiscretiser(
        variables=['SibSp', 'FamilySize'],
        bins=2
        ))
    
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
        tol=.1,
        n_categories=2
    ))
    
    
    numerical_pipe_con = make_pipeline(median, StandardScaler().set_output(transform="pandas"))
    numerical_equalwidth = make_pipeline(zero_inputer_eq, equalwidth_1, equalwidth_2)
    numerical_pipe_dis = make_pipeline(zero_inputer, MinMaxScaler().set_output(transform="pandas"))
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    
    preprocessor  = ColumnTransformer(
    transformers = [
    ("numerical_pipe_con", numerical_pipe_con, numerical_con), 
    ("numerical_pipe_equalwidth", numerical_equalwidth, ['Fare', 'SibSp', 'FamilySize']),
    ("numerical_pipe_dis", numerical_pipe_dis, numerical_dis), 
    ("categorical_pipe", categorical_pipe, categorical_var),
    ]
    )   
    
    pipe = make_pipeline(
        preprocessor.set_output(transform="pandas"),
        
        )
    
    return pipe


if __name__ == "__main__":
   print("Feature Engineering Pipeline carregado.")