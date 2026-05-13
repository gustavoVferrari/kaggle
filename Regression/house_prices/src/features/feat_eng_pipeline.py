from sklearn.pipeline import make_pipeline
from feature_engine.imputation import MeanMedianImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import BoxCoxTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from feature_engine.discretisation import DecisionTreeDiscretiser, GeometricWidthDiscretiser
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder


def apply_preprocessing(
    numerical_con_1: list,
    numerical_con_2: None, 
    numerical_dis: list, 
    categorical_var: list):
    
    
    # numerical_con_1
    median_numerical_con_1 = MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con_1)
    
    numerical_con_1_pipe = make_pipeline(
        median_numerical_con_1                 
        )
    
    # numerical_dis_1
    median_numerical_dis_1 = MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_dis)
    
    numerical_dis_1_pipe = make_pipeline(
        median_numerical_dis_1                 
        )
    
    # categorical_1
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = categorical_var
        ))       
    
    categorical_1_pipe = make_pipeline(
        cat_imputer              
        )

    preprocessor  = ColumnTransformer(
        transformers = [
            ("numerical_con_pipe", numerical_con_1_pipe, numerical_con_1),
            ("numerical_dis_pipe", numerical_dis_1_pipe, numerical_dis),
            ("categorical_pipe", categorical_1_pipe, categorical_var)
            
            ])
    
    pipe = make_pipeline(preprocessor.set_output(transform="pandas"))
    
    return pipe


def apply_preprocessing_pipeline_1(
    numerical_con_1:list,
    numerical_con_2:list, 
    numerical_dis_1:list, 
    categorical_1:list
    ):
    
    
    numerical_con_1 = [i for i in numerical_con_1 if i not in ['TotalBsmtSF']]
    
    # numerical continuous 1
    median_con_1 = MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con_1)
    
    outlier_1 = Winsorizer(
        variables=numerical_con_1, 
        capping_method='gaussian', 
        fold=5)
    
    log_transf = BoxCoxTransformer(
        variables=numerical_con_1
        )  
    
    # aplly robust sclaer only to BS    
    median_con_1_robust = MeanMedianImputer(
        imputation_method = 'median',
        variables = ['TotalBsmtSF'])
    
    robust_pipe = make_pipeline(
    RobustScaler(quantile_range=(25, 75)).set_output(transform="pandas")
    )
    
    outlier_2 = Winsorizer(
        variables=['TotalBsmtSF'], 
        capping_method='gaussian', 
        fold=5)
    

    num_con_1_robust_pipe = make_pipeline(
        median_con_1_robust,
        robust_pipe,
        outlier_2
        )
        
    num_con_1_pipe = make_pipeline(
        median_con_1,
        outlier_1,
        log_transf                
        )
    
    # numerical continuous 2
    median_con_2 = MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_con_2)    
    
    geom_disctser = GeometricWidthDiscretiser(
        variables=numerical_con_2
        )  
    
    num_con_2_pipe = make_pipeline(
        median_con_2,
        geom_disctser,                
        )
    
    # numerical discrete
    median_dis_1 = MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_dis_1)   

    
    num_dis_pipe = make_pipeline(
        median_dis_1                
        )
    
    # categorical 
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = categorical_1
        ))
    
    rare_label = make_pipeline(
        RareLabelEncoder(
        variables=categorical_1,
        tol=.10,
        n_categories=2
    ))

    encoder = make_pipeline(
        OneHotEncoder(
        variables = categorical_1, 
        drop_last = True
        ))
    
    categorical_pipe = make_pipeline(
        cat_imputer, 
        rare_label, 
        encoder
        )    
    
    preprocessor  = ColumnTransformer(
        transformers = [
            ("num_con_1_pipe", num_con_1_pipe, numerical_con_1),
            ("num_con_2_pipe", num_con_2_pipe, numerical_con_2),
            ("num_dis_pipe", num_dis_pipe, numerical_dis_1),
            ("categorical_pipe", categorical_pipe, categorical_1),
            ('num_con_1_robust', num_con_1_robust_pipe, ['TotalBsmtSF'])
            ])
    
    pipe = make_pipeline(   
        preprocessor.set_output(transform="pandas"),
        # StandardScaler().set_output(transform="pandas")               
        )      
    
    return pipe

class PreprocessingOrchestrator:
    def __init__(self, 
                 numerical_con_1: list, 
                 numerical_con_2: list, 
                 numerical_dis: list, 
                 categorical_var: list):
        """
        Inicializa o orquestrador de pipelines de preprocessamento.
        
        Args:
            numerical_con (list): Lista de variáveis numéricas contínuas.
            numerical_dis (list): Lista de variáveis numéricas discretas.
            categorical_var (list): Lista de variáveis categóricas.
        """
        self.numerical_con_1 = numerical_con_1
        self.numerical_con_2 = numerical_con_2
        self.numerical_dis = numerical_dis
        self.categorical_var = categorical_var
        
        self.methods = {
            "preprocessing":apply_preprocessing,
            "pipeline1": apply_preprocessing_pipeline_1,

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
        return pipeline_func(self.numerical_con_1, self.numerical_con_2, self.numerical_dis, self.categorical_var)