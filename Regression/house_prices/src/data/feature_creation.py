import numpy as np
import pandas as pd 
import os

def FeatureCreation(dataset_path:str, save_path:str, train:bool=True) -> None:
    
    """
    Executa a limpeza de dados e criação de features para o dataset especificado.

    Args:
        dataset_name (str): Nome do conjunto de dados (ex: 'train', 'test').
        **params: Dicionário contendo os caminhos 'raw', 'processed' e 'reports'.
    """


    # Leitura dos dados
    if train:
        dataset_path = os.path.join(dataset_path, "train.csv")
        df = pd.read_csv(dataset_path)
        df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(lambda row: np.nan if row == 0 else row)
        df['SalePrice'] = np.log1p(df['SalePrice'])
        
        
    else:
        dataset_path = os.path.join(dataset_path, "test.csv")
        df = pd.read_csv(dataset_path)         
        df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(lambda row: np.nan if row == 0 else row)

    if train:
        return df.to_parquet(
            os.path.join(
            save_path, "train_features.parquet"              
            ))
    else:
        return df.to_parquet(
            os.path.join(
            save_path, "test_features.parquet"              
            ))

if __name__ == "__main__":
   print("Feature Creation carregado.")