import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def save_data(file_path, name, dataframe:pd.DataFrame):
    """Salva os dados Parquet em uma pasta"""
    
    file = os.path.join(
        file_path,
        f'{name}.parquet')

    return dataframe.to_parquet(file)

def load_data(file_path):
    """Carrega os dados brutos de um arquivo Parquet."""
    return pd.read_parquet(file_path)

def clean_data(df):
    """Realiza a limpeza básica dos dados."""
       
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    
    """Divide os dados em conjuntos de treino e teste."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de processamento de dados carregado.")
