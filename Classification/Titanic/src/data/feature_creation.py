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
    else:
        dataset_path = os.path.join(dataset_path, "test.csv")
        df = pd.read_csv(dataset_path)         
    
    for col in ['Ticket', 'Cabin']:
        df[col] = df[col].str.replace(r'[^A-Za-z0-9]', '', regex=True)
        
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
        
    # Extração de prefixos
    df['Ticket_1p'] = df['Ticket'].apply(lambda row: row[:1] if pd.notnull(row) else row)
    df['Cabin_1p'] = df['Cabin'].apply(lambda row: row[:1] if pd.notnull(row) else row)
        
    # Engenharia de Atributos: Agrupamento de Embarque e Tamanho de Família
    df['Embarked_mod'] = df['Embarked'].map({'S': 'SQ', 'Q': 'SQ', 'C': 'C'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)    
    
    df['Fare'] = df['Fare'].apply(lambda x: np.abs(x))
 
    df = df.astype({
        'Pclass':str, 
        'Age':np.float64, 
        'SibSp':np.float64,
        'Parch':np.float64
        })      
 
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