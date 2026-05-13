"""Funcoes para gerar analises simples de variaveis em datasets parquet.

O modulo cria artefatos de analise exploratoria, como grafico de dados
ausentes, cardinalidade das colunas e separacao entre variaveis categoricas e
numericas.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


def MissingData(dataframe_path:str, plot_path:str) -> None:
    """Gera um grafico com a proporcao de valores ausentes por coluna.

    Le um arquivo parquet, calcula a media de valores nulos em cada coluna e
    salva um grafico de barras chamado `missing_data.png` no diretorio
    informado.

    Args:
        dataframe_path (str): Caminho do arquivo parquet a ser analisado.
        plot_path (str): Diretorio onde o grafico sera salvo.

    Returns:
        None.
    """
    
    df = pd.read_parquet(dataframe_path)       
   
   # missing data plot
    df.isna().mean().plot.bar(title='missing data')    
    
    path_save = os.path.join(
        plot_path,
        'missing_data.png')    
    
    plt.savefig(
        path_save, 
        dpi=300, 
        bbox_inches="tight")
    plt.close() 
    
def CardinalityAnalysis(dataframe_path:str, report_path:str) -> None:
    """Calcula a cardinalidade de cada coluna e salva o resultado em JSONL.

    A cardinalidade corresponde ao numero de valores unicos encontrados em cada
    coluna do dataframe. O resultado e salvo em `cardinality.jsonl` no diretorio
    informado.

    Args:
        dataframe_path (str): Caminho do arquivo parquet a ser analisado.
        report_path (str): Diretorio onde o relatorio JSONL sera salvo.

    Returns:
        None.
    """
    
    df = pd.read_parquet(dataframe_path)    
       
    cardinality = df.nunique().sort_values(ascending=False)
    
    dict_cardinality = {}
    for col in df.columns:
        dict_cardinality[col] =  str(df[col].nunique())
    
    cardinality_df = pd.DataFrame(
        dict_cardinality, 
        index=['cardinality']   
        )
    
    # cardinality 
    cardinality_df.to_json(
        os.path.join(
            report_path, 'cardinality.jsonl'
            ), 
        orient='records',
        lines=True)
    
def ColsTypeAnalysis(dataframe_path:str, report_path:str) -> None:
    """Identifica colunas categoricas e numericas e salva o relatorio.

    Classifica como categoricas as colunas dos tipos `category`, `object` e
    `bool`; classifica como numericas as colunas com dtype numerico. O resultado
    e salvo em `variable_type.jsonl` no diretorio informado.

    Args:
        dataframe_path (str): Caminho do arquivo parquet a ser analisado.
        report_path (str): Diretorio onde o relatorio JSONL sera salvo.

    Returns:
        None.
    """
    
    df = pd.read_parquet(dataframe_path)
    
    categorical_col = list(
        df.select_dtypes(
            include=['category','object', 'bool']).columns
        )    
    numerical_col = list(
        df.select_dtypes(include=['number']).columns
        )
    
    dict_var_type = dict(
        categorical_var = categorical_col,
        numerical_var = numerical_col) 
    
    with open(os.path.join(report_path, 'variable_type.jsonl'), 'w') as f:
        import json
        json.dump(dict_var_type, f)
    

if __name__ == "__main__":
   print("Feature Analysis carregado.")
