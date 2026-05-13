"""Funcoes auxiliares para carregar, salvar, limpar e dividir datasets.

O modulo concentra operacoes basicas de preparacao de dados usadas no projeto,
incluindo leitura/escrita em parquet e separacao entre treino e teste.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def save_data(file_path, name, dataframe: pd.DataFrame):
    """Salva um dataframe em formato parquet.

    Args:
        file_path (str): Diretorio onde o arquivo sera salvo.
        name (str): Nome do arquivo sem extensao.
        dataframe (pd.DataFrame): Dataframe que sera persistido.

    Returns:
        None: Retorno produzido por `DataFrame.to_parquet`.
    """

    file = os.path.join(
        file_path,
        f'{name}.parquet')

    return dataframe.to_parquet(file)


def load_data(file_path):
    """Carrega um dataframe a partir de um arquivo parquet.

    Args:
        file_path (str): Caminho completo do arquivo parquet.

    Returns:
        pd.DataFrame: Dados carregados do arquivo informado.
    """
    return pd.read_parquet(file_path)


def clean_data(df):
    """Executa a etapa de limpeza dos dados.

    Atualmente a funcao apenas retorna o dataframe recebido, servindo como ponto
    de extensao para regras futuras de limpeza e tratamento.

    Args:
        df (pd.DataFrame): Dataframe de entrada.

    Returns:
        pd.DataFrame: Dataframe apos a etapa de limpeza.
    """

    return df


def split_data(df, target_column, test_size=0.2, random_state=42):

    """Divide um dataframe em conjuntos de treino e teste.

    Separa a coluna alvo das variaveis explicativas e aplica
    `train_test_split` para criar os conjuntos de treino e teste.

    Args:
        df (pd.DataFrame): Dataframe completo contendo atributos e alvo.
        target_column (str): Nome da coluna alvo.
        test_size (float, optional): Proporcao dos dados destinada ao teste.
            Padrao: 0.2.
        random_state (int, optional): Semente usada para reproduzir a divisao.
            Padrao: 42.

    Returns:
        tuple: `X_train`, `X_test`, `y_train` e `y_test`.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Exemplo de uso
    print("Modulo de processamento de dados carregado.")
