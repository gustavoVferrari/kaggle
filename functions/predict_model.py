"""Funcoes auxiliares para carregar modelos treinados e gerar predicoes.

O modulo cobre leitura de modelos serializados com pickle e formatacao das
predicoes em dataframes, mantendo o indice dos dados de entrada.
"""

import pandas as pd
import pickle


def load_trained_model(model_path):
    """Carrega um modelo treinado salvo em arquivo pickle.

    Args:
        model_path (str): Caminho do arquivo contendo o modelo serializado.

    Returns:
        object: Modelo carregado do disco.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)

        return model


def make_prediction(model, input_data):
    """Gera predicoes e probabilidades para um modelo de classificacao.

    Usa `predict` para obter a classe prevista e `predict_proba` para obter a
    probabilidade da classe positiva. Os resultados sao retornados como
    dataframes separados, ambos com o mesmo indice de `input_data`.

    Args:
        model: Modelo treinado com metodos `predict` e `predict_proba`.
        input_data (pd.DataFrame): Dados usados para gerar as predicoes.

    Returns:
        tuple: `df_predictions` com a coluna `prediction` e `df_probabilities`
        com a coluna `probability`.
    """

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    df_predictions = pd.DataFrame(
        predictions,
        index=input_data.index,
        columns=["prediction"]
    )

    df_probabilities = pd.DataFrame(
        probabilities,
        index=input_data.index,
        columns=["probability"]
    )

    return df_predictions, df_probabilities


def make_prediction_reg(model, input_data):
    """Gera predicoes para um modelo de regressao.

    Usa `predict` para obter os valores previstos e retorna o resultado em um
    dataframe com o mesmo indice de `input_data`.

    Args:
        model: Modelo treinado com metodo `predict`.
        input_data (pd.DataFrame): Dados usados para gerar as predicoes.

    Returns:
        pd.DataFrame: Dataframe com a coluna `prediction`.
    """

    predictions = model.predict(input_data)

    df_predictions = pd.DataFrame(
        predictions,
        index=input_data.index,
        columns=["prediction"]
    )

    return df_predictions


if __name__ == "__main__":
    print("Modulo de predicao carregado.")
