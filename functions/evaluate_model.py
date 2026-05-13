"""Funcoes para avaliar modelos e persistir metricas em JSONL.

O modulo oferece helpers para calcular metricas de classificacao e regressao,
criar registros padronizados com metadados e salvar/carregar historicos de
metricas por arquivo.
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)


def evaluate_clf_model(model, X_test, y_test, threshold=0.5):
    """Avalia um modelo de classificacao binaria em dados de teste.

    Calcula relatorio de classificacao, acuracia, F1-score e ROC AUC. O ROC AUC
    usa as probabilidades brutas da classe positiva retornadas por
    `predict_proba`, enquanto as demais metricas usam os rotulos retornados por
    `predict`.

    Args:
        model: Estimador treinado com metodos `predict` e `predict_proba`.
        X_test (array-like): Matriz de atributos de teste.
        y_test (array-like): Rotulos verdadeiros do conjunto de teste.
        threshold (float, optional): Limiar usado para binarizar as
            probabilidades da classe positiva. Padrao: 0.5.

    Returns:
        dict: Dicionario com `classification_report`, `accuracy_score`,
        `f1_score` e `roc_auc_score`.
    """
    metrics = {}

    pred = model.predict(X_test)
    # Para roc_auc_score, usamos as probabilidades brutas, nao binarizadas.
    # Mantendo a logica original, esta variavel representa a predicao
    # binarizada com base no threshold.
    proba_raw = model.predict_proba(X_test)[:, 1]
    proba_binarized = np.where(proba_raw >= threshold, 1, 0)

    metrics['classification_report'] = classification_report(y_test, pred, output_dict=True)
    metrics['accuracy_score'] = accuracy_score(y_test, pred)
    metrics['f1_score'] = f1_score(y_test, pred)
    metrics['roc_auc_score'] = roc_auc_score(y_test, proba_raw)  # Usa probabilidades brutas para ROC AUC

    return metrics


def evaluate_reg_model(model, X_test, y_test):
    """Avalia um modelo de regressao em dados de teste.

    Args:
        model: Estimador treinado com metodo `predict`.
        X_test (array-like): Matriz de atributos de teste.
        y_test (array-like): Valores verdadeiros do conjunto de teste.

    Returns:
        dict: Dicionario com MAE, MSE, RMSE e R2.
    """
    metrics = {}

    pred = model.predict(X_test)

    metrics['mean_absolute_error'] = mean_absolute_error(y_test, pred)
    metrics['mean_squared_error'] = mean_squared_error(y_test, pred)
    metrics['root_mean_squared_error'] = np.sqrt(mean_squared_error(y_test, pred))
    metrics['r2_score'] = r2_score(y_test, pred)

    return metrics


def _create_metric_data(metric_name, model_name, dataset, score, undersampling=None, **kwargs):
    """Cria o dicionario padrao para uma metrica calculada.

    Args:
        metric_name (str): Nome da metrica.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        score (float): Valor numerico da metrica.
        undersampling (str, optional): Identificacao da estrategia de
            undersampling aplicada, quando houver. Padrao: None.
        **kwargs: Campos extras adicionados ao registro da metrica.

    Returns:
        dict: Registro padronizado da metrica com timestamp.
    """
    metric_data = {
        'model_name': model_name,
        'dataset': dataset,
        'metric_name': metric_name,
        'score': float(score),
        'undersampling': undersampling,
        'timestamp': datetime.now().isoformat()
    }
    metric_data.update(kwargs)
    return metric_data


def save_accuracy_score(accuracy_value, model_name, dataset, undersampling=None):
    """Cria o registro da metrica de acuracia.

    Args:
        accuracy_value (float): Valor calculado da acuracia.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data('accuracy_score', model_name, dataset, accuracy_value, undersampling=undersampling)


def save_f1_score(f1_value, model_name, dataset, average='weighted', undersampling=None):
    """Cria o registro da metrica F1-score.

    Args:
        f1_value (float): Valor calculado do F1-score.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        average (str, optional): Estrategia de media usada no calculo.
            Padrao: `"weighted"`.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data('f1_score', model_name, dataset, f1_value, undersampling=undersampling, average=average)


def save_roc_auc_score(roc_auc_value, model_name, dataset, average='weighted', multi_class='ovr', undersampling=None):
    """Cria o registro da metrica ROC AUC.

    Args:
        roc_auc_value (float): Valor calculado do ROC AUC.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        average (str, optional): Estrategia de media usada no calculo.
            Padrao: `"weighted"`.
        multi_class (str, optional): Estrategia para problemas multiclasse.
            Padrao: `"ovr"`.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data('roc_auc_score', model_name, dataset, roc_auc_value, undersampling=undersampling, average=average, multi_class=multi_class)


def save_classification_report(report_dict, model_name, dataset, undersampling=None):
    """Cria o registro do relatorio de classificacao.

    Args:
        report_dict (dict): Relatorio produzido por `classification_report`
            com `output_dict=True`.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro com o relatorio de classificacao e metadados.
    """
    metric_data = {
        'model_name': model_name,
        'dataset': dataset,
        'metric_name': 'classification_report',
        'report': report_dict,
        'undersampling': undersampling,
        'timestamp': datetime.now().isoformat()
    }
    return metric_data


def save_mean_absolute_error(mean_absolute_error_value, model_name, dataset, undersampling=None):
    """Cria o registro da metrica mean absolute error.

    Args:
        mean_absolute_error_value (float): Valor calculado do MAE.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data(
        'mean_absolute_error',
        model_name,
        dataset,
        mean_absolute_error_value,
        undersampling=undersampling
    )


def save_mean_squared_error(mean_squared_error_value, model_name, dataset, undersampling=None):
    """Cria o registro da metrica mean squared error.

    Args:
        mean_squared_error_value (float): Valor calculado do MSE.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data(
        'mean_squared_error',
        model_name,
        dataset,
        mean_squared_error_value,
        undersampling=undersampling
    )


def save_r2_score(r2_value, model_name, dataset, undersampling=None):
    """Cria o registro da metrica R2 score.

    Args:
        r2_value (float): Valor calculado do R2.
        model_name (str): Nome do modelo avaliado.
        dataset (str): Nome ou identificador do dataset usado.
        undersampling (str, optional): Estrategia de undersampling aplicada,
            quando houver. Padrao: None.

    Returns:
        dict: Registro padronizado da metrica.
    """
    return _create_metric_data('r2_score', model_name, dataset, r2_value, undersampling=undersampling)


class MetricsOrchestrator:
    """Orquestra persistencia e leitura de metricas em arquivos JSONL."""

    def __init__(self, output_dir='metrics'):
        """Inicializa o orquestrador de metricas.

        Cria o diretorio de saida, se necessario, e registra as funcoes
        disponiveis para serializar cada metrica.

        Args:
            output_dir (str, optional): Diretorio onde os arquivos JSONL serao
                salvos. Padrao: `"metrics"`.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.methods = {
            "accuracy_score": save_accuracy_score,
            "f1_score": save_f1_score,
            "roc_auc_score": save_roc_auc_score,
            "classification_report": save_classification_report,
            "mean_absolute_error": save_mean_absolute_error,
            "mean_squared_error": save_mean_squared_error,
            "r2_score": save_r2_score
        }

        self.metric_methods = list(self.methods.keys())

    def save_metric(self, metric_name, metric_value, model_name, dataset, undersampling=None, **kwargs):
        """Salva uma metrica especifica em arquivo JSONL.

        O arquivo recebe o nome da metrica, como `accuracy_score.jsonl`, e cada
        chamada adiciona uma nova linha JSON com o valor e os metadados da
        execucao.

        Args:
            metric_name (str): Nome da metrica a ser salva.
            metric_value: Valor ja calculado da metrica.
            model_name (str): Nome do modelo executado.
            dataset (str): Nome do dataset utilizado.
            undersampling (str, optional): Informacao sobre undersampling,
                quando aplicavel. Padrao: None.
            **kwargs: Argumentos adicionais aceitos pela funcao de registro da
                metrica, como `average` ou `multi_class`.

        Returns:
            dict: Dados da metrica salvos no arquivo.

        Raises:
            ValueError: Se `metric_name` nao estiver entre as metricas
            registradas.
        """
        if metric_name not in self.methods:
            raise ValueError(
                f"Metrica '{metric_name}' nao reconhecida. "
                f"Escolha entre: {self.metric_methods}"
            )

        metric_func = self.methods[metric_name]

        # Passa 'undersampling' explicitamente para a funcao de salvamento.
        metric_data = metric_func(metric_value, model_name, dataset, undersampling=undersampling, **kwargs)

        jsonl_file = self.output_dir / f"{metric_name}.jsonl"

        with open(jsonl_file, 'a', encoding='utf-8') as f:
            json.dump(metric_data, f, ensure_ascii=False)
            f.write('\n')

        # print(f"Metrica '{metric_name}' salva em: {jsonl_file}")

        return metric_data

    def save_all_metrics(self, metrics_dict, model_name, dataset, undersampling=None):
        """Salva todas as metricas reconhecidas presentes em um dicionario.

        Apenas as chaves conhecidas pelo orquestrador sao processadas. Chaves
        auxiliares como `f1_average`, `roc_average` e `roc_multi_class` sao
        usadas como parametros extras quando as metricas correspondentes estao
        presentes.

        Args:
            metrics_dict (dict): Dicionario contendo as metricas calculadas.
                Exemplo: {
                    'accuracy_score': 0.85,
                    'f1_score': 0.83,
                    'roc_auc_score': 0.90,
                    'classification_report': {...},
                    'mean_absolute_error': 1000.0,
                    'mean_squared_error': 2000000.0,
                    'r2_score': 0.75
                }
            model_name (str): Nome do modelo executado.
            dataset (str): Nome do dataset utilizado.
            undersampling (str, optional): Informacao sobre undersampling,
                quando aplicavel. Padrao: None.

        Returns:
            dict: Registros das metricas salvas, indexados pelo nome da
            metrica.
        """
        results = {}

        if 'accuracy_score' in metrics_dict:
            results['accuracy_score'] = self.save_metric(
                'accuracy_score',
                metrics_dict['accuracy_score'],
                model_name,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'f1_score' in metrics_dict:
            f1_average = metrics_dict.get('f1_average', 'weighted')
            results['f1_score'] = self.save_metric(
                'f1_score',
                metrics_dict['f1_score'],
                model_name,
                average=f1_average,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'roc_auc_score' in metrics_dict:
            roc_average = metrics_dict.get('roc_average', 'weighted')
            roc_multi_class = metrics_dict.get('roc_multi_class', 'ovr')
            results['roc_auc_score'] = self.save_metric(
                'roc_auc_score',
                metrics_dict['roc_auc_score'],
                model_name,
                average=roc_average,
                multi_class=roc_multi_class,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'classification_report' in metrics_dict:
            results['classification_report'] = self.save_metric(
                'classification_report',
                metrics_dict['classification_report'],
                model_name,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'mean_absolute_error' in metrics_dict:
            results['mean_absolute_error'] = self.save_metric(
                'mean_absolute_error',
                metrics_dict['mean_absolute_error'],
                model_name,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'mean_squared_error' in metrics_dict:
            results['mean_squared_error'] = self.save_metric(
                'mean_squared_error',
                metrics_dict['mean_squared_error'],
                model_name,
                dataset=dataset,
                undersampling=undersampling
            )

        if 'r2_score' in metrics_dict:
            results['r2_score'] = self.save_metric(
                'r2_score',
                metrics_dict['r2_score'],
                model_name,
                dataset=dataset,
                undersampling=undersampling
            )

        return results

    def load_metrics(self, metric_name):
        """Carrega as metricas salvas para um nome de metrica.

        Args:
            metric_name (str): Nome da metrica a ser carregada.

        Returns:
            list: Registros carregados do arquivo JSONL. Retorna lista vazia
            quando o arquivo nao existe.
        """
        jsonl_file = self.output_dir / f"{metric_name}.jsonl"

        if not jsonl_file.exists():
            print(f"Arquivo nao encontrado: {jsonl_file}")
            return []

        metrics = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                metrics.append(json.loads(line))

        return metrics

    def load_all_metrics(self):
        """Carrega todas as metricas registradas pelo orquestrador.

        Returns:
            dict: Dicionario em que cada chave e o nome da metrica e cada valor
            e a lista de registros carregados do respectivo arquivo JSONL.
        """
        all_metrics = {}

        for metric_name in self.metric_methods:
            all_metrics[metric_name] = self.load_metrics(metric_name)

        return all_metrics

    def get_metrics_summary(self, model_name=None):
        """Gera um resumo das metricas salvas.

        Para cada metrica com registros disponiveis, informa a quantidade de
        entradas, os modelos encontrados e o registro mais recente.

        Args:
            model_name (str, optional): Nome do modelo usado para filtrar os
                registros. Padrao: None.

        Returns:
            dict: Resumo das metricas disponiveis, opcionalmente filtrado por
            modelo.
        """
        all_metrics = self.load_all_metrics()
        summary = {}

        for metric_name, metrics_list in all_metrics.items():
            if model_name:
                metrics_list = [m for m in metrics_list if m['model_name'] == model_name]

            if metrics_list:
                summary[metric_name] = {
                    'count': len(metrics_list),
                    'models': list(set(m['model_name'] for m in metrics_list)),
                    'latest': metrics_list[-1] if metrics_list else None
                }

        return summary


if __name__ == "__main__":
   print("Evaluate Model carregado.")
