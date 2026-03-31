import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    f1_score,     
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Avalia o desempenho do modelo."""
    metrics = {}
    
    pred = model.predict(X_test)
    # Para roc_auc_score, geralmente usamos as probabilidades brutas, não binarizadas.
    # Se a intenção é usar a predição binarizada com um threshold específico, o nome da variável 'proba' pode ser confuso.
    # Mantendo a lógica original, 'proba' aqui representa a predição binarizada com base no threshold.
    proba_raw = model.predict_proba(X_test)[:, 1]
    proba_binarized = np.where(proba_raw >= threshold, 1, 0)
    
    metrics['classification_report'] = classification_report(y_test, pred, output_dict=True)
    metrics['accuracy_score'] = accuracy_score(y_test, pred)
    metrics['f1_score'] = f1_score(y_test, pred)
    metrics['roc_auc_score'] = roc_auc_score(y_test, proba_raw) # Usando probabilidades brutas para ROC AUC
    
    return metrics


def _create_metric_data(metric_name, model_name, dataset, score, undersampling=None, **kwargs):
    """Função auxiliar para criar o dicionário de dados da métrica."""
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
    """Salva a métrica de acurácia em um dicionário de dados."""
    return _create_metric_data('accuracy_score', model_name, dataset, accuracy_value, undersampling=undersampling)


def save_f1_score(f1_value, model_name, dataset, average='weighted', undersampling=None):
    """Salva a métrica F1-score em um dicionário de dados."""
    return _create_metric_data('f1_score', model_name, dataset, f1_value, undersampling=undersampling, average=average)


def save_roc_auc_score(roc_auc_value, model_name, dataset, average='weighted', multi_class='ovr', undersampling=None):
    """Salva a métrica ROC AUC em um dicionário de dados."""
    return _create_metric_data('roc_auc_score', model_name, dataset, roc_auc_value, undersampling=undersampling, average=average, multi_class=multi_class)


def save_classification_report(report_dict, model_name, dataset, undersampling=None):
    """Salva o relatório de classificação em um dicionário de dados."""
    metric_data = {
        'model_name': model_name,
        'dataset': dataset,
        'metric_name': 'classification_report',
        'report': report_dict,
        'undersampling': undersampling,
        'timestamp': datetime.now().isoformat()
    }
    return metric_data


class MetricsOrchestrator:
    def __init__(self, output_dir='metrics'):
        """
        Inicializa o orquestrador de métricas.
        
        Args:
            output_dir (str): Diretório onde os arquivos JSONL serão salvos.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.methods = {
            "accuracy_score": save_accuracy_score,
            "f1_score": save_f1_score,
            "roc_auc_score": save_roc_auc_score,
            "classification_report": save_classification_report
        }
        
        self.metric_methods = list(self.methods.keys())
        
    def save_metric(self, metric_name, metric_value, model_name, dataset, undersampling=None, **kwargs):
        """
        Salva uma métrica específica em arquivo JSONL.
        
        Args:
            metric_name (str): Nome da métrica a ser salva.
            metric_value: Valor já calculado da métrica.
            model_name (str): Nome do modelo executado.
            dataset (str): Nome do dataset utilizado.
            undersampling (str, optional): Informação sobre undersampling, se aplicável.
            **kwargs: Argumentos adicionais para a métrica específica (ex: 'average', 'multi_class').
            
        Returns:
            dict: Dados da métrica salvos.
        """
        if metric_name not in self.methods:
            raise ValueError(
                f"Métrica '{metric_name}' não reconhecida. "
                f"Escolha entre: {self.metric_methods}"
            )
        
        metric_func = self.methods[metric_name]
        
        # Passa 'undersampling' explicitamente para a função de salvamento
        metric_data = metric_func(metric_value, model_name, dataset, undersampling=undersampling, **kwargs)
        
        jsonl_file = self.output_dir / f"{metric_name}.jsonl"
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            json.dump(metric_data, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Métrica '{metric_name}' salva em: {jsonl_file}")
        
        return metric_data
    
    def save_all_metrics(self, metrics_dict, model_name, dataset, undersampling=None):
        """
        Salva todas as métricas disponíveis a partir de um dicionário.
        
        Args:
            metrics_dict (dict): Dicionário contendo as métricas calculadas.
                Exemplo: {
                    'accuracy_score': 0.85,
                    'f1_score': 0.83,
                    'roc_auc_score': 0.90,
                    'classification_report': {...}
                }
            model_name (str): Nome do modelo executado.
            dataset (str): Nome do dataset utilizado.
            undersampling (str, optional): Informação sobre undersampling, se aplicável.
            
        Returns:
            dict: Dicionário com todas as métricas salvas.
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
            
        return results
    
    def load_metrics(self, metric_name):
        """
        Carrega todas as métricas salvas de um arquivo JSONL específico.
        
        Args:
            metric_name (str): Nome da métrica a ser carregada.
            
        Returns:
            list: Lista de dicionários com as métricas.
        """
        jsonl_file = self.output_dir / f"{metric_name}.jsonl"
        
        if not jsonl_file.exists():
            print(f"Arquivo não encontrado: {jsonl_file}")
            return []
        
        metrics = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        return metrics
    
    def load_all_metrics(self):
        """
        Carrega todas as métricas de todos os arquivos JSONL.
        
        Returns:
            dict: Dicionário com todas as métricas carregadas.
        """
        all_metrics = {}
        
        for metric_name in self.metric_methods:
            all_metrics[metric_name] = self.load_metrics(metric_name)
        
        return all_metrics
    
    def get_metrics_summary(self, model_name=None):
        """
        Retorna um resumo das métricas salvas.
        
        Args:
            model_name (str, optional): Filtrar por nome do modelo.
            
        Returns:
            dict: Resumo das métricas.
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