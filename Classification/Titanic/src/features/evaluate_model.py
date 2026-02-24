
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    f1_score,     
    roc_auc_score)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    
    """Avalia o desempenho do modelo."""
    metrics = {}
    
    pred = model.predict(X_test)
    proba = np.where(model.predict_proba(X_test)[:,1]>=threshold,1,0)   
    
    
    metrics['classification_report'] = classification_report(y_test, pred, output_dict=True)
    metrics['accuracy_score']  = accuracy_score(y_test, pred)
    metrics['f1_score'] = f1_score(y_test, pred)
    # metrics['f1_average'] = 'weighted'
    metrics['roc_auc_score'] = roc_auc_score(y_test, proba)
    # metrics['roc_average'] = 'weighted'
    
    return metrics



def save_accuracy_score(y_true, y_pred, model_name, output_dir='metrics'):
    """Save accuracy score metric to JSONL file."""
    
    score = accuracy_score(y_true, y_pred)
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'accuracy_score',
        'score': float(score),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_f1_score(y_true, y_pred, model_name, average='weighted', output_dir='metrics'):
    """Save F1 score metric to JSONL file."""
    
    score = f1_score(y_true, y_pred, average=average)
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'f1_score',
        'average': average,
        'score': float(score),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_roc_auc_score(y_true, y_pred_proba, model_name, average='weighted', multi_class='ovr', output_dir='metrics'):
    """Save ROC AUC score metric to JSONL file."""
    
    score = roc_auc_score(y_true, y_pred_proba, average=average, multi_class=multi_class)
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'roc_auc_score',
        'average': average,
        'multi_class': multi_class,
        'score': float(score),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_classification_report(y_true, y_pred, model_name, output_dir='metrics'):
    """Save classification report metric to JSONL file."""
    
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'classification_report',
        'report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_accuracy_score(accuracy_value, model_name, output_dir='metrics'):
    """Save accuracy score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'accuracy_score',
        'score': float(accuracy_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_f1_score(f1_value, model_name, average='weighted', output_dir='metrics'):
    """Save F1 score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'f1_score',
        'average': average,
        'score': float(f1_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_roc_auc_score(roc_auc_value, model_name, average='weighted', multi_class='ovr', output_dir='metrics'):
    """Save ROC AUC score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'roc_auc_score',
        'average': average,
        'multi_class': multi_class,
        'score': float(roc_auc_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_classification_report(report_dict, model_name, output_dir='metrics'):
    """Save classification report metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'classification_report',
        'report': report_dict,
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data



def save_accuracy_score(accuracy_value, model_name):
    """Save accuracy score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'accuracy_score',
        'score': float(accuracy_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_f1_score(f1_value, model_name, average='weighted'):
    """Save F1 score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'f1_score',
        'average': average,
        'score': float(f1_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_roc_auc_score(roc_auc_value, model_name, average='weighted', multi_class='ovr'):
    """Save ROC AUC score metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'roc_auc_score',
        'average': average,
        'multi_class': multi_class,
        'score': float(roc_auc_value),
        'timestamp': datetime.now().isoformat()
    }
    
    return metric_data


def save_classification_report(report_dict, model_name):
    """Save classification report metric to JSONL file."""
    
    metric_data = {
        'model_name': model_name,
        'metric_name': 'classification_report',
        'report': report_dict,
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
        
    def save_metric(self, metric_name, metric_value, model_name, **kwargs):
        """
        Salva uma métrica específica em arquivo JSONL.
        
        Args:
            metric_name (str): Nome da métrica a ser salva.
            metric_value: Valor já calculado da métrica.
            model_name (str): Nome do modelo executado.
            **kwargs: Argumentos adicionais para a métrica específica.
            
        Returns:
            dict: Dados da métrica salvos.
        """
        if metric_name not in self.methods:
            raise ValueError(
                f"Métrica '{metric_name}' não reconhecida. "
                f"Escolha entre: {self.metric_methods}"
            )
        
        # Obtém a função correspondente
        metric_func = self.methods[metric_name]
        
        # Prepara os dados da métrica
        metric_data = metric_func(metric_value, model_name, **kwargs)
        
        # Define o caminho do arquivo JSONL
        jsonl_file = self.output_dir / f"{metric_name}.jsonl"
        
        # Salva em modo append (cumulativo)
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            json.dump(metric_data, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Métrica '{metric_name}' salva em: {jsonl_file}")
        
        return metric_data
    
    def save_all_metrics(self, metrics_dict, model_name):
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
            
        Returns:
            dict: Dicionário com todas as métricas salvas.
        """
        results = {}
        
        # Accuracy
        if 'accuracy_score' in metrics_dict:
            results['accuracy_score'] = self.save_metric(
                'accuracy_score', 
                metrics_dict['accuracy_score'], 
                model_name
            )
        
        # F1 Score
        if 'f1_score' in metrics_dict:
            f1_average = metrics_dict.get('f1_average', 'weighted')
            results['f1_score'] = self.save_metric(
                'f1_score', 
                metrics_dict['f1_score'], 
                model_name, 
                average=f1_average
            )
        
        # ROC AUC Score
        if 'roc_auc_score' in metrics_dict:
            roc_average = metrics_dict.get('roc_average', 'weighted')
            roc_multi_class = metrics_dict.get('roc_multi_class', 'ovr')
            results['roc_auc_score'] = self.save_metric(
                'roc_auc_score', 
                metrics_dict['roc_auc_score'], 
                model_name, 
                average=roc_average,
                multi_class=roc_multi_class
            )
        
        # Classification Report
        if 'classification_report' in metrics_dict:
            results['classification_report'] = self.save_metric(
                'classification_report', 
                metrics_dict['classification_report'], 
                model_name
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