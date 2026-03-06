import pandas as pd
from sklearn.pipeline import make_pipeline
import pickle



def train_model(X_train, y_train, clf_model, best_model_params):    
    
    # config
    seed_ = 23        

    # apply best params    
    clf_model.set_params(**best_model_params)      
    
    pipeline = make_pipeline(             
            clf_model)
    
    clf = pipeline.fit(X_train, y_train)

    return clf
    
    
def save_model(model, path):
        
    """Salva o modelo treinado em um arquivo."""
        
    with open(path, 'wb') as arquivo:
        pickle.dump(model, arquivo)
        
def save_pipeline(pipeline, path):
        
    """Salva o pipeline em um arquivo."""
        
    with open(path, 'wb') as arquivo:
        pickle.dump(pipeline, arquivo)
    
if __name__ == "__main__":    
    print("Train Model carregado.")