import pickle
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier



def train_model(X_train, y_train, clf_model, best_model_params):    
    
    # config
    seed_ = 23        

    # apply best params    
    clf_model.set_params(**best_model_params)      
    
    pipeline = make_pipeline(             
            clf_model)
    
    clf = pipeline.fit(X_train, y_train)

    return clf

def train_voting_model(X_train, y_train, models, best_models_params):    
    
    seed_ = 23       
   
    voting = VotingClassifier(
        estimators=[
            ('rf', models['rf'].set_params(**best_models_params['rf'], random_state=seed_)),
            ('ab', models['ab'].set_params(**best_models_params['ab'], random_state=seed_)),
            ('gb', models['gb'].set_params(**best_models_params['gb'], random_state=seed_)),
            ('lr', models['lr'].set_params(**best_models_params['lr'], random_state=seed_)),
            ('svc', models['sv'].set_params(**best_models_params['sv'], random_state=seed_)),
            ('mlc', models['ml'].set_params(**best_models_params['ml'], random_state=seed_)),
            ('hgc', models['hg'].set_params(**best_models_params['hg'], random_state=seed_)),
            ],
        voting='soft',
        flatten_transform=True)
    
    pipeline = make_pipeline(
            # PCA(n_components=0.9, svd_solver='full'), 
            voting)
    
    voting_fitted = pipeline.fit(X_train, y_train) 
    
    return voting_fitted
    
    
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