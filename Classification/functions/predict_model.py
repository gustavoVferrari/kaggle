import pandas as pd
import pickle


def load_trained_model(model_path):
    """Carrega um modelo salvo do disco."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
        return model


def make_prediction(model, input_data):
    """Realiza predições em novos dados."""    
   
    
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

if __name__ == "__main__":
    print("Módulo de predição carregado.")