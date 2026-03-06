from imblearn.under_sampling import (
    TomekLinks, 
    OneSidedSelection, 
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    NeighbourhoodCleaningRule,
    NearMiss
    )

def apply_oneSidedSelection(X_train, y_train):
    """ Apply One Sided Selection undersampling technique."""
    
    oss = OneSidedSelection(
    sampling_strategy='auto',
    random_state=23,
    n_neighbors=1,
    n_jobs=-1)   

    X_resampled, y_resampled = oss.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_nearMissV1(X_train, y_train):
    """ Apply One Sided Selection undersampling technique."""
    
    oss = NearMiss(
    sampling_strategy='auto',
    version=1,
    n_neighbors=3,
    n_jobs=-1)   

    X_resampled, y_resampled = oss.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_nearMissV2(X_train, y_train):
    """ Apply One Sided Selection undersampling technique."""
    
    oss = NearMiss(
    sampling_strategy='auto',
    version=2,
    n_neighbors=3,
    n_jobs=-1)   

    X_resampled, y_resampled = oss.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_nearMissV3(X_train, y_train):
    """ Apply One Sided Selection undersampling technique."""
    
    oss = NearMiss(
    sampling_strategy='auto',
    version=3,
    n_neighbors=3,
    n_jobs=-1)   

    X_resampled, y_resampled = oss.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_editedNearestNeighbours(X_train, y_train):
    """ Apply Edited Nearest Neighbours undersampling technique."""
    enn  = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,    
    n_jobs=-1)

    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_neighbourhoodCleaningRule(X_train, y_train):
    """ Apply neighbourhood Cleaning Rule undersampling technique."""
    enn  = NeighbourhoodCleaningRule(
    sampling_strategy='auto',
    n_neighbors=3,   
    threshold_cleaning=0.5, 
    n_jobs=-1)

    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_tomekLinks(X_train, y_train):
    """ Apply Tomek Links undersampling technique."""
    
    tl = TomekLinks(
    sampling_strategy='auto',
    n_jobs=-1)

    X_resampled, y_resampled = tl.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_repeatedEditedNearestNeighbours(X_train, y_train):
    """ Apply repeated Nearest Neighbours undersampling technique."""
    
    renn  = RepeatedEditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,    
    kind_sel='all',
    max_iter=100,
    n_jobs=-1)

    X_resampled, y_resampled = renn.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

class UnderSampligOrchestrator:
    def __init__(self):
        self.methods = {
            "OneSidedSelection": apply_oneSidedSelection,
            "EditedNearestNeighbours": apply_editedNearestNeighbours,
            "TomekLinks": apply_tomekLinks,
            "RepeatedEditedNearestNeighbours": apply_repeatedEditedNearestNeighbours,
            "NeighbourhoodCleaningRule": apply_neighbourhoodCleaningRule,
            "NearMissV1": apply_nearMissV1,
            "NearMissV2": apply_nearMissV2,
            "NearMissV3": apply_nearMissV3,
        }
        
        under_samplig_methods = list(self.methods.keys())
        
    def apply(self, method_name, X_train, y_train):
        """
        Executa o método de undersampling escolhido.
        
        Args:
            method_name (str): O nome do método a ser aplicado.
            X_train: Dados de entrada.
            y_train: Rótulos de entrada.
            
        Returns:
            X_resampled, y_resampled: Dados após o balanceamento.
        """
        if method_name not in self.methods:
            raise ValueError(f"Método '{method_name}' não reconhecido. Escolha entre: {self.under_samplig_methods}")
        
        # Obtém a função correspondente e a executa
        sampling_func = self.methods[method_name]
        return sampling_func(X_train, y_train)
        
if __name__ == "__main__":
   print("UnderSampling carregado.")