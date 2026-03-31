import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from feature_engine.selection import DropCorrelatedFeatures, SmartCorrelatedSelection
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import SelectBySingleFeaturePerformance

def apply_MutualInformation_reg(X_train, y_train):
    
    """ Apply Mutual information technique."""
    
    mi = mutual_info_regression(X_train, y_train)
    mi = pd.Series(mi, name = 'mutual information')
    mi.index = X_train.columns
    mi.sort_values(ascending=False, inplace=True)
    
    return mi


def apply_MutualInformation_classif(X_train, y_train):
    
    """ Apply Mutual information technique."""
    
    mi = mutual_info_classif(X_train, y_train)
    mi = pd.Series(mi, name = 'mutual information')
    mi.index = X_train.columns
    mi.sort_values(ascending=False, inplace=True)
    
    return mi


def apply_Anova(X_train, y_train):
    
    """ Apply Anova technique."""
    
    select = X_train.columns.str.contains("numerical")
    cols = X_train.columns
    anova = f_classif(X_train[cols[select]], y_train)
    s = pd.Series(anova[1], index=cols[select], name='Anova')      
    s.sort_values(ascending=True, inplace=True) 
    
    return s

def apply_QuiSquare(X_train, y_train):
    
    """ Apply Qui Square technique."""
    
    categorical_col = list(X_train.select_dtypes(include=['category','object', 'bool']).columns)
    
    chi_ls = []
    
    for feature in categorical_col:
        print("Feature:", feature)
        # create contingency table
        arr1 = np.array((y_train.values.flatten()))
        arr2 = np.array(X_train[feature].values.flatten())
        c = pd.crosstab(arr1, arr2)        
        # chi_test
        p_value = stats.chi2_contingency(c)[1]
        chi_ls.append(p_value)
        
    chi = pd.Series(chi_ls, index=categorical_col, name='QuiSquare')    
    chi_sorted = chi.sort_values(ascending=True)
    
    return chi_sorted

def apply_PearsonCorrelation(X_train, y_train):
    
    """ Apply paerson Correlation technique."""
    
    numerical_col = list(X_train.select_dtypes(include=['number']).columns)
    
    corr_matrix = X_train[numerical_col].corr(method='pearson')
    
    return corr_matrix

def apply_SmartCorrelatedSelection(X_train, y_train):
    
    """ Apply paerson Correlation technique."""
    
    numerical_col = list(X_train.select_dtypes(include=['number']).columns)
    
   
    rf = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    # correlation selector
    sel = SmartCorrelatedSelection(
        variables=numerical_col, # if none, selector examines all numerical variables
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=rf,
        scoring="roc_auc",
        cv=3,
    )

    sel.fit(X_train, y_train.values.ravel())
    
    corr_features = list(sel.correlated_feature_sets_[0])
    
    return {"corr_feature" : corr_features, "corr_2_drop" : sel.features_to_drop_}


class FeatureSelectionOrchestrator:
    def __init__(self):
        self.methods = {
            "QuiSquare": apply_QuiSquare,
            "Anova": apply_Anova,
            "MutualInformationClassif": apply_MutualInformation_classif,
            "MutualInformationReg": apply_MutualInformation_reg,
            "PearsonCorrelation": apply_PearsonCorrelation,
            "SmartCorrelatedSelection": apply_SmartCorrelatedSelection
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
    
def SelectSingleFeature(clf, metric, X_train:pd.DataFrame, y_train: pd.DataFrame, threshold = 0.5):    

# set up the selector
    sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=clf,
    scoring=metric,
    cv=3,
    threshold=threshold,
)

    # find predictive features
    sel.fit(X_train, y_train)
    
    return [sel.feature_performance_, sel.feature_performance_std_]
        
if __name__ == "__main__":
   print("Feature Selection carregado.")