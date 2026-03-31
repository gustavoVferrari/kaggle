import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix)
import os


def threshold_optimization(y_true, pred_proba:np.array, model_name=None, path=None):
    
    dict_tpr = {}
    dict_fpr = {}

    for i in np.linspace(0.01,1,100):
        pred_threshold = np.where(pred_proba >= i, 1, 0)
        cm = confusion_matrix(y_true, pred_threshold)
        
        tp = cm[1,1] # true positive rate
        tn = cm[0,0] # true negative rate
        fn = cm[1,0] # false positive rate
        fp = cm[0,1] # false positive rate

        dict_tpr[i] = tp / (tp + fn)
        dict_fpr[i] = 1 - (tn / (tn + fp))
        
    np_tpr = np.array(list(dict_tpr.values()))
    np_fpr = np.array(list(dict_fpr.values()))
    np_tpr = np.sort(np_tpr)
    np_fpr = np.sort(np_fpr)

    idx = np.argmax(np_tpr - np_fpr) # max point
    
    print(f'True Positive Rate {np_tpr[idx]}')
    print(f'False Positive rate {np_fpr[idx]}')
    
    df_tpr = pd.DataFrame(
        dict_tpr.values(), 
        index=dict_tpr.keys(), 
        columns=['TPR'])
    df_fpr = pd.DataFrame(
        dict_fpr.values(), 
        index=dict_fpr.keys(), 
        columns=['FPR'])
    df_threshold = df_fpr.join(df_tpr)
    
    df_threshold['diff'] = df_threshold['TPR'] - df_threshold['FPR'] 
    df_threshold.reset_index(inplace=True)
    df_threshold.rename(
        columns={'index':'threshold'}, 
        inplace=True)

    thresh_idx = int(df_threshold['diff'].argmax())
    thresh = df_threshold.iloc[thresh_idx]['threshold']

    print(f'ponto de corte ótimo {thresh}')
    
    if path != None:
        save_path = os.path.join(
            path,
            f"threshold_optimization_{model_name}.png"
    )
         
        plt.figure(figsize=(12,6))
        plt.title('ROC Curve')
        plt.plot(
            df_threshold['FPR'], 
            df_threshold['TPR'], 
            linestyle='--', 
            label = model_name)

        plt.scatter(
            df_threshold.iloc[thresh_idx]['FPR'] ,
            df_threshold.iloc[thresh_idx]['TPR'], 
            color = 'red')

        plt.axhline(
            y = df_threshold.iloc[thresh_idx]['TPR'], 
            linestyle = '--',
            alpha = 0.4, 
            color = 'black')
        plt.axvline(
            x = df_threshold.iloc[thresh_idx]['FPR'], 
            linestyle = '--', 
            alpha = 0.4, 
            color = 'black')

        plt.text(
            x = df_threshold.iloc[thresh_idx]['FPR'], 
            y = df_threshold.iloc[thresh_idx]['TPR'], 
            s = np.round(thresh,3))

        plt.xlabel('1 - specificity (False Positive rate)')
        plt.ylabel('sensitivity (True Positive Rate)')
        plt.legend()
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches="tight"
            )
        plt.close() 
    
    return thresh