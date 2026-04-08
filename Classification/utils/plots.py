import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_categorical_data(df:pd.DataFrame, target, classification = True):
    
    categorical_col = df.select_dtypes(include=['category','object', 'bool']).columns    
    
    for col in categorical_col:
        plt.figure(figsize=(15,6))
        plt.title(f"{col}")
        sns.countplot(x = col, data = df)
        plt.show()
        
    if classification == True:
        
        for col in categorical_col:
            crostab = pd.crosstab(
                index=df[col], 
                columns=df[target], 
                normalize='index'
                )
            
            crostab.plot(
                kind='bar',
                figsize=(15,6),
                stacked=True,
                title=col
                );
            plt.show()
            
        for col in categorical_col:
            (df.loc[:,[col]]
            .value_counts(normalize=True)
            .sort_values()
            .plot.bar(figsize=(12,6)))
            plt.axhline(0.05, c='red')
            plt.show()


def plot_numerical_data(df:pd.DataFrame, target, classification = True):
    
    numerical_col = df.select_dtypes(include=['number']).columns    
    
    for col in numerical_col:
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize = (15,4))

        axes[0].set_title("Histogram " + col)
        axes[0].hist(df[col], bins=30)
        axes[0].grid()

        axes[1].set_title("Boxplot " + col)
        axes[1].boxplot(df[col])
        axes[1].grid()
        plt.show()
    
    if classification is True:
        for col in numerical_col:    
            sns.kdeplot(
                data=df, 
                x=col, 
                hue=target)
            plt.show()
    
    
    corr_matrix = df[numerical_col].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))
    mask = mask[1:, :-1]

    plt.figure(figsize=(12,12))
    plt.title("Correlation Matrix")
    sns.heatmap(corr_matrix.iloc[1:,:-1], 
                mask=mask , 
                annot=True, 
                cmap='flare', 
                linewidths=2, 
                square=True);
    plt.show()
    

def Pearson_correlation(corr_matrix, title, path=None):    

    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))
    mask = mask[1:, :-1]

    plt.figure(figsize=(12,12))
    plt.title("Correlation Matrix")
    sns.heatmap(corr_matrix.iloc[1:,:-1], 
                mask=mask , 
                annot=True, 
                cmap='flare', 
                linewidths=2, 
                square=True);
    
    if path != None:
        save_path = os.path.join(
            path,
            f"{title}.png"
    )
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches="tight"
            )
        plt.close()  
    
def Bar_plot(s:pd.Series, title, path=None):    
    
    plt.figure(figsize=(20, 10))
    plt.bar(s.index, s.values, color="skyblue")
    plt.xticks(rotation=90)
    plt.axhline(y=0.05, color='r', linestyle='-')  
    plt.ylabel("p value")
    plt.title(title)
    
    if path != None:
        save_path = os.path.join(
            path,
            f"{title}.png"
    )
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches="tight"
            )
        plt.close()  
        
def cross_validation_plot(
    save_path,
    model_name, 
    df:pd.DataFrame, 
    X:str='val_score', 
    y:str='fold'
    ):
    
    sns.pointplot(
        data=df, 
        y='val_score', 
        x='fold')
    
    plt.title(f'score per fold {model_name}')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def separation_plan_plot(
    save_path,
    model_name:str,
    df:pd.DataFrame,     
    target:str):

    sns.histplot(data=df, x='probability', hue=target)
    plt.title(f'separation plan {model_name}')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close() 