import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_categorical_data(df:pd.DataFrame, target, classification = True):
    """Plot categorical feature distributions and, optionally, target splits.

    Args:
        df (pd.DataFrame): Input dataset.
        target: Target column name used when `classification` is True.
        classification (bool, optional): If True, also plot target-conditioned
            distributions and normalized category frequencies. Defaults to True.

    Returns:
        None
    """
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
    """Plot numerical feature distributions and a correlation heatmap.

    Args:
        df (pd.DataFrame): Input dataset.
        target: Target column name used when `classification` is True.
        classification (bool, optional): If True, also plot KDE curves split by
            target. Defaults to True.

    Returns:
        None
    """
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
    """Plot and optionally save a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix to visualize.
        title: Plot title and output filename stem when saving.
        path (str, optional): Directory where the figure will be saved. If
            omitted, the plot is only displayed. Defaults to None.

    Returns:
        None
    """

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
    """Plot a bar chart for a pandas Series and optionally save it.

    Args:
        s (pd.Series): Series whose index is used for the x-axis and whose
            values are plotted on the y-axis.
        title: Plot title and output filename stem when saving.
        path (str, optional): Directory where the figure will be saved. If
            omitted, the plot is only displayed. Defaults to None.

    Returns:
        None
    """
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
    """Plot validation scores per fold and save the figure.

    Args:
        save_path: Full path where the figure will be saved.
        model_name: Model name used in the plot title.
        df (pd.DataFrame): DataFrame containing fold-level validation scores.
        X (str, optional): Column name for validation scores. Present for API
            compatibility, but the current implementation uses ``val_score``
            directly. Defaults to 'val_score'.
        y (str, optional): Column name for fold labels. Present for API
            compatibility, but the current implementation uses ``fold``
            directly. Defaults to 'fold'.

    Returns:
        None
    """
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
    """Plot the separation of predicted probabilities by target class.

    Args:
        save_path: Full path where the figure will be saved.
        model_name (str): Model name used in the plot title.
        df (pd.DataFrame): DataFrame containing a ``probability`` column.
        target (str): Target column name used for the hue grouping.

    Returns:
        None
    """
    sns.histplot(data=df, x='probability', hue=target)
    plt.title(f'separation plan {model_name}')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close() 
