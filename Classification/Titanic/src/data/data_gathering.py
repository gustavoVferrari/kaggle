# read data from kaggle

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Authetication
api = KaggleApi()
api.authenticate()

def DataGatheting(competition, path):
    
    """Função para baixar os dados do Kaggle"""
   
    return api.competition_download_files(
        competition=competition,
        path=path)     

def Unzipdata(input_path, output_path):       
    
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
             

    
if __name__ == "__main__":
   print("Módulo de Data Gathering carregado.")