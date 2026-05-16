# Kaggle

Repositorio com pipelines de ciencia de dados para competicoes do Kaggle. O projeto esta organizado por tipo de problema e competicao, com funcoes compartilhadas para preparacao de dados, engenharia de atributos, selecao de modelos, treinamento, avaliacao e geracao de submissao.

## Estrutura do projeto

```text
.
|-- Classification/
|   `-- Titanic/
|       |-- config/
|       |-- data/
|       |-- main/
|       |-- models/
|       |-- notebooks/
|       |-- reports/
|       `-- src/
|-- Regression/
|   `-- house_prices/
|       |-- config/
|       |-- data/
|       |-- main/
|       |-- models/
|       |-- notebook/
|       |-- reports/
|       `-- src/
|-- functions/
|-- utils/
`-- README.md
```

- `Classification/Titanic`: pipeline de classificacao para a competicao Titanic.
- `Regression/house_prices`: pipeline de regressao para a competicao House Prices.
- `functions`: funcoes reutilizaveis de treino, avaliacao, predicao, validacao cruzada, selecao de modelo e undersampling.
- `utils`: funcoes auxiliares para graficos e exportacao de resultados.

## Pre-requisitos

- Python 3.10 ou superior.
- Conta no Kaggle com aceite das regras das competicoes usadas.
- Credenciais da API do Kaggle configuradas localmente.
- Ambiente virtual Python recomendado.

Crie e ative um ambiente virtual no Windows/PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instale as dependencias usadas pelos scripts:

```powershell
pip install pandas numpy scikit-learn pyyaml kaggle matplotlib seaborn xgboost lightgbm imbalanced-learn feature-engine scipy tensorflow pyarrow
```

## Autenticacao no Kaggle

Os scripts de coleta usam a API oficial do Kaggle. Para autenticar:

1. Acesse sua conta no Kaggle.
2. Va em `Account`.
3. Gere o token da API em `Create New API Token`.
4. Salve o arquivo `kaggle.json` no diretorio correto:

No Windows:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

No Linux/macOS:

```text
~/.kaggle/kaggle.json
```

Importante: antes de baixar dados de uma competicao, aceite as regras da competicao no site do Kaggle.

## Configuracao de caminhos

Cada competicao possui arquivos YAML em `config/`. O arquivo `config.yaml` define `init_path` com caminho absoluto, por exemplo:

- `Classification/Titanic/config/config.yaml`
- `Regression/house_prices/config/config.yaml`

Se voce clonar ou mover o projeto para outro diretorio, atualize o valor de `init_path` antes de executar os scripts.

## Como executar o pipeline Titanic

Execute os comandos a partir da raiz do repositorio.

### 1. Baixar e extrair os dados

```powershell
python "Classification/Titanic/main/[01] Data_Gathering.py"
```

Este script baixa a competicao `titanic` e extrai os arquivos em:

```text
Classification/Titanic/data/raw
```

### 2. Criar features iniciais

```powershell
python "Classification/Titanic/main/[02] Feature_Creation.py"
```

Este script le `train.csv` e `test.csv` de `data/raw` e gera arquivos Parquet em:

```text
Classification/Titanic/data/processed
```

### 3. Analisar e selecionar features

Scripts disponiveis:

```powershell
python "Classification/Titanic/main/[03] Feature_Analysis.py"
python "Classification/Titanic/main/[04] Feature_Selection_pre.py"
python "Classification/Titanic/main/[06] Feature_Selection_filter.py"
```

Essas etapas geram analises, tabelas e graficos em `reports/` e ajudam a ajustar as configuracoes de features.

### 4. Aplicar feature engineering

```powershell
python "Classification/Titanic/main/[05] Feature_Eng.py"
```

Este script usa as configuracoes em `pipeline.yaml`, aplica os pipelines de preprocessamento e salva os datasets finais em:

```text
Classification/Titanic/data/feature_eng
```

### 5. Treinar modelos

Modelos single, voting, multi-model e ANN possuem scripts separados. Exemplos:

```powershell
python "Classification/Titanic/main/Main_Single_Model_lite.py"
python "Classification/Titanic/main/Main_voting_Model_lite.py"
python "Classification/Titanic/main/Main_Multi_Model_lite.py"
python "Classification/Titanic/main/Main_ANN_Model_lite.py"
```

Tambem existem variacoes com undersampling:

```powershell
python "Classification/Titanic/main/Main_Single_Model_lite_undersamplig.py"
python "Classification/Titanic/main/Main_Single_Model_lite_undersamplig_all.py"
python "Classification/Titanic/main/Main_Voting_Model_lite_undersamplig_all.py"
```

Os modelos treinados, metricas, predicoes e graficos sao salvos em subpastas de:

```text
Classification/Titanic/models
Classification/Titanic/reports
```

### 6. Gerar arquivo de submissao

Para gerar uma submissao com modelo single:

```powershell
python "Classification/Titanic/main/Main_single_Model_Submission.py"
```

Tambem ha scripts de submissao para voting model e ANN:

```powershell
python "Classification/Titanic/main/Main_voting_Model_Submission.py"
python "Classification/Titanic/main/Main_ANN_Submission.py"
```

Os arquivos CSV de submissao sao salvos em:

```text
Classification/Titanic/data/submission
```

## Como executar o pipeline House Prices

Execute os comandos a partir da raiz do repositorio.

### 1. Baixar e extrair os dados

```powershell
python "Regression/house_prices/main/[01] Data_Gathering.py"
```

Este script baixa a competicao `house-prices-advanced-regression-techniques` e extrai os arquivos em:

```text
Regression/house_prices/data/raw
```

### 2. Criar features iniciais

```powershell
python "Regression/house_prices/main/[02] Feature_Creation.py"
```

Este script le `train.csv` e `test.csv` de `data/raw` e gera arquivos Parquet em:

```text
Regression/house_prices/data/processed
```

### 3. Analisar e selecionar features

Scripts disponiveis:

```powershell
python "Regression/house_prices/main/[03] Feature_Analysis.py"
python "Regression/house_prices/main/[04] Feature_Selection_pre.py"
```

Essas etapas geram analises, tabelas e graficos em `reports/` e ajudam a ajustar as configuracoes de features.

### 4. Aplicar feature engineering

```powershell
python "Regression/house_prices/main/[05] Feature_Eng.py"
```

Este script usa as configuracoes em `pipeline.yaml`, aplica os pipelines de preprocessamento e salva os datasets finais em:

```text
Regression/house_prices/data/feature_eng
```

### 5. Treinar modelos

Scripts principais:

```powershell
python "Regression/house_prices/main/Main_Single_Model_lite.py"
python "Regression/house_prices/main/Main_voting_Model_lite.py"
python "Regression/house_prices/main/Main_ANN_Model_lite.py"
```

Os modelos treinados, metricas, predicoes e graficos sao salvos em subpastas de:

```text
Regression/house_prices/models
Regression/house_prices/reports
```

### 6. Gerar arquivo de submissao

```powershell
python "Regression/house_prices/main/Main_single_Model_Submission.py"
```

O arquivo CSV de submissao e salvo em:

```text
Regression/house_prices/data/submission
```

## Saidas geradas

Cada competicao segue a mesma convencao geral:

- `data/raw`: arquivos originais baixados do Kaggle.
- `data/processed`: datasets intermediarios com features iniciais.
- `data/feature_eng`: datasets finais apos preprocessamento e engenharia de atributos.
- `data/submission`: arquivos CSV prontos para envio ao Kaggle.
- `models/...`: modelos treinados, predicoes, metricas e graficos por tipo de modelo.
- `reports/plots`: graficos de analise exploratoria e avaliacao.
- `reports/jsonl`: metricas e tabelas exportadas em JSONL.

## Observacoes importantes

- Os dados, modelos treinados, arquivos `.csv`, `.parquet`, `.json`, `.yaml`, logs e artefatos sao ignorados pelo `.gitignore`. Eles precisam ser baixados ou gerados localmente.
- Os scripts de treino dependem dos arquivos gerados nas etapas anteriores. Execute o pipeline na ordem: download, feature creation, feature engineering, treino e submissao.
- Os scripts `Main_*_Submission.py` esperam que o modelo correspondente ja exista em `models/...`.
- Alguns scripts usam parametros fixos no bloco `if __name__ == "__main__"`. Para testar outro modelo, pipeline ou threshold, ajuste esses argumentos diretamente no script ou chame a funcao principal a partir de outro arquivo.
- Os notebooks em `notebooks/` e `notebook/` servem para exploracao, analise e experimentacao, mas o fluxo reproduzivel principal esta nos scripts em `main/`.

## Licenca

Consulte o arquivo `LICENSE` para os termos de uso do projeto.
