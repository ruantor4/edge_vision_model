"""
settings.py

Arquivo de configuração central do projeto edge-vision-model.

Responsabilidades:
- Definir paths do projeto
- Centralizar referências ao dataset externo (somente leitura)
- Definir diretórios de artifacts da modelagem
- Centralizar parâmetros globais da pipeline de modelagem e avaliação
"""
from pathlib import Path

# ROOT DO PROJETO

ROOT_DIR = Path(__file__).resolve().parent.parent


# DATASET EXTERNO

# Diretório raiz onde o dataset original está armazenado
DATASET_DIR = ROOT_DIR.parent / "dataset_original"

# Splits esperados do dataset
DATASET_SPLITS = ("train", "valid", "test")

# Estrutura do dataset
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"


# ARTIFACTS GERADOS PELA MODELAGEM

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Dados preparados por algoritimo
ARTIFACTS_PREPARED_DATA_DIR = ARTIFACTS_DIR / "prepared_data"

# Pesos e modelos de treinamento
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"

# Métricas de avaliação dos modelos
ARTIFACTS_METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Resultados comparativos entre modelos
ARTIFACTS_COMPARISONS_DIR = ARTIFACTS_DIR / "comparisons"


# LOGS

LOGS_DIR = ROOT_DIR / "logs"


# PARÂMETROS GLOBAIS DA PIPELINE DE MODELAGEM E AVALIAÇÃO

# Flag para ativar/desativar a etapa de preparação de dados
ENABLE_DATA_PREPARATION = True

# Flag para ativar/desativar a etapa de treinamento
ENABLE_TRAINING = True

# Flag para ativar/desativar a etapa de avaliação
ENABLE_EVALUATION = True

# Flag para ativar/desativar a etapa de comparação entre modelos
ENABLE_COMPARISON = True


# CONFIGURAÇÕES DE EXECUÇÃO

# Semente global para reprodutibilidade (quando aplicável)
RANDOM_SEED = 42

# Indica se os artefatos existentes podem ser sobrescritos
ALLOW_OVERWRITE_ARTIFACTS = False