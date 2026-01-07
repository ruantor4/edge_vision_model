"""
settings.py

Arquivo de configuração central do projeto edge-vision-model.

Este módulo atua como a fonte única de verdade (single source of truth)
para parâmetros globais, paths e flags de controle da pipeline.

Responsabilidades:
- Definir o diretório raiz do projeto
- Centralizar referências ao dataset externo (somente leitura)
- Declarar a estrutura de artifacts gerados pela modelagem
- Centralizar parâmetros globais da pipeline de treinamento, avaliação
  e comparação entre modelos

"""
from pathlib import Path

# ============================================================
# ROOT DO PROJETO
# ============================================================


# Diretório raiz do projeto 
ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATASET EXTERNO
# ============================================================

# Diretório raiz onde o dataset original está armazenado
DATASET_DIR = ROOT_DIR.parent / "dataset_original"

# Splits esperados do dataset
DATASET_SPLITS = ("train", "valid", "test")

# Estrutura do dataset
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"

# ============================================================
# ARTIFACTS GERADOS PELA MODELAGEM
# ============================================================

# Diretório raiz de artifacts do projeto
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Datasets preparados por algoritmo (YOLO, SSD, Faster R-CNN)
ARTIFACTS_PREPARED_DATA_DIR = ARTIFACTS_DIR / "prepared_data"

# Pesos e modelos de treinamento
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"

# Métricas oficiais de avaliação (COCO)
ARTIFACTS_METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Resultados comparativos entre modelos
ARTIFACTS_COMPARISONS_DIR = ARTIFACTS_DIR / "comparisons"

# ============================================================
# LOGS
# ============================================================

# Diretório central de logs da aplicação
LOGS_DIR = ROOT_DIR / "logs"

# ============================================================
# PARÂMETROS GLOBAIS DA PIPELINE
# ============================================================

# Flag para ativar/desativar a etapa de preparação de dados
ENABLE_DATA_PREPARATION = True

# Flag para ativar/desativar a etapa de treinamento
ENABLE_TRAINING = True

# Flag para ativar/desativar a etapa de avaliação
ENABLE_EVALUATION = True

# Flag para ativar/desativar a etapa de comparação entre modelos
ENABLE_COMPARISON = True

# ============================================================
# CONFIGURAÇÕES DE EXECUÇÃO
# ============================================================

# Seed global para reprodutibilidade (quando aplicável)
RANDOM_SEED = 42

# Indica se os artefatos existentes podem ser sobrescritos
ALLOW_OVERWRITE_ARTIFACTS = False

# ============================================================
# PATHS DOS ARQUIVOS DE CONFIGURAÇÃO
# ============================================================

# Diretório de arquivos de configuração
CONFIG_DIR = ROOT_DIR / "config"

# Configurações específicas de modelos
CONFIG_MODELS_DIR = CONFIG_DIR / "models"

YOLO_CONFIG_PATH = CONFIG_MODELS_DIR / "yolo.yaml"
SSD_CONFIG_PATH = CONFIG_MODELS_DIR / "ssd.yaml"
FASTER_RCNN_CONFIG_PATH = CONFIG_MODELS_DIR / "faster_rcnn.yaml"

# Configurações globais
DATASET_CONFIG_PATH = CONFIG_DIR / "dataset.yaml"
METRICS_CONFIG_PATH = CONFIG_DIR / "metrics.yaml"

# ============================================================
# ARTEFATOS FINAIS CONSOLIDADOS
# ============================================================

# CSV final consolidado (métricas + custo computacional)
ARTIFACTS_FINAL_COMPARISON_CSV = (
    ARTIFACTS_COMPARISONS_DIR / "models_final_comparison.csv"
)

# Diretório para plots comparativos e visualizações
ARTIFACTS_PLOTS_DIR = ARTIFACTS_COMPARISONS_DIR / "plots"
