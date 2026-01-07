"""
config_validator.py

Responsável pela validação estrutural dos arquivos de configuração
do projeto edge-vision-model.

Este módulo realiza apenas validações estáticas e estruturais
dos arquivos YAML de configuração, garantindo que a pipeline
tenha os contratos mínimos necessários antes da execução.

Escopo:
- Carregamento seguro de arquivos YAML
- Validação estrutural de dataset.yaml
- Validação estrutural de metrics.yaml
- Validação estrutural de arquivos de configuração de modelos
"""

from pathlib import Path
import yaml

# ============================================================
# UTILITÁRIOS DE CARREGAMENTO
# ============================================================

def load_yaml(path: Path) -> dict:
    """
    Carrega um arquivo YAML e retorna seu conteúdo como dicionário.

    Args:
        path (Path): Caminho para o arquivo YAML.

    Returns:
        dict: Conteúdo do arquivo YAML.

    Raises:
        FileNotFoundError: Caso o arquivo não exista.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

# ============================================================
# VALIDAÇÃO DE DATASET
# ============================================================

def validate_dataset_config(config: dict) -> None:
    """
    Valida a estrutura mínima do arquivo dataset.yaml.

    Verifica apenas a existência das chaves obrigatórias,
    sem validar valores ou caminhos físicos.

    Args:
        config (dict): Conteúdo carregado do dataset.yaml.

    Raises:
        ValueError: Caso alguma chave obrigatória esteja ausente.
    """
    required_keys = ["dataset"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente no dataset.yaml: {key}")

    dataset = config["dataset"]

    for field in ["root_dir", "splits", "structure", "classes"]:
        if field not in dataset:
            raise ValueError(f"Campo obrigatório ausente em dataset.yaml: {field}")


# ============================================================
# VALIDAÇÃO DE MÉTRICAS
# ============================================================

def validate_metrics_config(config: dict) -> None:
    """
    Valida a estrutura mínima do arquivo metrics.yaml.

    Garante a existência do bloco de avaliação e da lista
    de métricas esperadas.

    Args:
        config (dict): Conteúdo carregado do metrics.yaml.

    Raises:
        ValueError: Caso a estrutura mínima não seja atendida.
    """
    required_keys = ["evaluation"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente em metrics.yaml: {key}")

    evaluation = config["evaluation"]

    if "metrics" not in evaluation or not isinstance(evaluation["metrics"], list):
        raise ValueError("Lista de métricas inválida ou ausente em metrics.yaml")

# ============================================================
# VALIDAÇÃO DE CONFIGURAÇÃO DE MODELO
# ============================================================

def validate_model_config(config: dict) -> None:
    """
    Valida a estrutura mínima de um arquivo de configuração de modelo.

    Esta validação é genérica e se aplica a arquivos como:
    - yolo.yaml
    - ssd.yaml
    - faster_rcnn.yaml

    Args:
        config (dict): Conteúdo carregado do arquivo de configuração do modelo.

    Raises:
        ValueError: Caso alguma chave estrutural obrigatória esteja ausente.
    """
    required_keys = ["model", "dataset", "training", "evaluation", "artifacts"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente no model config: {key}")

    model = config["model"]
    if "name" not in model:
        raise ValueError("Campo 'model.name' ausente no model config")

    training = config["training"]
    for field in ["preferred_device", "fallback_device"]:
        if field not in training:
            raise ValueError(f"Campo obrigatório ausente em training: {field}")