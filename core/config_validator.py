"""
config_validator.py

Responsável pela validação estrutural dos arquivos de configuração
do projeto edge-vision-model.

Este módulo verifica apenas a existência e a estrutura mínima
dos arquivos YAML, sem executar lógica de treinamento ou avaliação.
"""

from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    """
    Carrega um arquivo YAML e retorna seu conteúdo como dicionário.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_dataset_config(config: dict) -> None:
    """
    Valida a estrutura mínima do dataset.yaml.
    """

    required_keys = ["dataset"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente no dataset.yaml: {key}")

    dataset = config["dataset"]

    for field in ["root_dir", "splits", "structure", "classes"]:
        if field not in dataset:
            raise ValueError(f"Campo obrigatório ausente em dataset.yaml: {field}")


def validate_metrics_config(config: dict) -> None:
    """
    Valida a estrutura mínima do metrics.yaml.
    """

    required_keys = ["evaluation"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente em metrics.yaml: {key}")

    evaluation = config["evaluation"]

    if "metrics" not in evaluation or not isinstance(evaluation["metrics"], list):
        raise ValueError("Lista de métricas inválida ou ausente em metrics.yaml")


def validate_model_config(config: dict) -> None:
    """
    Valida a estrutura mínima de um arquivo de configuração de modelo.
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