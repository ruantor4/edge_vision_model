"""
config_validator.py

Validação estrutural dos arquivos de configuração do projeto.
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
    required_keys = ["dataset"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente no dataset.yaml: {key}")

    dataset = config["dataset"]

    for field in ["root_dir", "splits", "structure", "classes"]:
        if field not in dataset:
            raise ValueError(f"Campo obrigatório ausente em dataset.yaml: {field}")


def validate_metrics_config(config: dict) -> None:
    required_keys = ["evaluation"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Chave obrigatória ausente em metrics.yaml: {key}")

    evaluation = config["evaluation"]

    if "metrics" not in evaluation or not isinstance(evaluation["metrics"], list):
        raise ValueError("Lista de métricas inválida ou ausente em metrics.yaml")
