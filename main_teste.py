"""
main.py

Entrada mínima para validação de configurações do projeto.
"""

import logging
from pathlib import Path

from utils.logging_global import setup_logging
from core.config_validator import (
    load_yaml,
    validate_dataset_config,
    validate_metrics_config,
    validate_model_config,
)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando validação das configurações")

    # ============================
    # DATASET
    # ============================
    dataset_config_path = Path("config/dataset.yaml")
    dataset_config = load_yaml(dataset_config_path)
    validate_dataset_config(dataset_config)
    logger.info("dataset.yaml validado com sucesso")

    # ============================
    # METRICS
    # ============================
    metrics_config_path = Path("config/metrics.yaml")
    metrics_config = load_yaml(metrics_config_path)
    validate_metrics_config(metrics_config)
    logger.info("metrics.yaml validado com sucesso")

    # ============================
    # MODEL (YOLO)
    # ============================
    yolo_config_path = Path("config/models/yolo.yaml")
    yolo_config = load_yaml(yolo_config_path)
    validate_model_config(yolo_config)
    logger.info("yolo.yaml validado com sucesso")

    logger.info("Validação de configurações concluída com sucesso")


if __name__ == "__main__":
    main()