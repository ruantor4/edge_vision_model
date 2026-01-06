"""
run_prepare.py

Interface de linha de comando para executar a
ETAPA DE PREPARAÇÃO DO DATASET da pipeline edge-vision-model.

Este script:
- configura logging
- valida configurações globais
- executa a preparação e organização do dataset por arquitetura
"""

import logging
from pathlib import Path

from utils.logging_global import setup_logging

from core.config_validator import (
    load_yaml,
    validate_dataset_config,
)

from core.dataset_preparer import prepare_dataset

from config.settings import (
    DATASET_CONFIG_PATH,
    LOGS_DIR,
)


SUPPORTED_MODELS = ("yolo", "ssd", "faster_rcnn")


def main() -> None:
    # ============================
    # ETAPA 0 – PREPARAÇÃO DO AMBIENTE
    # ============================
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================
    # ETAPA 1 – LOGGING
    # ============================
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando preparação do dataset - Edge Vision Model")

    # ============================
    # ETAPA 2 – VALIDAÇÃO DE CONFIGURAÇÕES
    # ============================
    dataset_config = load_yaml(Path(DATASET_CONFIG_PATH))
    validate_dataset_config(dataset_config)

    logger.info("Configurações validadas com sucesso")

    # ============================
    # ETAPA 3 – PREPARAÇÃO DO DATASET
    # ============================
    logger.info("Executando preparação e organização do dataset")

    for model_name in SUPPORTED_MODELS:
        logger.info(f"Preparando dataset para o modelo: {model_name}")
        prepare_dataset(model_name=model_name)

    logger.info("Preparação do dataset concluída com sucesso")


if __name__ == "__main__":
    main()
