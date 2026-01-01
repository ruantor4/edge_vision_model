"""
main.py

Main mínimo para validar settings e logging do edge-vision-model.
"""

import logging

from config.settings import (
    ROOT_DIR,
    DATASET_DIR,
    LOGS_DIR,
)

from utils.logging_global import setup_logging


def main() -> None:

    # ETAPA 0 – PREPARAÇÃO DO AMBIENTE

    LOGS_DIR.mkdir(parents=True, exist_ok=True)


    # ETAPA 1 – LOGGING

    setup_logging()
    logger = logging.getLogger(__name__)


    # TESTES BÁSICOS

    logger.info("Teste de inicialização do edge-vision-model")
    logger.info("ROOT_DIR: %s", ROOT_DIR)
    logger.info("DATASET_DIR: %s", DATASET_DIR)
    logger.info("DATASET_DIR existe? %s", DATASET_DIR.exists())
    logger.info("LOGS_DIR: %s", LOGS_DIR)

    logger.info("Teste de settings e logging finalizado com sucesso")


if __name__ == "__main__":
    main()
