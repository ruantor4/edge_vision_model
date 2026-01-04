"""
run_evaluate.py

Interface de linha de comando para executar a
ETAPA DE AVALIAÇÃO FINAL da pipeline edge-vision-model.

Este script:
- configura logging
- valida o ambiente
- delega execução ao evaluation_runner
"""

import logging

from utils.logging_global import setup_logging
from core.evaluation_runner import run_evaluation


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando execução da avaliação final")
    run_evaluation()
    logger.info("Avaliação final concluída com sucesso")


if __name__ == "__main__":
    main()
