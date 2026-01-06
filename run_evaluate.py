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

# ============================================================
# ENTRYPOINT DA ETAPA DE AVALIAÇÃO FINAL
# ============================================================

def main() -> None:
    """
    Ponto de entrada da etapa de avaliação final da pipeline.

    Responsável por:
    - Inicializar o sistema de logging
    - Acionar a execução da avaliação via evaluation_runner
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando execução da avaliação final")
    run_evaluation()
    logger.info("Avaliação final concluída com sucesso")


if __name__ == "__main__":
    main()
