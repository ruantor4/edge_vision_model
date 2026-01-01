"""
logging_global.py

Responsável por configurar o sistema de logging do projeto.

Este módulo:
- cria um logger global
- define formato de log
- define arquivo de saída
"""

from datetime import datetime
import logging

from config.settings import LOGS_DIR

def setup_logging()-> None:
    """
    Configura o logger global com formato e arquivo de saída.
    """
    # Evita configuração duplicada de handlers
    if logging.getLogger().handlers:
        return

    # Data atual para nome do arquivo de log
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"edge_vision_model_{date_str}.log"

    # Formato do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level = logging.INFO,
        format = log_format,
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )