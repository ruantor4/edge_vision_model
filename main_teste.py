"""
main.py

Entrada mínima para validação de configurações do projeto.
"""

import logging
from pathlib import Path

from utils.logging_global import setup_logging
from trainers.yolo_trainer import train_yolo
from trainers.ssd_trainer import train_ssd
from core.dataset_preparer import prepare_dataset
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

    # ============================
    # MODEL - SSD
    # ============================
    ssd_config = load_yaml(Path("config/models/ssd.yaml"))
    validate_model_config(ssd_config)
    logger.info("ssd.yaml validado com sucesso")

    
    # ============================
    # MODEL - FASTER R-CNN
    # ============================
    faster_rcnn_config = load_yaml(Path("config/models/faster_rcnn.yaml"))
    validate_model_config(faster_rcnn_config)
    logger.info("faster_rcnn.yaml validado com sucesso")
    
    logger.info("Validações de configurações concluída com sucesso")

    # ============================
    # PREPARE DATASETS
    # ============================
    logger.info("Iniciando teste de preparação de datasets")

    # Teste YOLO
    logger.info("=== TESTE: YOLO ===")
    prepare_dataset("yolo")

    # Teste SSD
    logger.info("=== TESTE: SSD ===")
    prepare_dataset("ssd")

    # Teste Faster R-CNN
    logger.info("=== TESTE: Faster R-CNN ===")
    prepare_dataset("faster_rcnn")

    logger.info("Teste de preparação de datasets concluído com sucesso")

    # ============================
    # TESTE ISOLADO DE TREINAMENTO
    # ============================
    
    # YOLO TRAINER
    logger.info("Iniciando teste isolado do yolo_trainer")
    
    logger.info("Validando contrato do yolo_trainer")

    _ = train_yolo  # apenas valida que a função existe e é importável

    logger.info("yolo_trainer validado com sucesso")

    logger.info("Main de teste finalizado com sucesso")
    
    # SSD TRAINER
    logger.info("Validando contrato do ssd_trainer")
    
    _ = train_ssd
    
    logger.info("ssd_trainer validado com sucesso (sem execução)")

    logger.info("Main de teste finalizado com sucesso")


if __name__ == "__main__":
    main()