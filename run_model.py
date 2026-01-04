"""
run_model.py

Orquestrador de execução da pipeline de modelagem do projeto edge-vision-model.
"""

import argparse
import logging
from pathlib import Path

from utils.logging_global import setup_logging

from core.config_validator import (
    load_yaml,
    validate_dataset_config,
    validate_metrics_config,
    validate_model_config,
)
from trainers.yolo_trainer import train_yolo
from trainers.ssd_trainer import train_ssd
from trainers.faster_rcnn_trainer import train_faster_rcnn

from config.settings import (
    YOLO_CONFIG_PATH,
    SSD_CONFIG_PATH,
    FASTER_RCNN_CONFIG_PATH,
    DATASET_CONFIG_PATH,
    METRICS_CONFIG_PATH,
)


def run_train(model_name: str) -> None:
    logger = logging.getLogger(__name__)

    model = model_name.lower().strip()
    logger.info(f"Iniciando treinamento do modelo: {model}")

    if model == "yolo":
        model_config = load_yaml(Path(YOLO_CONFIG_PATH))
        validate_model_config(model_config)
        train_yolo(model_config)

    elif model == "ssd":
        model_config = load_yaml (Path(SSD_CONFIG_PATH))
        validate_model_config(model_config)
        train_ssd(model_config)

    elif model == "faster_rcnn":
        model_config = load_yaml(Path(FASTER_RCNN_CONFIG_PATH))
        validate_model_config(model_config)
        train_faster_rcnn(model_config)

    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    logger.info(f"Treinamento finalizado para o modelo: {model}")


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Execução da pipeline de modelagem (edge-vision-model)"
    )

    parser.add_argument(
        "action",
        choices=["train"],
        help="Ação a ser executada"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["yolo", "ssd", "faster_rcnn"],
        help="Modelo a ser utilizado"
    )

    args = parser.parse_args()

    logger.info("Iniciando execução da pipeline")


    # VALIDAÇÕES GLOBAIS
    dataset_config = load_yaml(Path(DATASET_CONFIG_PATH))
    validate_dataset_config(dataset_config)

    metrics_config = load_yaml(Path(METRICS_CONFIG_PATH))
    validate_metrics_config(metrics_config)

    logger.info("Configurações globais validadas com sucesso")

    # ============================
    # EXECUÇÃO
    # ============================
    if args.action == "train":
        run_train(model_name=args.model)


if __name__ == "__main__":
    main()
