"""
yolo_trainer.py

Responsável pelo treinamento do modelo YOLO
dentro da pipeline edge-vision-model.

Este módulo:
- consome exclusivamente datasets já preparados
- executa o treinamento do modelo YOLO (Ultralytics)
- delega controle de epochs e early stopping ao framework
- salva pesos, logs e metadados do treinamento
  em artifacts/models/yolo
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from ultralytics import YOLO

from config.settings import (
    DATASET_SPLITS,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    ARTIFACTS_PREPARED_DATA_DIR,
    ARTIFACTS_MODELS_DIR,
)

logger = logging.getLogger(__name__)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _validate_prepared_dataset(prepared_dir: Path) -> None:
    """
    Valida a estrutura mínima esperada do dataset preparado para YOLO.
    """
    logger.info("Validando dataset preparado para YOLO")

    if not prepared_dir.exists():
        raise FileNotFoundError(
            f"Diretório preparado não encontrado: {prepared_dir}"
        )

    for split in DATASET_SPLITS:
        split_dir = prepared_dir / split

        images_dir = split_dir / IMAGES_DIRNAME
        labels_dir = split_dir / LABELS_DIRNAME

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Imagens não encontradas em: {images_dir}"
            )

        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Labels YOLO não encontradas em: {labels_dir}"
            )

    data_yaml = prepared_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml não encontrado em: {data_yaml}"
        )

    logger.info("Dataset preparado para YOLO validado com sucesso")


def _ensure_output_dir(path: Path) -> None:
    """
    Garante que o diretório de saída do modelo exista.
    """
    if not path.exists():
        logger.info(f"Criando diretório de saída do modelo: {path}")
        path.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning(
            f"Diretório de saída do modelo já existe e será reutilizado: {path}"
        )


def _save_training_metadata(
    output_dir: Path,
    model_config: Dict[str, Any]
) -> None:
    """
    Salva metadados do treinamento para rastreabilidade.
    """
    metadata_path = output_dir / "training_metadata.txt"

    logger.info(f"Salvando metadados do treinamento em: {metadata_path}")

    with open(metadata_path, "w") as f:
        json.dump(model_config, f, indent=4)


# ============================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================================

def train_yolo(model_config: Dict[str, Any]) -> None:
    """
    Executa o treinamento do modelo YOLO utilizando
    controle AUTOMÁTICO do Ultralytics.

    - epochs
    - validação
    - early stopping
    - checkpoints

    Tudo é gerenciado internamente pelo framework.
    """
    logger.info("Iniciando treinamento do modelo YOLO")

    prepared_data_dir = ARTIFACTS_PREPARED_DATA_DIR / "yolo"
    output_model_dir = ARTIFACTS_MODELS_DIR / "yolo"

    _validate_prepared_dataset(prepared_data_dir)
    _ensure_output_dir(output_model_dir)

    training_cfg = model_config["training"]
    early_cfg = training_cfg.get("early_stopping", {})

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    lr = training_cfg["learning_rate"]

    patience = early_cfg.get("patience", 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"YOLO | epochs={epochs}, batch_size={batch_size}, "
        f"lr={lr}, device={device}, patience={patience}"
    )

    model = YOLO(model_config["model"]["version"])

    # ============================================================
    # TREINAMENTO AUTOMÁTICO (ULTRALYTICS)
    # ============================================================
    model.train(
        data=str(prepared_data_dir / "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        imgsz=training_cfg.get("image_size", 640),
        device=device,
        project=str(output_model_dir),
        name="train",
        exist_ok=True,
        patience=patience,
        verbose=False,
    )

    # ============================================================
    # SALVAMENTO FINAL
    # ============================================================
    trainer = model.trainer

    if trainer.best is not None:
        best_ckpt = Path(trainer.best)
        best_ckpt.rename(output_model_dir / "best_model.pt")

    if trainer.last is not None:
        last_ckpt = Path(trainer.last)
        last_ckpt.rename(output_model_dir / "final_model.pt")

    _save_training_metadata(
        output_dir=output_model_dir,
        model_config=model_config,
    )

    logger.info("Treinamento do modelo YOLO finalizado com sucesso")
