"""
ssd_trainer.py

Responsável pelo treinamento do modelo SSD
dentro da pipeline edge-vision-model.

Este módulo:
- consome exclusivamente datasets já preparados
- executa o treinamento do modelo SSD
- salva pesos, logs e metadados do treinamento
  em artifacts/models/ssd
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO

from evaluators.ssd_evaluator import evaluate_ssd

from config.settings import (
    DATASET_SPLITS,
    IMAGES_DIRNAME,
    ARTIFACTS_PREPARED_DATA_DIR,
    ARTIFACTS_MODELS_DIR,
)


logger = logging.getLogger(__name__)

# FUNÇÕES AUXILIARES

def _validate_prepared_dataset(prepared_dir: Path) -> None:
    logger.info("Validando dataset preparado para SSD")

    if not prepared_dir.exists():
        raise FileNotFoundError(f"Diretório preparado não encontrado: {prepared_dir}")
    
    for split in DATASET_SPLITS:
        split_dir = prepared_dir / split
        if not (split_dir / IMAGES_DIRNAME).exists():
            raise FileNotFoundError(f"Imagens não encontradas em: {split_dir}")
        if not (split_dir / "annotations.json").exists():
            raise FileNotFoundError(f"Annotations não encontradas em: {split_dir}")

    logger.info("Dataset preparado validado com sucesso")


def _ensure_output_dir(path: Path) -> None:
    """
    Garante que o diretório de saída do modelo exista.

    Regras:
    - Se não existir, cria.
    - Se existir, reutiliza.
    """
    if not path.exists():
        logger.info(f"Criando diretório de saída do modelo: {path}")
        path.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning(f"Diretório de saída do modelo já existe e será reutilizado: {path}")


def _save_training_metadata(
    output_dir: Path,
    model_config: Dict[str, Any]
) -> None:
    """
    Salva metadados do treinamento para rastreabilidade.

    Armazena:
    - parâmetros do modelo
    - hiperparâmetros de treino
    """ 
    metadata_path = output_dir / "training_metadata.txt"

    logger.info(f"Salvando metadados do treinamento em: {metadata_path}")

    with open(metadata_path, "w") as f:
        json.dump(model_config, f, indent=4)


# FUNÇÃO PRINCIPAL DE TREINAMENTO

def train_ssd(model_config: Dict[str, Any]) -> None:
    """
    Executa o treinamento do modelo SSD.

    Fluxo:
    1. Resolve paths do dataset preparado e saída do modelo
    2. Valida a existência do dataset preparado
    3. Cria diretório de saída do modelo
    4. Inicializa o backend SSD
    5. Executa o treinamento
    6. Salva pesos e metadados
    """

    logger.info("Iniciando treinamento do modelo SSD")

    # Diretórios do dataset preparado
    prepared_data_dir = ARTIFACTS_PREPARED_DATA_DIR / "ssd"

    # Directorios de saída do modelo
    output_model_dir = ARTIFACTS_MODELS_DIR / "ssd"

    # Validação do dataset preparado
    _validate_prepared_dataset(prepared_data_dir)

    # Garantia do diretório de saída do modelo
    _ensure_output_dir(output_model_dir)

    # =========================
    # EXTRAÇÃO DO YAML
    # =========================
    training_cfg = model_config["training"]
    early_cfg = training_cfg.get("early_stopping", {})

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    lr = training_cfg["learning_rate"]
    weight_decay = training_cfg["weight_decay"]

    early_enabled = early_cfg.get("enabled", False)
    patience = early_cfg.get("patience", 0)
    min_delta = early_cfg.get("min_delta", 0.0)
    monitor_metric = early_cfg.get("monitor", "AP_50_95_all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"SSD | epochs={epochs}, batch_size={batch_size}, "
        f"lr={lr}, device={device}"
    )

    # =========================
    # DATASETS
    # =========================
    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        dataset = CocoDetection(
            root=prepared_data_dir / split / IMAGES_DIRNAME,
            annFile=prepared_data_dir / split / "annotations.json",
            transform=ToTensor(),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    train_loader = _make_loader("train", shuffle=True)
    val_loader = _make_loader("valid", shuffle=False)

    # =========================
    # COCO GT (VALID)
    # =========================
    val_annotations_path = (
        prepared_data_dir / "valid" / "annotations.json"
    )
    coco_gt_valid = COCO(str(val_annotations_path))

    # =========================
    # MODELO SSD
    # =========================
    model = ssd300_vgg16(weights="DEFAULT")
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # =========================
    # LOOP DE TREINO
    # =========================
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            valid_images = []
            valid_targets = []

            for img, anns in zip(images, targets):
                # anns = lista de anotações COCO da imagem

                if len(anns) == 0:
                    continue

                boxes = []
                labels = []

                for ann in anns:
                    # COCO bbox = [x, y, w, h]
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])

                if len(boxes) == 0:
                    continue

                valid_images.append(img.to(device))
                valid_targets.append(
                    {
                        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                    }
                )

            # Se o batch inteiro for negativo, pula
            if len(valid_images) == 0:
                continue

            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        logger.info(f"[Epoch {epoch+1}] Train loss: {epoch_loss:.4f}")

        # =========================
        # VALIDAÇÃO (COCO REAL)
        # =========================
        metrics = evaluate_ssd(
            model=model,
            dataloader=val_loader,
            coco_gt=coco_gt_valid,
            device=device,
            model_name="ssd",
            split_name="valid",
            score_threshold=0.05,
            save_metrics=False,
        )

        current_metric = metrics.get(monitor_metric)

        if current_metric is None:
            raise KeyError(
                f"Métrica '{monitor_metric}' não encontrada "
                f"nas métricas retornadas: {list(metrics.keys())}"
            )

        logger.info(
            f"[Epoch {epoch+1}] Validation {monitor_metric}: "
            f"{current_metric:.4f}"
        )

        # =========================
        # EARLY STOPPING
        # =========================
        if early_enabled:
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                patience_counter = 0

                torch.save(
                    model.state_dict(),
                    output_model_dir / "best_model.pth",
                )

                logger.info(
                    f"Novo melhor modelo salvo "
                    f"({monitor_metric}={best_metric:.4f})"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"EarlyStopping: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                logger.info("Early stopping acionado")
                break

    # SALVAMENTO DE ARTEFATOS
    
    torch.save(model.state_dict(), output_model_dir / "final_model.pth")
    _save_training_metadata(
        output_dir=output_model_dir,
        model_config=model_config,
    )

    logger.info("Treinamento do modelo SSD finalizado com sucesso")