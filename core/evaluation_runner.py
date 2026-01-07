"""
evaluation_runner.py

Responsável pela execução da etapa de AVALIAÇÃO da pipeline
edge-vision-model.

Este módulo:
- lê o contrato formal de avaliação em config/metrics.yaml
- executa avaliação COCO no split definido (ex: test)
- carrega modelos já treinados (YOLO, SSD, Faster R-CNN)
- delega inferência aos evaluators específicos
- gera métricas oficiais em CSV (artifacts/metrics)
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import yaml
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)

from evaluators.yolo_evaluator import evaluate_yolo
from evaluators.ssd_evaluator import evaluate_ssd
from evaluators.faster_rcnn_evaluator import evaluate_faster_rcnn

from config.settings import (
    ROOT_DIR,
    IMAGES_DIRNAME,
    ARTIFACTS_PREPARED_DATA_DIR,
    ARTIFACTS_MODELS_DIR,
    ARTIFACTS_METRICS_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONTRATO DE AVALIAÇÃO
# ============================================================

def _load_metrics_config() -> Dict:
    """
    Carrega o contrato formal de avaliação (metrics.yaml).

    Responsável apenas por:
    - localizar o arquivo de configuração
    - carregar o YAML
    - retornar o bloco 'evaluation'
    """
    path = ROOT_DIR / "config" / "metrics.yaml"

    if not path.exists():
        raise FileNotFoundError(f"metrics.yaml não encontrado: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg["evaluation"]


# ============================================================
# DATALOADER COCO
# ============================================================

def _build_dataloader(
    prepared_dir: Path,
    split: str,
) -> DataLoader:
    """
    Constrói o DataLoader no formato COCO a partir
    do dataset previamente preparado.

    Utiliza:
    - imagens do split informado
    - arquivo annotations.json no padrão COCO
    """
    dataset = CocoDetection(
        root=prepared_dir / split / IMAGES_DIRNAME,
        annFile=prepared_dir / split / "annotations.json",
        transform=ToTensor(),
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

# ============================================================
# EXECUÇÃO POR MODELO
# ============================================================

def _run_yolo(split: str, conf: float, device: torch.device) -> None:
    """
    Executa a avaliação do modelo YOLO.
    """
    logger.info("Avaliando YOLO")

    prepared = ARTIFACTS_PREPARED_DATA_DIR / "yolo"
    model_dir = ARTIFACTS_MODELS_DIR / "yolo"
    weights = model_dir / "best_model.pt"

    model = YOLO(str(weights))
    model.to(device)

    dataloader = _build_dataloader(prepared, split)
    coco_gt = COCO(str(prepared / split / "annotations.json"))

    evaluate_yolo(
        model=model,
        dataloader=dataloader,
        coco_gt=coco_gt,
        device=device,
        model_name="yolo",
        split_name=split,
        score_threshold=conf,
        save_metrics=True,
    )


def _run_ssd(split: str, conf: float, device: torch.device) -> None:
    """
    Executa a avaliação do modelo SSD.
    """
    logger.info("Avaliando SSD")

    prepared = ARTIFACTS_PREPARED_DATA_DIR / "ssd"
    model_dir = ARTIFACTS_MODELS_DIR / "ssd"
    weights = model_dir / "best_model.pth"

    model = ssd300_vgg16(weights=None)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    dataloader = _build_dataloader(prepared, split)
    coco_gt = COCO(str(prepared / split / "annotations.json"))

    evaluate_ssd(
        model=model,
        dataloader=dataloader,
        coco_gt=coco_gt,
        device=device,
        model_name="ssd",
        split_name=split,
        score_threshold=conf,
        save_metrics=True,
    )


def _run_faster_rcnn(split: str, conf: float, device: torch.device) -> None:
    """
    Executa a avaliação do modelo Faster R-CNN.
    """
    logger.info("Avaliando Faster R-CNN")

    prepared = ARTIFACTS_PREPARED_DATA_DIR / "faster_rcnn"
    model_dir = ARTIFACTS_MODELS_DIR / "faster_rcnn"
    weights = model_dir / "best_model.pth"

    model = fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    dataloader = _build_dataloader(prepared, split)
    coco_gt = COCO(str(prepared / split / "annotations.json"))

    evaluate_faster_rcnn(
        model=model,
        dataloader=dataloader,
        coco_gt=coco_gt,
        device=device,
        model_name="faster_rcnn",
        split_name=split,
        score_threshold=conf,
        save_metrics=True,
    )

# ============================================================
# ENTRYPOINT DA AVALIAÇÃO
# ============================================================

def run_evaluation() -> None:
    """
    Orquestra a execução completa da etapa de avaliação.

    Fluxo:
    1. Cria diretório de métricas (se necessário)
    2. Carrega o contrato formal de avaliação
    3. Define split, threshold e dispositivo
    4. Executa avaliação por modelo
    """
    logger.info("=== INICIANDO AVALIAÇÃO FINAL ===")

    ARTIFACTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_metrics_config()

    split = cfg["evaluation_split"]
    conf = cfg["thresholds"]["confidence_threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _run_yolo(split, conf, device)
    _run_ssd(split, conf, device)
    _run_faster_rcnn(split, conf, device)

    logger.info("=== AVALIAÇÃO FINAL CONCLUÍDA ===")
