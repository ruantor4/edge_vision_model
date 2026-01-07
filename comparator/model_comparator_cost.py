"""
model_comparator_cost.py

Responsável por medir e comparar o custo computacional de inferência
entre diferentes modelos de detecção de objetos.

Escopo desta versão:
- Inferência controlada em subconjunto fixo do dataset de teste
- Medição de tempo médio por imagem
- Cálculo de FPS
- Medição de pico de uso de VRAM (quando GPU disponível)
- Exportação de resultados em CSV e JSON

Observações importantes:
- NÃO realiza treinamento
- NÃO recalcula métricas de avaliação (AP/AR)
- Utiliza pesos já treinados (best_model.*)
- Mede apenas inferência (padrão profissional)
"""

from pathlib import Path
import csv
import json
import time
from typing import Dict, List

import torch
from torchvision import transforms
from PIL import Image

from config.settings import (
    ARTIFACTS_MODELS_DIR,
    ARTIFACTS_COMPARISONS_DIR,
    DATASET_DIR,
    IMAGES_DIRNAME,
)


# ==============================
# CONFIGURAÇÃO
# ==============================

# Número de imagens usadas para benchmark de inferência
NUM_BENCHMARK_IMAGES = 50

# Número de execuções repetidas para média (estabilidade)
NUM_REPEATS = 3


# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def get_device() -> torch.device:
    """
    Retorna o dispositivo disponível para inferência.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_images(limit: int) -> List[Path]:
    """
    Carrega paths de um subconjunto fixo de imagens do split test.
    """
    test_images_dir = DATASET_DIR / "test" / IMAGES_DIRNAME

    if not test_images_dir.exists():
        raise FileNotFoundError(
            f"Diretório de imagens de teste não encontrado: {test_images_dir}"
        )

    images = sorted(test_images_dir.glob("*"))

    if len(images) < limit:
        raise ValueError(
            f"Número insuficiente de imagens para benchmark "
            f"(encontrado={len(images)}, requerido={limit})"
        )

    return images[:limit]


def measure_inference_time(
    model,
    images: List[Path],
    device: torch.device,
    preprocess
) -> Dict[str, float]:
    """
    Executa inferência controlada e mede tempo e uso de memória.
    """
    model.eval()
    model.to(device)

    times: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(NUM_REPEATS):
            for img_path in images:
                image = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)

                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()

                times.append(end - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    vram_peak_mb = None
    if device.type == "cuda":
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "mean_inference_time_ms": avg_time * 1000,
        "fps": fps,
        "vram_peak_mb": vram_peak_mb,
    }


# ==============================
# MODELOS (STUBS CONTROLADOS)
# ==============================

def load_yolo_model(weights_path: Path, device: torch.device):
    """
    Carrega modelo YOLO para inferência.
    """
    from ultralytics import YOLO
    return YOLO(weights_path)


def load_torchvision_model(model_name: str, weights_path: Path, device: torch.device):
    """
    Carrega modelos Faster R-CNN ou SSD (torchvision) para inferência de custo computacional.

    Observação:
    - O head de classificação é removido do state_dict quando incompatível,
      pois não é relevante para benchmark de inferência.
    """
    if model_name == "faster_rcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(num_classes=2)
        head_prefixes = [
            "roi_heads.box_predictor.cls_score",
            "roi_heads.box_predictor.bbox_pred",
        ]

    elif model_name == "ssd":
        from torchvision.models.detection import ssd300_vgg16
        model = ssd300_vgg16(num_classes=2)
        head_prefixes = [
            "head.classification_head",
            "head.regression_head",
        ]

    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    state_dict = torch.load(weights_path, map_location=device)

    # ==============================
    # REMOÇÃO EXPLÍCITA DO HEAD
    # ==============================
    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in head_prefixes)
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    return model


# ==============================
# COMPARATOR PRINCIPAL
# ==============================

def compare_models_cost() -> Dict[str, Dict]:
    """
    Executa benchmark de custo computacional para todos os modelos.
    """
    device = get_device()
    images = load_test_images(NUM_BENCHMARK_IMAGES)

    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    results: Dict[str, Dict] = {}

    # YOLO
    yolo_weights = ARTIFACTS_MODELS_DIR / "yolo" / "best_model.pt"
    yolo_model = load_yolo_model(yolo_weights, device)

    results["YOLO"] = measure_inference_time(
        model=yolo_model.model,
        images=images,
        device=device,
        preprocess=preprocess,
    )

    # Faster R-CNN
    frcnn_weights = ARTIFACTS_MODELS_DIR / "faster_rcnn" / "best_model.pth"
    frcnn_model = load_torchvision_model(
        model_name="faster_rcnn",
        weights_path=frcnn_weights,
        device=device,
    )

    results["FASTER RCNN"] = measure_inference_time(
        model=frcnn_model,
        images=images,
        device=device,
        preprocess=preprocess,
    )

    # SSD
    ssd_weights = ARTIFACTS_MODELS_DIR / "ssd" / "best_model.pth"
    ssd_model = load_torchvision_model(
        model_name="ssd",
        weights_path=ssd_weights,
        device=device,
    )

    results["SSD"] = measure_inference_time(
        model=ssd_model,
        images=images,
        device=device,
        preprocess=preprocess,
    )

    return results


# ==============================
# EXPORTAÇÃO
# ==============================

def export_results(results: Dict[str, Dict]) -> None:
    """
    Exporta resultados de custo computacional em CSV e JSON.
    """
    ARTIFACTS_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = ARTIFACTS_COMPARISONS_DIR / "models_cost_comparison.csv"
    json_path = ARTIFACTS_COMPARISONS_DIR / "models_cost_comparison.json"

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "mean_inference_time_ms", "fps", "vram_peak_mb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model, data in results.items():
            row = {"model": model}
            row.update(data)
            writer.writerow(row)

    # JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

