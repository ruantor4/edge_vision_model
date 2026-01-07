"""
base_coco_evaluator.py

Evaluator base para modelos de detecção de objetos utilizando
o protocolo de avaliação COCO (mAP).

Responsabilidades:
- Executar inferência em um dataloader
- Coletar predições no formato COCO
- Avaliar resultados usando pycocotools
- Retornar métricas padronizadas (mAP / AR)
- Persistir métricas em artifacts/metrics

Este evaluator é:
- Agnóstico ao modelo (YOLO, SSD, Faster R-CNN)
- Reutilizável por adaptadores específicos
- Usado tanto para validação (early stopping)
  quanto para avaliação final (test)
"""

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config.settings import ARTIFACTS_METRICS_DIR


# ============================================================
# TIPOS AUXILIARES
# ============================================================

# Estrutura esperada de uma predição individual no formato COCO
CocoPrediction = Dict[str, float]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _ensure_metrics_dir() -> None:
    """
    Garante que o diretório de métricas exista.
    """
    ARTIFACTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _build_coco_predictions(
    predictions: List[CocoPrediction]
) -> List[Dict]:
    """
    Constrói a lista de predições no formato esperado
    pelo pycocotools.

    Cada predição deve conter:
    - image_id
    - category_id
    - bbox [x, y, w, h]
    - score
    """
    coco_results = []

    for pred in predictions:
        coco_results.append(
            {
                "image_id": int(pred["image_id"]),
                "category_id": int(pred["category_id"]),
                "bbox": [
                    float(pred["x"]),
                    float(pred["y"]),
                    float(pred["w"]),
                    float(pred["h"]),
                ],
                "score": float(pred["score"]),
            }
        )

    return coco_results

# ============================================================
# FUNÇÃO PRINCIPAL DE AVALIAÇÃO
# ============================================================

def evaluate_coco(
    *,
    coco_gt: COCO,
    predictions: List[CocoPrediction],
    model_name: str,
    split_name: str,
    save_metrics: bool = True,
) -> Dict[str, float]:
    """
    Avalia predições de detecção de objetos utilizando
    o protocolo COCO.

    Parâmetros:
    - coco_gt: objeto COCO carregado com annotations.json
    - predictions: lista de predições já convertidas para COCO
    - model_name: nome do modelo avaliado (ex: faster_rcnn)
    - split_name: split avaliado (valid ou test)
    - save_metrics: se True, salva métricas em artifacts

    Retorna:
    - Dicionário com métricas COCO padronizadas
    """

    if len(predictions) == 0:
        raise ValueError("Nenhuma predição fornecida para avaliação COCO")

    coco_results = _build_coco_predictions(predictions)

    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP_50_95_all": float(coco_eval.stats[0]),
        "AP_50_all": float(coco_eval.stats[1]),
        "AP_75_all": float(coco_eval.stats[2]),
        "AR_50_95_all": float(coco_eval.stats[8]),
    }

    if save_metrics:
        _ensure_metrics_dir()
        _save_metrics_csv(
            metrics=metrics,
            model_name=model_name,
            split_name=split_name,
        )

    return metrics

# ============================================================
# PERSISTÊNCIA DE MÉTRICAS
# ============================================================

def _save_metrics_csv(
    *,
    metrics: Dict[str, float],
    model_name: str,
    split_name: str,
) -> None:
    """
    Salva métricas COCO em formato CSV para posterior
    comparação entre modelos.
    """
    output_path = (
        ARTIFACTS_METRICS_DIR / f"{model_name}_{split_name}.csv"
    )

    header = ",".join(metrics.keys())
    values = ",".join(str(v) for v in metrics.values())

    with open(output_path, "w") as f:
        f.write(header + "\n")
        f.write(values + "\n")
