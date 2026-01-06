"""
faster_rcnn_evaluator.py

Evaluator específico para modelos Faster R-CNN (torchvision),
responsável por adaptar as predições do modelo para o formato
COCO e delegar a avaliação ao base_coco_evaluator.

Responsabilidades:
- Executar inferência com Faster R-CNN
- Converter outputs do torchvision para predições COCO
- Chamar o evaluator base (COCO)
- Retornar métricas padronizadas (mAP / AR)

Este módulo:
- NÃO treina modelos
- NÃO decide early stopping
- NÃO compara resultados
- Atua apenas como adaptador de saída
"""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from pycocotools.coco import COCO

from evaluators.base_coco_evaluator import evaluate_coco


# ============================================================
# FUNÇÃO PRINCIPAL DE AVALIAÇÃO
# ============================================================

def evaluate_faster_rcnn(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    coco_gt: COCO,
    device: torch.device,
    model_name: str,
    split_name: str,
    score_threshold: float = 0.05,
    save_metrics: bool = True,
) -> Dict[str, float]:
    """
    Avalia um modelo Faster R-CNN utilizando o protocolo COCO.

    Parâmetros:
    - model: modelo Faster R-CNN já carregado
    - dataloader: DataLoader do split avaliado (valid ou test)
    - coco_gt: objeto COCO carregado com annotations.json
    - device: dispositivo de execução (cpu ou cuda)
    - model_name: nome do modelo (ex: faster_rcnn)
    - split_name: nome do split avaliado (valid ou test)
    - score_threshold: score mínimo para considerar predição
    - save_metrics: se True, salva métricas em artifacts

    Retorna:
    - Dicionário com métricas COCO padronizadas
    """

    model.eval()

    predictions: List[Dict] = []

    with torch.no_grad():
        for images, targets in dataloader:

            # targets: lista de listas de anotações COCO
            images = [img.to(device) for img in images]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                
                # Se a imagem não possui anotações GT, ignora
                if len(target) == 0:
                    continue

                # Todas as anotações da lista pertencem à mesma imagem
                image_id = int(target[0]["image_id"])

                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    if score.item() < score_threshold:
                        continue

                    x_min, y_min, x_max, y_max = box.tolist()

                    predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "x": float(x_min),
                        "y": float(y_min),
                        "w": float(x_max - x_min),
                        "h": float(y_max - y_min),
                        "score": float(score.item()),
                    }
                )

    if len(predictions) == 0:
        raise ValueError(
            "Nenhuma predição válida gerada pelo Faster R-CNN "
            "para avaliação COCO"
        )

    # Delegação da avaliação ao evaluator base COCO
    metrics = evaluate_coco(
        coco_gt=coco_gt,
        predictions=predictions,
        model_name=model_name,
        split_name=split_name,
        save_metrics=save_metrics,
    )

    return metrics