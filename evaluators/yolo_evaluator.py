"""
yolo_evaluator.py

Evaluator específico para modelos YOLO,
responsável por adaptar as predições do YOLO para o formato
COCO e delegar a avaliação ao base_coco_evaluator.

Responsabilidades:
- Executar inferência com YOLO
- Converter predições YOLO (cx, cy, w, h) para COCO (x, y, w, h)
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

def evaluate_yolo(
    *,
    model,
    dataloader: DataLoader,
    coco_gt: COCO,
    device: torch.device,
    model_name: str,
    split_name: str,
    score_threshold: float = 0.05,
    save_metrics: bool = True,
) -> Dict[str, float]:
    """
    Avalia um modelo YOLO utilizando o protocolo COCO.

    Parâmetros:
    - model: modelo YOLO já carregado
    - dataloader: DataLoader do split avaliado (valid ou test)
    - coco_gt: objeto COCO carregado com annotations.json
    - device: dispositivo de execução (cpu ou cuda)
    - model_name: nome do modelo (ex: yolo)
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
            # ====================================================
            # 🔧 CONVERSÃO OBRIGATÓRIA
            # torch.Tensor (CHW, float) -> np.ndarray (HWC, uint8)
            # ====================================================
            np_images = []

            for img in images:
                img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                img_np = (img_np * 255).astype("uint8")      # float -> uint8
                np_images.append(img_np)

            # ====================================================
            # INFERÊNCIA YOLO (API CORRETA)
            # ====================================================
            results = model.predict(
                source=np_images,
                imgsz=640,
                conf=score_threshold,
                device=device,
                verbose=False,
            )

            # Itera por imagem do batch
            for i, (result, target) in enumerate(zip(results, targets)):
                if len(target) == 0:
                    continue

                image_id = int(target[0]["image_id"])

                # YOLO (Ultralytics) costuma expor .boxes
                boxes = result.boxes

                if boxes is None:
                    continue

                for box in boxes:
                    score = float(box.conf.item())
                    if score < score_threshold:
                        continue

                    cls_id = int(box.cls.item()) + 1
                    x_center, y_center, w, h = box.xywh[0].tolist()

                    # Converte YOLO (cx, cy, w, h) → COCO (x, y, w, h)
                    x_min = x_center - (w / 2)
                    y_min = y_center - (h / 2)

                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": cls_id,
                            "x": float(x_min),
                            "y": float(y_min),
                            "w": float(w),
                            "h": float(h),
                            "score": score,
                        }
                    )

    if len(predictions) == 0:
        raise ValueError(
            "Nenhuma predição válida gerada pelo YOLO "
            "para avaliação COCO"
        )

    metrics = evaluate_coco(
        coco_gt=coco_gt,
        predictions=predictions,
        model_name=model_name,
        split_name=split_name,
        save_metrics=save_metrics,
    )

    return metrics