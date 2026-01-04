"""
dataset_preparer.py

Responsável por preparar datasets de detecção de objetos
para diferentes algoritmos de modelagem.

Este módulo:
- lê o dataset RAW (somente leitura)
- converte e organiza dados por algoritmo
- gera datasets preparados em artifacts
"""
import json
from pathlib import Path
import logging
import shutil
from PIL import Image

from config.settings import(
    DATASET_DIR,
    DATASET_SPLITS,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    ALLOW_OVERWRITE_ARTIFACTS,
    ARTIFACTS_PREPARED_DATA_DIR
)

logger = logging.getLogger(__name__)

# FUNÇOES AUXILIARES

def _validate_raw_dataset() -> None:
    """
    Validação mínima do dataset RAW antes da preparação.

    Verifica:
    - existência do diretório raiz do dataset
    - existência dos splits esperados
    - existência das pastas images/ e labels/ em cada split
    """
    logger.info("Validando estrutura do dataset RAW")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {DATASET_DIR}")
    
    for split in DATASET_SPLITS:
        split_dir = DATASET_DIR /split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split ausente no dataset: {split_dir}")
        
        images_dir = split_dir / IMAGES_DIRNAME
        labels_dir = split_dir / LABELS_DIRNAME

        if not images_dir.exists():
            raise FileNotFoundError(f"Pasta de imagens ausente: {images_dir}")

        if not labels_dir.exists():
            raise FileNotFoundError(f"Pasta de labels ausente: {labels_dir}")

    logger.info("Estrutura do dataset RAW validado com sucesso")

def _ensure_output_dirs(path: Path) -> None:
    """
    Garante que os diretórios de saída existam.

    Regras:
    - Se o diretório não existir, cria.
    - Se existir e ALLOW_OVERWRITE_ARTIFACTS=False, falha.
    - Se existir e ALLOW_OVERWRITE_ARTIFACTS=True, reutiliza.
    """
    
    # diretório não existe → cria
    if not path.exists():
        logger.info(f"Criando diretório de saída: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return
    
    # diretório existe e sobrescrita NÃO permitida → falha
    if not ALLOW_OVERWRITE_ARTIFACTS:
        raise FileExistsError(f"Diretório de saída já existe: {path}")

    # diretório existe e sobrescrita permitida → reutiliza
    logger.warning(f"Diretório de saída já existe e será reutilizado: {path}")
    
   
def _copy_split_structure(destination_root: Path) -> None:
    """
    Replica a estrutura de splits do dataset no diretório de saída.

    Cria, para cada split:
    - pasta do split (train/valid/test)
    - subpastas images/ e labels/

    Não copia arquivos.
    """
    logger.info(f"Criando estrutura de splits no diretório de saída: {destination_root}")
    
    # Percorre cada split esperado definido em settings.py
    for split in DATASET_SPLITS:
        split_dir = destination_root / split
        images_dir = split_dir / IMAGES_DIRNAME
        labels_dir = split_dir / LABELS_DIRNAME

        # Cria a pasta images/ do split
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Cria a pasta labels/ do split
        labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Estrutura de splits criada com sucesso")


def _create_output_dirs(path: Path) -> None:
    """
    Cria diretórios de saída de forma segura.

    Esta função é um wrapper simples que centraliza a criação
    de diretórios de saída, delegando a política de sobrescrita
    para _ensure_output_dirs().
    """
    _ensure_output_dirs(path)


def _convert_yolo_split_to_coco(
    raw_split_dir: Path,
    prepared_split_dir: Path,
) -> None:
    """
    Converte um split do dataset do formato YOLO para COCO.

    - Lê imagens e labels YOLO
    - Converte bounding boxes normalizadas para pixels
    - Gera annotations.json no formato COCO
    """

    images_dir = raw_split_dir / IMAGES_DIRNAME
    labels_dir = raw_split_dir / LABELS_DIRNAME

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "object",
            }
        ],
    }

    image_id = 1
    annotation_id = 1

    for image_path in images_dir.iterdir():
        if not image_path.is_file():
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, x_c, y_c, w, h = map(float, parts)

                    bbox_width = w * width
                    bbox_height = h * height
                    x_min = (x_c * width) - (bbox_width / 2)
                    y_min = (y_c * height) - (bbox_height / 2)

                    coco["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [
                                round(x_min, 2),
                                round(y_min, 2),
                                round(bbox_width, 2),
                                round(bbox_height, 2),
                            ],
                            "area": round(bbox_width * bbox_height, 2),
                            "iscrowd": 0,
                        }
                    )

                    annotation_id += 1

        image_id += 1

    annotations_path = prepared_split_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(f"Annotations COCO geradas em: {annotations_path}")


def _generate_yolo_data_yaml(prepared_yolo_dir: Path, class_names: list[str]) -> None:
    """
    Gera o arquivo data.yaml exigido pelo Ultralytics YOLO.
    """
    data_yaml_path = prepared_yolo_dir / "data.yaml"

    content = {
        "path": str(prepared_yolo_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    import yaml
    with open(data_yaml_path, "w") as f:
        yaml.dump(content, f, sort_keys=False)

# FUNÇOES PRINCIPAIS

def prepare_dataset(model_name: str) -> None:
    """
    Prepara o dataset conforme o algoritmo informado.

    Esta função atua como ponto de entrada público do módulo,
    validando o nome do modelo e delegando a preparação para
    a implementação específica.
    """
    logger.info(f"Iniciando preparação do dataset para o modelo: {model_name}")

    # Normaliza o nome do modelo para evitar inconsistências
    model = model_name.lower().strip()

    # Validação explicita o modelo solicitado
    if model not in ("yolo", "ssd", "faster_rcnn"):
        raise ValueError(f"Modelo desconhecido para preparação de dataset: {model_name}")
    
    # Delegação para a implementação específica
    if model == "yolo":
        prepare_yolo_dataset()
    elif model == "ssd":
        prepare_ssd_dataset()
    elif model == "faster_rcnn":
        prepare_faster_rcnn_dataset()
    
    logger.info(f"Preparação do dataset para o modelo {model_name} concluída com sucesso")


def prepare_yolo_dataset() -> None:
    """
    Prepara o dataset no formato YOLO.

    Fluxo:
    1. Valida a estrutura do dataset RAW
    2. Cria o diretório de saída do dataset preparado
    3. Replica a estrutura de splits (train/valid/test)
    4. Copia imagens e labels para o diretório preparado
    """
    logger.info("Preparando dataset no formato YOLO")

    # Passo 1: Validação do dataset RAW
    _validate_raw_dataset()

    # 2. Diretório raiz do dataset preparado para YOLO
    output_dir = ARTIFACTS_PREPARED_DATA_DIR / "yolo"

    # Criação segura do diretório raiz do artifact
    _create_output_dirs(output_dir)

    # Replica a estrutura de splits (train/valid/test)
    _copy_split_structure(output_dir)

    # Cópia dos arquivos (imagens e labels) do RAW para o dataset preparado
    for split in DATASET_SPLITS:
        raw_split_dir = DATASET_DIR / split
        prepared_split_dir = output_dir / split

        raw_images_dir = raw_split_dir / IMAGES_DIRNAME
        raw_labels_dir = raw_split_dir / LABELS_DIRNAME

        prepared_images_dir = prepared_split_dir / IMAGES_DIRNAME
        prepared_labels_dir = prepared_split_dir / LABELS_DIRNAME

        logger.info(f"Copiando arquivos do split: {split}")

        # Copia imagens
        for image_path in raw_images_dir.iterdir():
            if image_path.is_file():
                shutil.copy(image_path, prepared_images_dir / image_path.name)
        
        # Copia labels
        for label_path in raw_labels_dir.iterdir():
            if label_path.is_file():
                shutil.copy(label_path, prepared_labels_dir / label_path.name)
        
        _convert_yolo_split_to_coco(
            raw_split_dir=raw_split_dir,
            prepared_split_dir=prepared_split_dir
        )
        
    # Gera Yaml
    _generate_yolo_data_yaml(
        prepared_yolo_dir=output_dir,
        class_names=["mouse"],
    )

    logger.info("Dataset no formato YOLO preparado com sucesso")


def prepare_ssd_dataset() -> None:
    """
    Prepara o dataset para treinamento com SSD.

    Fluxo:
    1. Valida a estrutura do dataset RAW
    2. Cria o diretório de saída do dataset preparado
    3. Replica a estrutura de splits
    4. Converte o dataset para o formato exigido pelo SSD (ex: COCO/VOC)
    """
    logger.info("Iniciando preparação do dataset no formato SSD")

    # Passo 1: Validação do dataset RAW
    _validate_raw_dataset()

    # 2. Diretório raiz do dataset preparado para SSD
    output_dir = ARTIFACTS_PREPARED_DATA_DIR / "ssd"

    # Criação segura do diretório raiz do artifact
    _create_output_dirs(output_dir)

    # 3. Replica a estrutura de splits (train/valid/test)
    _copy_split_structure(output_dir)

    # 4. Conversão específica para o formato SSD
    for split in DATASET_SPLITS:
        logger.info((f"Preparando split {split} para o formato SSD"))

        raw_split_dir = DATASET_DIR / split
        prepare_split_dir = output_dir / split
        prepared_images_dir = prepare_split_dir / IMAGES_DIRNAME

        # Copia imagens
        for image_path in (raw_split_dir / IMAGES_DIRNAME).iterdir():
            if image_path.is_file():
                shutil.copy(image_path, prepared_images_dir / image_path.name)

        _convert_yolo_split_to_coco( 
            raw_split_dir=raw_split_dir,
            prepared_split_dir=prepare_split_dir,
        )

    logger.info("Dataset no formato SSD preparado com sucesso")


def prepare_faster_rcnn_dataset() -> None:
    """
    Prepara o dataset para treinamento com Faster R-CNN.

    Fluxo:
    1. Valida a estrutura do dataset RAW
    2. Cria o diretório de saída do dataset preparado
    3. Replica a estrutura de splits
    4. Converte o dataset para o formato COCO
    """
    logger.info("Iniciando preparação do dataset no formato Faster R-CNN")

    # Passo 1: Validação do dataset RAW
    _validate_raw_dataset()

    # 2. Diretório raiz do dataset preparado para Faster R-CNN
    output_dir = ARTIFACTS_PREPARED_DATA_DIR / "faster_rcnn"

    # Criação segura do diretório raiz do artifact
    _create_output_dirs(output_dir)

    # 3. Replica a estrutura de splits (train/valid/test)
    _copy_split_structure(output_dir)

    # 4. Conversão específica para o formato Faster R-CNN
    for split in DATASET_SPLITS:
        logger.info(f"Convertendo split {split} para o formato Faster R-CNN")

        raw_split_dir = DATASET_DIR / split 
        prepare_split_dir = output_dir / split
        prepared_images_dir = prepare_split_dir / IMAGES_DIRNAME

        # Copia imagens
        for image_path in (raw_split_dir / IMAGES_DIRNAME).iterdir():
            if image_path.is_file():
                shutil.copy(image_path, prepared_images_dir / image_path.name)
        
        _convert_yolo_split_to_coco(  
            raw_split_dir=raw_split_dir,
            prepared_split_dir=prepare_split_dir,
        )

    logger.info(
        f"Dataset no formato Faster R-CNN preparado com sucesso em: {output_dir}"
    )
