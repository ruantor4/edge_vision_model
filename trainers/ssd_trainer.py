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

from config.settings import (
    DATASET_SPLITS,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    ARTIFACTS_PREPARED_DATA_DIR,
    ARTIFACTS_MODELS_DIR,
)
logger = logging.getLogger(__name__)

# FUNÇÕES AUXILIARES

def _validate_prepared_dataset(prepared_dir: Path) -> None:
    """
    Validação mínima do dataset preparado para SSD.

    Verifica:
    - existência do diretório raiz
    - existência dos splits esperados
    - existência das pastas images/
    - existência do arquivo de anotações (ex: annotations.json)
    """
    logger.info("Validando dataset preparado para SSD")

    if not prepared_dir.exists():
        raise FileNotFoundError(
            f"Diretório preparado não encontrado: {prepared_dir}"
        )
    
    for split in DATASET_SPLITS:
        split_dir = prepared_dir / split
        images_dir = split_dir / IMAGES_DIRNAME
        annotations_path = split / "annotations.json"

        if not split_dir.exists():
            raise FileNotFoundError(f"Split não encontrado: {split_dir}")
        
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Pasta de imagens não encontrada: {images_dir}"
            )
        
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Arquivo de anotações não encontrado: {annotations_path}"
            )

    logger.info("Dataset preparado para SSD validado com sucesso")


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

    # 1. Validação do dataset preparado
    _validate_prepared_dataset(prepared_data_dir)

    # 2. Garantia do diretório de saída do modelo
    _ensure_output_dir(output_model_dir)

    # EXTRAÇÃO DE PARAMETROS DO YAML

    model_name = model_config.get("model_name")
    epochs = model_config.get("epochs")
    batch_size = model_config.get("batch_size")
    image_size = model_config.get("image_size")


    logger.info(
        f"Configuração SSD | model={model_name}, "
        f"epochs={epochs}, batch_size={batch_size}, image_size={image_size}"
    )

    """
    INICIALIZAÇÃO DO BACKEND SSD

    - iniciar backend SSD com os parâmetros extraídos
    - carregar pesos base
    - definir device (CPU/GPU)

    EXECUÇÃO DO TREINAMENTO

    - Executar loop de treinamento
    - Salvar pesos finais
    - Registrar logs de treino
    """

    # SALVAMENTO DE ARTEFATOS

    _save_training_metadata(
        output_dir=output_model_dir,
        model_config=model_config,
    )

    logger.info("Treinamento do modelo SSD finalizado com sucesso")