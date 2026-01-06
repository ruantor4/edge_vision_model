"""
model_comparator_metrics.py

Responsável por consolidar e comparar métricas de avaliação
entre diferentes modelos de detecção de objetos.

Este módulo atua exclusivamente sobre métricas de avaliação
no padrão COCO, previamente geradas na etapa de avaliação.

Escopo:
- Leitura de arquivos CSV de métricas (COCO-style)
- Consolidação das métricas por modelo
- Geração de rankings por métrica
- Exportação de resultados consolidados em CSV e JSON
"""

from pathlib import Path
import csv
import json
from typing import Dict, List

from config.settings import (
    ARTIFACTS_METRICS_DIR,
    ARTIFACTS_COMPARISONS_DIR,
)


# ==============================
# CONFIGURAÇÃO
# ==============================

# Métricas COCO consideradas na comparação
METRICS_FIELDS = (
    "AP_50_95_all",
    "AP_50_all",
    "AP_75_all",
    "AR_50_95_all",
)

# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def load_metrics_csv(csv_path: Path) -> Dict[str, float]:
    """
    Carrega um arquivo CSV de métricas e retorna um dicionário estruturado.

    Requisitos:
    - O CSV deve conter exatamente uma linha de métricas
    - Os campos devem seguir o padrão definido em METRICS_FIELDS

    Args:
        csv_path (Path): Caminho para o arquivo CSV.

    Returns:
        Dict[str, float]: Métricas convertidas para float.

    Raises:
        ValueError: Caso o CSV não contenha exatamente uma linha de dados.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 1:
        raise ValueError(
            f"O arquivo {csv_path.name} deve conter exatamente uma linha de métricas"
        )

    return {metric: float(rows[0][metric]) for metric in METRICS_FIELDS}


def extract_model_name(csv_path: Path) -> str:
    """
    Extrai o nome do modelo a partir do nome do arquivo CSV.

    Exemplo:
        - yolo_test.csv        -> YOLO
        - faster_rcnn_test.csv -> FASTER RCNN

    Args:
        csv_path (Path): Caminho para o arquivo CSV.

    Returns:
        str: Nome normalizado do modelo.
    """
    return (
        csv_path.stem
        .replace("_test", "")
        .replace("_", " ")
        .upper()
    )


# ==============================
# COMPARATOR PRINCIPAL
# ==============================

def compare_models(csv_files: List[Path]) -> Dict[str, Dict]:
    """
    Consolida métricas de avaliação de múltiplos modelos.

    Args:
        csv_files (List[Path]): Lista de arquivos CSV de métricas.

    Returns:
        Dict[str, Dict]: Resultados consolidados por modelo.
    """
    results: Dict[str, Dict] = {}

    for csv_file in csv_files:
        model_name = extract_model_name(csv_file)
        metrics = load_metrics_csv(csv_file)

        results[model_name] = {
            "metrics": metrics,
            "computational_cost": "not_measured",
            "implementation_complexity": "not_measured",
        }

    return results


def compute_rankings(results: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Gera rankings de modelos para cada métrica avaliada.

    Args:
        results (Dict[str, Dict]): Resultados consolidados por modelo.

    Returns:
        Dict[str, List[str]]: Rankings ordenados por métrica.
    """
    rankings: Dict[str, List[str]] = {}

    for metric in METRICS_FIELDS:
        rankings[metric] = sorted(
            results.keys(),
            key=lambda model: results[model]["metrics"][metric],
            reverse=True,
        )

    return rankings


# ==============================
# EXPORTAÇÃO
# ==============================

def export_csv(results: Dict[str, Dict], output_path: Path) -> None:
    """
    Exporta um CSV consolidado contendo métricas por modelo.

    Args:
        results (Dict[str, Dict]): Resultados consolidados.
        output_path (Path): Caminho de saída do arquivo CSV.
    """
    fieldnames = ["model", *METRICS_FIELDS]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, data in results.items():
            row = {"model": model_name}
            row.update(data["metrics"])
            writer.writerow(row)


def export_json(
    results: Dict[str, Dict],
    rankings: Dict[str, List[str]],
    output_path: Path,
) -> None:
    """
    Exporta um JSON técnico estruturado contendo métricas e rankings.

    Args:
        results (Dict[str, Dict]): Resultados consolidados por modelo.
        rankings (Dict[str, List[str]]): Rankings por métrica.
        output_path (Path): Caminho de saída do arquivo JSON.
    """
    payload = {
        "models": results,
        "rankings": rankings,
        "notes": {
            "computational_cost": "not measured in this stage",
            "implementation_complexity": "not measured in this stage",
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

