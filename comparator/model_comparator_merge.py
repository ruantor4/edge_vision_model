"""
model_comparator_merge.py

Responsável por consolidar os resultados finais da comparação
entre modelos de detecção de objetos.

Este módulo atua exclusivamente na etapa de consolidação,
unificando os artefatos gerados pelos comparators de métricas
e de custo computacional.

Escopo:
- Leitura dos arquivos CSV gerados pelos comparators
- Consolidação dos resultados por modelo
- Padronização e prefixação das colunas de custo computacional
- Exportação do resultado final em CSV e JSON
"""

from pathlib import Path
import csv
import json
from typing import Dict

from config.settings import (
    ARTIFACTS_COMPARISONS_DIR,
)


# ==============================
# CONFIGURAÇÃO
# ==============================

# Arquivo de métricas consolidadas (AP/AR)
METRICS_FILE = ARTIFACTS_COMPARISONS_DIR / "models_comparison.csv"

# Arquivo de custo computacional (tempo, FPS, VRAM)
COST_FILE = ARTIFACTS_COMPARISONS_DIR / "models_cost_comparison.csv"

# Artefatos finais consolidados
OUTPUT_CSV = ARTIFACTS_COMPARISONS_DIR / "models_final_comparison.csv"
OUTPUT_JSON = ARTIFACTS_COMPARISONS_DIR / "models_final_comparison.json"


# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def load_csv_as_dict(csv_path: Path, key_field: str) -> Dict[str, Dict]:
    """
    Carrega um arquivo CSV e indexa seus registros por um campo chave.

    Args:
        csv_path (Path): Caminho para o arquivo CSV.
        key_field (str): Campo utilizado como chave primária.

    Returns:
        Dict[str, Dict]: Dicionário indexado pelo campo chave.

    Raises:
        FileNotFoundError: Caso o arquivo CSV não exista.
        ValueError: Caso o CSV esteja vazio ou inválido.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = {}

        for row in reader:
            key = row[key_field]
            data[key] = row

    if not data:
        raise ValueError(f"CSV vazio ou inválido: {csv_path.name}")

    return data


def merge_results(
    metrics_data: Dict[str, Dict],
    cost_data: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Realiza a consolidação entre métricas de avaliação e custo computacional.

    Para cada modelo:
    - Mantém as métricas originais
    - Prefixa os campos de custo computacional com 'cost_'
    - Converte valores numéricos para float
    - Preserva valores ausentes como None

    Args:
        metrics_data (Dict[str, Dict]): Métricas de avaliação por modelo.
        cost_data (Dict[str, Dict]): Métricas de custo computacional por modelo.

    Returns:
        Dict[str, Dict]: Dados consolidados por modelo.

    Raises:
        ValueError: Caso um modelo presente nas métricas não exista nos dados de custo.
    """
    merged: Dict[str, Dict] = {}

    for model, metrics in metrics_data.items():
        if model not in cost_data:
            raise ValueError(
                f"Modelo '{model}' presente nas métricas "
                f"não encontrado nos dados de custo."
            )

        merged[model] = {
            "model": model,
            **{k: float(v) for k, v in metrics.items() if k != "model"},
            **{
                f"cost_{k}": (
                    float(v) if v not in ("", None) else None
                )
                for k, v in cost_data[model].items()
                if k != "model"
            },
        }

    return merged


# ==============================
# EXPORTAÇÃO
# ==============================

def export_csv(merged_data: Dict[str, Dict]) -> None:
    """
    Exporta o resultado final consolidado em formato CSV.

    Args:
        merged_data (Dict[str, Dict]): Dados consolidados por modelo.
    """
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(next(iter(merged_data.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in merged_data.values():
            writer.writerow(row)


def export_json(merged_data: Dict[str, Dict]) -> None:
    """
    Exporta o resultado final consolidado em formato JSON.

    O payload inclui referência explícita aos arquivos de origem,
    garantindo rastreabilidade do processo de comparação.

    Args:
        merged_data (Dict[str, Dict]): Dados consolidados por modelo.
    """
    payload = {
        "models": merged_data,
        "source_files": {
            "metrics": METRICS_FILE.name,
            "cost": COST_FILE.name,
        },
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

