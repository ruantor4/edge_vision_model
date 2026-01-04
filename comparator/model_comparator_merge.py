"""
model_comparator_merge.py

Responsável por consolidar os resultados finais da comparação
entre modelos de detecção de objetos.

Este módulo:
- Lê os artefatos gerados pelos comparators de métricas e custo computacional
- Realiza a consolidação por modelo
- Exporta os resultados finais em CSV e JSON

Observações:
- NÃO recalcula métricas
- NÃO executa inferência
- NÃO interpreta resultados
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

METRICS_FILE = ARTIFACTS_COMPARISONS_DIR / "models_comparison.csv"
COST_FILE = ARTIFACTS_COMPARISONS_DIR / "models_cost_comparison.csv"

OUTPUT_CSV = ARTIFACTS_COMPARISONS_DIR / "models_final_comparison.csv"
OUTPUT_JSON = ARTIFACTS_COMPARISONS_DIR / "models_final_comparison.json"


# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def load_csv_as_dict(csv_path: Path, key_field: str) -> Dict[str, Dict]:
    """
    Carrega um CSV e indexa os registros por um campo chave.
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
    Realiza merge entre métricas e custo computacional.
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
    Exporta CSV final consolidado.
    """
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(next(iter(merged_data.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in merged_data.values():
            writer.writerow(row)


def export_json(merged_data: Dict[str, Dict]) -> None:
    """
    Exporta JSON final consolidado.
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


# ==============================
# MAIN
# ==============================

def main() -> None:
    """
    Ponto de entrada do consolidador final.
    """
    metrics_data = load_csv_as_dict(METRICS_FILE, key_field="model")
    cost_data = load_csv_as_dict(COST_FILE, key_field="model")

    merged_data = merge_results(metrics_data, cost_data)

    export_csv(merged_data)
    export_json(merged_data)


if __name__ == "__main__":
    main()
