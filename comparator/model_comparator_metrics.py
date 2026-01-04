"""
model_comparator.py

Responsável por consolidar e comparar métricas de avaliação
entre diferentes modelos de detecção de objetos.

Escopo desta versão:
- Leitura de CSVs de métricas (COCO-style)
- Consolidação por modelo
- Geração de rankings por métrica
- Exportação de resultados em CSV e JSON

Observação:
- Custo computacional e facilidade de implementação
  NÃO são medidos nesta etapa.
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
    Carrega um CSV de métricas e retorna um dicionário.

    Requisitos:
    - CSV deve conter exatamente uma linha de métricas
    - Campos devem seguir o padrão COCO definido em METRICS_FIELDS
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
    Consolida métricas de múltiplos modelos.

    Retorna um dicionário estruturado por modelo.
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
    Gera rankings de modelos para cada métrica definida.
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
    Exporta um CSV consolidado com métricas por modelo.
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
    Exporta um JSON técnico estruturado, contendo métricas e rankings.
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


# ==============================
# MAIN
# ==============================

def main() -> None:
    """
    Ponto de entrada do comparator de modelos.

    Espera que os arquivos *_test.csv estejam presentes em:
    ARTIFACTS_METRICS_DIR
    """
    csv_files = [
        ARTIFACTS_METRICS_DIR / "faster_rcnn_test.csv",
        ARTIFACTS_METRICS_DIR / "ssd_test.csv",
        ARTIFACTS_METRICS_DIR / "yolo_test.csv",
    ]

    # Validação explícita de existência dos arquivos
    for csv_file in csv_files:
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Arquivo de métricas não encontrado: {csv_file}"
            )

    results = compare_models(csv_files)
    rankings = compute_rankings(results)

    ARTIFACTS_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    export_csv(
        results,
        ARTIFACTS_COMPARISONS_DIR / "models_comparison.csv",
    )

    export_json(
        results,
        rankings,
        ARTIFACTS_COMPARISONS_DIR / "models_comparison.json",
    )


if __name__ == "__main__":
    main()
