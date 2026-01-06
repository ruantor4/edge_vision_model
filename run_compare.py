"""
run_compare.py

Interface de linha de comando para executar a
ETAPA DE COMPARAÇÃO da pipeline edge-vision-model.

Este script atua como orquestrador da fase analítica do projeto,
coordenando comparações entre modelos já avaliados.

Responsabilidades:
- Configurar logging da aplicação
- Executar a consolidação de métricas de avaliação (COCO)
- Executar o benchmark de custo computacional
- Realizar o merge final dos resultados analíticos
"""
import logging

from utils.logging_global import setup_logging

from comparator.model_comparator_cost import (
    compare_models_cost,
    export_results,
)

from comparator.model_comparator_metrics import (
    compare_models,
    compute_rankings,
    export_csv,
    export_json,
)

from comparator.model_comparator_merge import (
    load_csv_as_dict,
    merge_results,
    export_csv,
    export_json,
)

from config.settings import (
    ARTIFACTS_METRICS_DIR,
    ARTIFACTS_COMPARISONS_DIR,
)


def main() -> None:
    """
    Ponto de entrada da etapa de comparação da pipeline.

    Executa, em ordem:
    1. Benchmark de custo computacional
    2. Consolidação de métricas de avaliação
    3. Merge final dos resultados analíticos
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=== INICIANDO ETAPA DE COMPARAÇÃO ===")

    ARTIFACTS_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. CUSTO COMPUTACIONAL
    # ============================================================
    logger.info("Executando benchmark de custo computacional")

    cost_results = compare_models_cost()
    export_results(cost_results)

    # ============================================================
    # 2. MÉTRICAS DE AVALIAÇÃO
    # ============================================================
    logger.info("Consolidando métricas de avaliação")

    csv_files = [
        ARTIFACTS_METRICS_DIR / "faster_rcnn_test.csv",
        ARTIFACTS_METRICS_DIR / "ssd_test.csv",
        ARTIFACTS_METRICS_DIR / "yolo_test.csv",
    ]

    for csv_file in csv_files:
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Arquivo de métricas não encontrado: {csv_file}"
            )

    metrics_results = compare_models(csv_files)
    rankings = compute_rankings(metrics_results)

    export_csv(
        metrics_results,
        ARTIFACTS_COMPARISONS_DIR / "models_comparison.csv",
    )

    export_json(
        metrics_results,
        rankings,
        ARTIFACTS_COMPARISONS_DIR / "models_comparison.json",
    )

    # ============================================================
    # 3. MERGE FINAL
    # ============================================================
    logger.info("Consolidando resultados finais")

    metrics_data = load_csv_as_dict(
        ARTIFACTS_COMPARISONS_DIR / "models_comparison.csv",
        key_field="model",
    )

    cost_data = load_csv_as_dict(
        ARTIFACTS_COMPARISONS_DIR / "models_cost_comparison.csv",
        key_field="model",
    )

    merged_data = merge_results(metrics_data, cost_data)

    export_csv(merged_data)
    export_json(merged_data)

    logger.info("=== ETAPA DE COMPARAÇÃO CONCLUÍDA ===")


if __name__ == "__main__":
    main()