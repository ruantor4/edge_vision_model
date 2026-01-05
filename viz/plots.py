"""
plots.py

Responsável por gerar visualizações comparativas a partir
dos resultados finais consolidados do projeto.

Este módulo:
- Lê o arquivo models_final_comparison.csv
- Gera gráficos comparativos de qualidade e custo computacional
- Salva os gráficos como imagens em artifacts/comparisons/plots

Observações:
- NÃO recalcula métricas
- NÃO executa inferência
- NÃO interpreta resultados
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import (
    ARTIFACTS_FINAL_COMPARISON_CSV,
    ARTIFACTS_PLOTS_DIR,
)

# ==============================
# FUNÇÕES AUXILIARES
# ==============================

def load_final_comparison() -> pd.DataFrame:
    """
    Carrega o CSV final consolidado de comparação entre modelos.
    """
    if not ARTIFACTS_FINAL_COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo de comparação final não encontrado: "
            f"{ARTIFACTS_FINAL_COMPARISON_CSV}"
        )

    return pd.read_csv(ARTIFACTS_FINAL_COMPARISON_CSV)


def save_bar_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
) -> None:
    """
    Gera e salva um gráfico de barras simples para uma métrica específica.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric])
    plt.title(title)
    plt.xlabel("Modelo")
    plt.ylabel(ylabel)
    plt.tight_layout()

    output_path = ARTIFACTS_PLOTS_DIR / output_name
    plt.savefig(output_path)
    plt.close()

# ==============================
# PLOTS DE QUALIDADE
# ==============================

def plot_quality_metrics(df: pd.DataFrame) -> None:
    """
    Gera gráficos relacionados às métricas de qualidade dos modelos.
    """
    save_bar_plot(
        df=df,
        metric="AP_50_95_all",
        title="AP_50_95 – Desempenho Global",
        ylabel="AP",
        output_name="ap_50_95_all.png",
    )

    save_bar_plot(
        df=df,
        metric="AP_75_all",
        title="AP_75 – Precisão Espacial",
        ylabel="AP",
        output_name="ap_75_all.png",
    )

    save_bar_plot(
        df=df,
        metric="AR_50_95_all",
        title="AR_50_95 – Recall",
        ylabel="AR",
        output_name="ar_50_95_all.png",
    )

# ==============================
# PLOTS DE CUSTO COMPUTACIONAL
# ==============================

def plot_cost_metrics(df: pd.DataFrame) -> None:
    """
    Gera gráficos relacionados ao custo computacional dos modelos.
    """
    save_bar_plot(
        df=df,
        metric="cost_mean_inference_time_ms",
        title="Tempo Médio de Inferência por Imagem (ms)",
        ylabel="Tempo (ms)",
        output_name="mean_inference_time_ms.png",
    )

    save_bar_plot(
        df=df,
        metric="cost_fps",
        title="Frames por Segundo (FPS)",
        ylabel="FPS",
        output_name="fps.png",
    )

    save_bar_plot(
        df=df,
        metric="cost_vram_peak_mb",
        title="Pico de Uso de VRAM",
        ylabel="VRAM (MB)",
        output_name="vram_peak_mb.png",
    )

# ==============================
# MAIN
# ==============================

def main() -> None:
    """
    Ponto de entrada para geração das visualizações comparativas.
    """
    ARTIFACTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_final_comparison()

    plot_quality_metrics(df)
    plot_cost_metrics(df)


if __name__ == "__main__":
    main()