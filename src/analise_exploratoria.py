"""Gera tabelas e gráficos para responder às perguntas do Datathon.

Execute na raiz do projeto:
    .venv\\Scripts\\python.exe src\\analise_exploratoria.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BASE_PEDE_CONSOLIDADA_TRATADA_V2.xlsx"
OUT = ROOT / "outputs" / "eda"
FIG = OUT / "figuras"
TAB = OUT / "tabelas"

INDICADORES = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
PALETTE = ["#1D4ED8", "#F59E0B", "#DC2626", "#059669", "#7C3AED"]


def salvar(nome: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / nome, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def preparar() -> pd.DataFrame:
    df = pd.read_excel(DATA, sheet_name="BASE_CONSOLIDADA")
    df["ANO_REFERENCIA"] = df["ANO_REFERENCIA"].astype(int)
    # Definição operacional; deve ser validada com o dicionário/regra de negócio.
    df["NIVEL_DEFASAGEM"] = pd.cut(
        df["DEFASAGEM"],
        bins=[-np.inf, -2, -1, np.inf],
        labels=["Severa (≤ -2)", "Moderada (-1)", "Sem defasagem (≥ 0)"],
    )
    return df


def grafico_defasagem(df: pd.DataFrame) -> None:
    ordem = ["Severa (≤ -2)", "Moderada (-1)", "Sem defasagem (≥ 0)"]
    tabela = pd.crosstab(
        df["ANO_REFERENCIA"], df["NIVEL_DEFASAGEM"], normalize="index"
    )[ordem]
    tabela.to_csv(TAB / "defasagem_por_ano.csv", encoding="utf-8-sig")
    ax = tabela.plot(
        kind="bar", stacked=True, figsize=(9, 5), color=[PALETTE[2], PALETTE[1], PALETTE[3]]
    )
    ax.set(title="Perfil de defasagem por ano", xlabel="", ylabel="Percentual de alunos")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    for container in ax.containers:
        labels = [f"{value:.1%}" if value >= 0.04 else "" for value in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9)
    sns.despine()
    salvar("01_perfil_defasagem.png")


def grafico_indicadores(df: pd.DataFrame) -> None:
    medias = df.groupby("ANO_REFERENCIA")[INDICADORES].mean()
    medias.to_csv(TAB / "medias_indicadores_por_ano.csv", encoding="utf-8-sig")
    longo = medias.reset_index().melt(
        "ANO_REFERENCIA", var_name="Indicador", value_name="Média"
    )
    grid = sns.FacetGrid(
        longo, col="Indicador", col_wrap=4, sharey=False, height=2.7, aspect=1.15
    )
    grid.map_dataframe(
        sns.lineplot, x="ANO_REFERENCIA", y="Média", marker="o", color=PALETTE[0]
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Ano", "Média")
    for ax in grid.axes.flat:
        ax.set_xticks([2022, 2023, 2024])
        ax.grid(axis="y", alpha=0.2)
    grid.figure.suptitle("Evolução anual dos indicadores", y=1.03, fontsize=14)
    grid.figure.savefig(
        FIG / "02_evolucao_indicadores.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(grid.figure)


def grafico_ida_fase(df: pd.DataFrame) -> None:
    tabela = df.pivot_table(
        index="FASE_NUMERICA", columns="ANO_REFERENCIA", values="IDA", aggfunc="mean"
    )
    tabela.to_csv(TAB / "ida_medio_fase_ano.csv", encoding="utf-8-sig")
    plt.figure(figsize=(7, 6))
    sns.heatmap(tabela, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=10)
    plt.title("IDA médio por fase e ano")
    plt.xlabel("Ano")
    plt.ylabel("Fase")
    salvar("03_ida_por_fase_ano.png")


def grafico_engajamento(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(
        data=df,
        x="IEG",
        y="IDA",
        scatter_kws={"alpha": 0.18, "s": 14},
        line_kws={"color": PALETTE[2]},
        ax=axes[0],
    )
    sns.regplot(
        data=df,
        x="IEG",
        y="IPV",
        scatter_kws={"alpha": 0.18, "s": 14},
        line_kws={"color": PALETTE[2]},
        ax=axes[1],
    )
    axes[0].set_title(f"IEG × IDA (Spearman = {df[['IEG','IDA']].corr('spearman').iloc[0,1]:.2f})")
    axes[1].set_title(f"IEG × IPV (Spearman = {df[['IEG','IPV']].corr('spearman').iloc[0,1]:.2f})")
    fig.suptitle("Engajamento, desempenho e ponto de virada")
    salvar("04_engajamento_desempenho_ipv.png")


def grafico_correlacoes(df: pd.DataFrame) -> None:
    corr = df[INDICADORES].corr(method="spearman")
    corr.to_csv(TAB / "correlacao_spearman.csv", encoding="utf-8-sig")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Relações entre os indicadores (Spearman)")
    salvar("05_correlacoes_indicadores.png")


def grafico_autoavaliacao(df: pd.DataFrame) -> None:
    base = df.dropna(subset=["IAA", "IDA", "IEG"]).copy()
    base["REFERENCIA_REAL"] = base[["IDA", "IEG"]].mean(axis=1)
    base["DIFERENCA_IAA"] = base["IAA"] - base["REFERENCIA_REAL"]
    resumo = base.groupby("ANO_REFERENCIA")["DIFERENCA_IAA"].describe()
    resumo.to_csv(TAB / "coerencia_autoavaliacao.csv", encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=base,
        x="ANO_REFERENCIA",
        y="DIFERENCA_IAA",
        color=PALETTE[0],
        showfliers=False,
    )
    plt.axhline(0, color=PALETTE[2], linestyle="--", linewidth=1)
    plt.title("Autoavaliação menos média de desempenho e engajamento")
    plt.xlabel("Ano")
    plt.ylabel("IAA − média(IDA, IEG)")
    sns.despine()
    salvar("06_coerencia_autoavaliacao.png")


def grafico_coorte(df: pd.DataFrame) -> None:
    alunos_3_anos = df.groupby("RA")["ANO_REFERENCIA"].nunique()
    ids = alunos_3_anos[alunos_3_anos == 3].index
    coorte = df[df["RA"].isin(ids)]
    medias = coorte.groupby("ANO_REFERENCIA")[INDICADORES].mean()
    medias.to_csv(TAB / "coorte_3_anos_medias.csv", encoding="utf-8-sig")
    longo = medias[["IDA", "IEG", "IAA", "IPS", "IPV", "INDE"]].reset_index().melt(
        "ANO_REFERENCIA", var_name="Indicador", value_name="Média"
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=longo,
        x="ANO_REFERENCIA",
        y="Média",
        hue="Indicador",
        marker="o",
        palette="tab10",
    )
    plt.xticks([2022, 2023, 2024])
    plt.title(f"Evolução da coorte acompanhada nos três anos (n={len(ids)})")
    plt.xlabel("Ano")
    plt.ylabel("Média")
    plt.legend(title="", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    sns.despine()
    salvar("07_coorte_tres_anos.png")


def grafico_transicao_pedra(df: pd.DataFrame) -> None:
    base = df.sort_values(["RA", "ANO_REFERENCIA"]).copy()
    base["PEDRA_PROXIMO_ANO"] = base.groupby("RA")["PEDRA"].shift(-1)
    base["ANO_PROXIMO"] = base.groupby("RA")["ANO_REFERENCIA"].shift(-1)
    base = base[base["ANO_PROXIMO"] == base["ANO_REFERENCIA"] + 1]
    pedras = ["Quartzo", "Ágata", "Ametista", "Topázio"]
    tabela = pd.crosstab(
        base["PEDRA"], base["PEDRA_PROXIMO_ANO"], normalize="index"
    ).reindex(index=pedras, columns=pedras)
    tabela.to_csv(TAB / "transicao_pedra.csv", encoding="utf-8-sig")
    plt.figure(figsize=(7, 6))
    sns.heatmap(tabela, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1)
    plt.title("Transição de classificação para o ano seguinte")
    plt.xlabel("Classificação no ano seguinte")
    plt.ylabel("Classificação no ano atual")
    salvar("08_transicao_pedra.png")


def grafico_cobertura(df: pd.DataFrame) -> None:
    cobertura = (
        df.groupby("ANO_REFERENCIA")[INDICADORES]
        .agg(lambda serie: serie.notna().mean())
        .T
    )
    cobertura.to_csv(TAB / "cobertura_indicadores.csv", encoding="utf-8-sig")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cobertura, annot=True, fmt=".0%", cmap="Greens", vmin=0, vmax=1)
    plt.title("Cobertura dos indicadores por ano")
    plt.xlabel("Ano")
    plt.ylabel("Indicador")
    salvar("09_cobertura_indicadores.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    df = preparar()
    grafico_defasagem(df)
    grafico_indicadores(df)
    grafico_ida_fase(df)
    grafico_engajamento(df)
    grafico_correlacoes(df)
    grafico_autoavaliacao(df)
    grafico_coorte(df)
    grafico_transicao_pedra(df)
    grafico_cobertura(df)
    print(f"Gráficos salvos em: {FIG}")
    print(f"Tabelas salvas em: {TAB}")


if __name__ == "__main__":
    main()
