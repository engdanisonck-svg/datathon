"""Dashboard analÃ­tico do Datathon Passos MÃ¡gicos.

ExecuÃ§Ã£o:
    .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "BASE_PEDE_CONSOLIDADA_TRATADA_V2.xlsx"
INDICADORES = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
CORES_DEFASAGEM = {
    "Severa (â‰¤ -2)": "#DC2626",
    "Moderada (-1)": "#F59E0B",
    "Sem defasagem (â‰¥ 0)": "#059669",
}


st.set_page_config(
    page_title="Passos MÃ¡gicos | Datathon",
    page_icon="ðŸ“Š",
    layout="wide",
)


@st.cache_data(show_spinner="Carregando a base consolidada...")
def carregar_dados(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="BASE_CONSOLIDADA")
    df["ANO_REFERENCIA"] = df["ANO_REFERENCIA"].astype(int)
    df["NIVEL_DEFASAGEM"] = pd.cut(
        df["DEFASAGEM"],
        bins=[-np.inf, -2, -1, np.inf],
        labels=["Severa (â‰¤ -2)", "Moderada (-1)", "Sem defasagem (â‰¥ 0)"],
    )
    return df


def formatar_numero(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    anos = sorted(df["ANO_REFERENCIA"].dropna().unique())
    anos_escolhidos = st.sidebar.multiselect("Ano", anos, default=anos)

    fases = sorted(df["FASE_NUMERICA"].dropna().unique())
    fases_escolhidas = st.sidebar.multiselect("Fase", fases, default=fases)

    generos = sorted(df["GENERO"].dropna().unique())
    generos_escolhidos = st.sidebar.multiselect("GÃªnero", generos, default=generos)

    instituicoes = sorted(df["INSTITUICAO_ENSINO"].dropna().astype(str).unique())
    instituicoes_escolhidas = st.sidebar.multiselect(
        "Instituição de ensino", instituicoes, default=instituicoes
    )

    filtrado = df[
        df["ANO_REFERENCIA"].isin(anos_escolhidos)
        & df["FASE_NUMERICA"].isin(fases_escolhidas)
        & df["GENERO"].isin(generos_escolhidos)
        & df["INSTITUICAO_ENSINO"].astype(str).isin(instituicoes_escolhidas)
    ].copy()
    st.sidebar.caption(f"{formatar_numero(len(filtrado))} registros selecionados")
    return filtrado


def grafico_sem_dados(mensagem: str = "Não há¡ dados para os filtros selecionados.") -> None:
    st.info(mensagem)


def visao_geral(df: pd.DataFrame) -> None:
    st.subheader("Visão geral")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", formatar_numero(len(df)))
    col2.metric("Alunos Ãºnicos", formatar_numero(df["RA"].nunique()))
    col3.metric("INDE médio", f"{df['INDE'].mean():.2f}" if df["INDE"].notna().any() else "â€”")
    revisar = (df["QUALIDADE_REGISTRO"] == "REVISAR").mean()
    col4.metric("Registros para revisar", f"{revisar:.1%}")

    esquerda, direita = st.columns(2)
    por_ano = df.groupby("ANO_REFERENCIA", as_index=False).agg(
        Registros=("REGISTRO_ID", "size"), Alunos=("RA", "nunique")
    )
    esquerda.plotly_chart(
        px.bar(
            por_ano,
            x="ANO_REFERENCIA",
            y="Registros",
            text_auto=True,
            title="Registros por ano",
            labels={"ANO_REFERENCIA": "Ano"},
        ),
        width="stretch",
    )
    qualidade = df["QUALIDADE_REGISTRO"].value_counts().rename_axis("Qualidade").reset_index(name="Registros")
    direita.plotly_chart(
        px.bar(
            qualidade,
            x="Qualidade",
            y="Registros",
            text_auto=True,
            color="Qualidade",
            title="Qualidade dos registros",
            color_discrete_map={"OK": "#059669", "REVISAR": "#DC2626"},
        ),
        width="stretch",
    )

    cobertura = (
        df.groupby("ANO_REFERENCIA")[INDICADORES]
        .agg(lambda serie: serie.notna().mean())
        .T
    )
    st.plotly_chart(
        px.imshow(
            cobertura,
            text_auto=".0%",
            aspect="auto",
            zmin=0,
            zmax=1,
            color_continuous_scale="Greens",
            title="Cobertura dos indicadores",
            labels={"x": "Ano", "y": "Indicador", "color": "Cobertura"},
        ),
        width="stretch",
    )


def questao_1(df: pd.DataFrame) -> None:
    st.subheader("1. Adequação do ní­vel IAN e defasagem")
    st.caption(
        "Definição operacional usada: moderada = -1; severa -2. "
        "A regra deve ser confirmada com a área de negócio."
    )
    tabela = (
        df.groupby(["ANO_REFERENCIA", "NIVEL_DEFASAGEM"], observed=True)
        .size()
        .rename("Alunos")
        .reset_index()
    )
    tabela["Percentual"] = tabela.groupby("ANO_REFERENCIA")["Alunos"].transform(
        lambda valores: valores / valores.sum()
    )
    fig = px.bar(
        tabela,
        x="ANO_REFERENCIA",
        y="Percentual",
        color="NIVEL_DEFASAGEM",
        text=tabela["Percentual"].map(lambda valor: f"{valor:.1%}"),
        barmode="stack",
        color_discrete_map=CORES_DEFASAGEM,
        labels={"ANO_REFERENCIA": "Ano", "NIVEL_DEFASAGEM": "Ní­vel"},
        title="Perfil de defasagem por ano",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


def questao_2(df: pd.DataFrame) -> None:
    st.subheader("2. Desempenho acadêmico” IDA")
    medias = df.groupby("ANO_REFERENCIA", as_index=False)["IDA"].mean()
    esquerda, direita = st.columns([1, 1.4])
    esquerda.plotly_chart(
        px.line(
            medias,
            x="ANO_REFERENCIA",
            y="IDA",
            markers=True,
            title="IDA médio por ano",
            labels={"ANO_REFERENCIA": "Ano", "IDA": "IDA médio"},
        ),
        width="stretch",
    )
    fase_ano = df.pivot_table(
        index="FASE_NUMERICA", columns="ANO_REFERENCIA", values="IDA", aggfunc="mean"
    )
    direita.plotly_chart(
        px.imshow(
            fase_ano,
            text_auto=".2f",
            aspect="auto",
            zmin=0,
            zmax=10,
            color_continuous_scale="YlGnBu",
            title="IDA médio por fase e ano",
            labels={"x": "Ano", "y": "Fase", "color": "IDA"},
        ),
        width="stretch",
    )


def questao_3(df: pd.DataFrame) -> None:
    st.subheader("3. Engajamento em relação do IEG com IDA e IPV")
    col1, col2 = st.columns(2)
    for coluna, eixo, titulo in [
        (col1, "IDA", "IEG Ã— desempenho acadêmico"),
        (col2, "IPV", "IEG Ã— ponto de virada"),
    ]:
        base = df.dropna(subset=["IEG", eixo])
        rho = base[["IEG", eixo]].corr(method="spearman").iloc[0, 1]
        fig = px.scatter(
            base,
            x="IEG",
            y=eixo,
            color="ANO_REFERENCIA",
            opacity=0.35,
            title=f"{titulo} Â· Spearman = {rho:.2f}",
            hover_data=["RA", "FASE_NUMERICA"],
        )
        coluna.plotly_chart(fig, width="stretch")


def questao_4(df: pd.DataFrame) -> None:
    st.subheader("4. Autoavaliação de coerência do IAA")
    base = df.dropna(subset=["IAA", "IDA", "IEG"]).copy()
    base["REFERENCIA_OBSERVADA"] = base[["IDA", "IEG"]].mean(axis=1)
    base["DIFERENCA_IAA"] = base["IAA"] - base["REFERENCIA_OBSERVADA"]
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.scatter(
            base,
            x="REFERENCIA_OBSERVADA",
            y="IAA",
            color="ANO_REFERENCIA",
            opacity=0.4,
            title="IAA versus média de IDA e IEG",
            labels={"REFERENCIA_OBSERVADA": "Média de IDA e IEG"},
        ),
        width="stretch",
    )
    col2.plotly_chart(
        px.box(
            base,
            x="ANO_REFERENCIA",
            y="DIFERENCA_IAA",
            points=False,
            title="DiferenÃ§a entre percepção e referência observada",
            labels={"DIFERENCA_IAA": "IAA ’ média(IDA, IEG)", "ANO_REFERENCIA": "Ano"},
        ),
        width="stretch",
    )
    st.caption(
        "Valores positivos indicam autoavaliação acima da média combinada de "
        "desempenho e engajamento; nÃo representam, isoladamente, erro de percepção."
    )


def construir_transicoes(df: pd.DataFrame) -> pd.DataFrame:
    base = df.sort_values(["RA", "ANO_REFERENCIA"]).copy()
    for coluna in ["ANO_REFERENCIA", "IDA", "IEG", "DEFASAGEM"]:
        base[f"PROX_{coluna}"] = base.groupby("RA")[coluna].shift(-1)
    base = base[base["PROX_ANO_REFERENCIA"] == base["ANO_REFERENCIA"] + 1].copy()
    base["DELTA_IDA"] = base["PROX_IDA"] - base["IDA"]
    base["DELTA_IEG"] = base["PROX_IEG"] - base["IEG"]
    return base


def questao_5(df_completo: pd.DataFrame, ids_filtrados: set[str]) -> None:
    st.subheader("5. Aspectos psicossociais â€” IPS antes de quedas")
    transicoes = construir_transicoes(df_completo)
    transicoes = transicoes[transicoes["RA"].isin(ids_filtrados)]
    if transicoes.empty:
        grafico_sem_dados("NÃo há¡ transições anuais para os alunos selecionados.")
        return
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.scatter(
            transicoes,
            x="IPS",
            y="DELTA_IDA",
            color="ANO_REFERENCIA",
            opacity=0.4,
            title="IPS atual Ã— variação do IDA no ano seguinte",
            labels={"DELTA_IDA": "IDA seguinte âˆ’ IDA atual"},
        ),
        width="stretch",
    )
    col2.plotly_chart(
        px.scatter(
            transicoes,
            x="IPS",
            y="DELTA_IEG",
            color="ANO_REFERENCIA",
            opacity=0.4,
            title="IPS atual Ã— variação do IEG no ano seguinte",
            labels={"DELTA_IEG": "IEG seguinte âˆ’ IEG atual"},
        ),
        width="stretch",
    )
    st.caption(
        "Esta visão temporal mais apropriada que uma correlação do mesmo ano, "
        "mas ainda mostra associação, não causalidade."
    )


def questao_6(df: pd.DataFrame) -> None:
    st.subheader("6. Aspectos psicopedagógicos â€” IPP e adequações")
    base = df.dropna(subset=["IPP", "DEFASAGEM"]).copy()
    if base.empty:
        grafico_sem_dados("IPP não estão¡ disponí­vel para os filtros selecionados.")
        return
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.box(
            base,
            x="NIVEL_DEFASAGEM",
            y="IPP",
            color="NIVEL_DEFASAGEM",
            color_discrete_map=CORES_DEFASAGEM,
            points=False,
            title="Distribuição do IPP por nÃ­vel de defasagem",
            labels={"NIVEL_DEFASAGEM": "NÃ­vel de defasagem"},
        ),
        width="stretch",
    )
    resumo = (
        base.groupby(["ANO_REFERENCIA", "NIVEL_DEFASAGEM"], observed=True)["IPP"]
        .mean()
        .reset_index()
    )
    col2.plotly_chart(
        px.line(
            resumo,
            x="ANO_REFERENCIA",
            y="IPP",
            color="NIVEL_DEFASAGEM",
            markers=True,
            color_discrete_map=CORES_DEFASAGEM,
            title="IPP médio por nÃ­vel e ano",
            labels={"ANO_REFERENCIA": "Ano", "NIVEL_DEFASAGEM": "Ní­vel"},
        ),
        width="stretch",
    )
    st.info("O IPP não possui observações em 2022; comparações temporais começam em 2023.")


def questao_7(df: pd.DataFrame) -> None:
    st.subheader("7. Ponto de virada â€” fatores associados ao IPV")
    fatores = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP"]
    correlacoes = (
        df[fatores + ["IPV"]]
        .corr(method="spearman")["IPV"]
        .drop("IPV")
        .sort_values()
        .rename("Correlação")
        .rename_axis("Indicador")
        .reset_index()
    )
    col1, col2 = st.columns([0.8, 1.2])
    col1.plotly_chart(
        px.bar(
            correlacoes,
            x="Correlação",
            y="Indicador",
            orientation="h",
            text_auto=".2f",
            title="Associações de cada indicador com o IPV",
        ),
        width="stretch",
    )
    matriz = df[fatores + ["IPV"]].corr(method="spearman")
    col2.plotly_chart(
        px.imshow(
            matriz,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Matriz de correlação",
        ),
        width="stretch",
    )


def questao_8(df: pd.DataFrame) -> None:
    st.subheader("8. Multidimensionalidade â€” indicadores associados ao INDE")
    fatores = ["IDA", "IEG", "IPS", "IPP"]
    correlacoes = (
        df[fatores + ["INDE"]]
        .corr(method="spearman")["INDE"]
        .drop("INDE")
        .sort_values()
        .rename("Correlação")
        .rename_axis("Indicador")
        .reset_index()
    )
    col1, col2 = st.columns([0.8, 1.2])
    col1.plotly_chart(
        px.bar(
            correlacoes,
            x="Correlação",
            y="Indicador",
            orientation="h",
            text_auto=".2f",
            title="Associação individual com o INDE",
        ),
        width="stretch",
    )
    base = df.dropna(subset=["IDA", "IEG", "IPS", "IPP", "INDE"]).copy()
    if not base.empty:
        base["MEDIA_4_INDICADORES"] = base[fatores].mean(axis=1)
        col2.plotly_chart(
            px.scatter(
                base,
                x="MEDIA_4_INDICADORES",
                y="INDE",
                color="ANO_REFERENCIA",
                opacity=0.4,
                title="Combinação média de IDA, IEG, IPS e IPP Ã— INDE",
                labels={"MEDIA_4_INDICADORES": "Média dos quatro indicadores"},
            ),
            width="stretch",
        )
    else:
        col2.info("Nãoo há¡ observações completas dos quatro indicadores.")


def main() -> None:
    st.title("Datathon Passos Mágicos")
    st.write(
        "Análise interativa dos indicadores educacionais de 2022 a 2024. "
        "Use os filtros laterais para explorar segmentos especÃ­ficos."
    )
    if not DATA_PATH.exists():
        st.error(f"Base não encontrada: {DATA_PATH}")
        st.stop()

    df_completo = carregar_dados(DATA_PATH)
    df = aplicar_filtros(df_completo)
    if df.empty:
        st.warning("Nenhum registro corresponde Ã  combinações atual de filtros.")
        st.stop()

    abas = st.tabs(
        [
            "Visão geral",
            "1 Â· IAN",
            "2 Â· IDA",
            "3 Â· IEG",
            "4 Â· IAA",
            "5 Â· IPS",
            "6 Â· IPP",
            "7 Â· IPV",
            "8 Â· INDE",
            "Modelo de risco",
        ]
    )
    with abas[0]:
        visao_geral(df)
    with abas[1]:
        questao_1(df)
    with abas[2]:
        questao_2(df)
    with abas[3]:
        questao_3(df)
    with abas[4]:
        questao_4(df)
    with abas[5]:
        questao_5(df_completo, set(df["RA"]))
    with abas[6]:
        questao_6(df)
    with abas[7]:
        questao_7(df)
    with abas[8]:
        questao_8(df)
    with abas[9]:
        st.subheader("9. Previsão de risco de defasagem")
        st.info(
            "A interface está reservada para o modelo. Ela será ativada depois da "
            "validação temporal, avaliações e análise de vazamento de dados."
        )
        st.write(
            "O alvo proposto Ã© a ocorrência de defasagem negativa no ano seguinte. "
            "O conjunto longitudinal disponí­vel possui transições 2022 á ’2023 e "
            "2023 á ’2024."
        )


if __name__ == "__main__":
    main()

