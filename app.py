"""Dashboard analítico do Datathon Passos Mágicos.

Execução:
    .venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "BASE_PEDE_CONSOLIDADA_TRATADA_V2.xlsx"
MODEL_PATH = ROOT / "outputs" / "modelo" / "modelo_risco.joblib"
MODEL_META_PATH = ROOT / "outputs" / "modelo" / "metadados_modelo.json"
MODEL_IMPORTANCE_PATH = ROOT / "outputs" / "modelo" / "importancia_variaveis.csv"
INDICADORES = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
DICIONARIO_INDICES = {
    "IAN": (
        "Indicador de Adequação de Nível",
        "Representa a adequação do aluno à fase esperada e ajuda a identificar situações de defasagem.",
    ),
    "IDA": (
        "Indicador de Desempenho Acadêmico",
        "Sintetiza o desempenho acadêmico a partir das notas disponíveis nas disciplinas.",
    ),
    "IEG": (
        "Indicador de Engajamento",
        "Expressa o grau de participação e envolvimento do aluno nas atividades do programa.",
    ),
    "IAA": (
        "Indicador de Autoavaliação",
        "Registra a percepção do aluno sobre si mesmo e sobre o próprio desenvolvimento.",
    ),
    "IPS": (
        "Indicador Psicossocial",
        "Reúne aspectos sociais, emocionais e psicológicos observados no acompanhamento do aluno.",
    ),
    "IPP": (
        "Indicador Psicopedagógico",
        "Resume a avaliação psicopedagógica relacionada à aprendizagem e às necessidades de apoio.",
    ),
    "IPV": (
        "Indicador de Ponto de Virada",
        "Sinaliza mudanças acadêmicas, emocionais ou de engajamento associadas à transformação da trajetória.",
    ),
    "INDE": (
        "Índice de Desenvolvimento Educacional",
        "É a nota global do aluno, composta pela combinação ponderada dos diferentes indicadores.",
    ),
}
CORES_DEFASAGEM = {
    "Severa (≤ -2)": "#DC2626",
    "Moderada (-1)": "#F59E0B",
    "Sem defasagem (≥ 0)": "#059669",
}


st.set_page_config(
    page_title="Passos Mágicos | Datathon",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Carregando a base consolidada...")
def carregar_dados(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="BASE_CONSOLIDADA")
    return preparar_dados(df)


@st.cache_data(show_spinner="Carregando a base enviada...")
def carregar_dados_enviados(conteudo: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(conteudo), sheet_name="BASE_CONSOLIDADA")
    return preparar_dados(df)


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df["ANO_REFERENCIA"] = df["ANO_REFERENCIA"].astype(int)
    df["NIVEL_DEFASAGEM"] = pd.cut(
        df["DEFASAGEM"],
        bins=[-np.inf, -2, -1, np.inf],
        labels=["Severa (≤ -2)", "Moderada (-1)", "Sem defasagem (≥ 0)"],
    )
    return df


@st.cache_resource(show_spinner="Carregando o modelo preditivo...")
def carregar_modelo():
    modelo = joblib.load(MODEL_PATH)
    metadados = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    return modelo, metadados


def formatar_numero(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    with st.sidebar.expander("Dicionário dos índices"):
        st.caption(
            "Os indicadores são apresentados, em geral, numa escala de 0 a 10. "
            "Valores mais altos costumam representar uma condição mais favorável, "
            "mas devem ser interpretados em conjunto e conforme o ano."
        )
        for sigla, (nome, descricao) in DICIONARIO_INDICES.items():
            st.markdown(f"**{sigla} — {nome}**")
            st.write(descricao)

    anos = sorted(df["ANO_REFERENCIA"].dropna().unique())
    anos_escolhidos = st.sidebar.multiselect("Ano", anos, default=anos)

    fases = sorted(df["FASE_NUMERICA"].dropna().unique())
    fases_escolhidas = st.sidebar.multiselect("Fase", fases, default=fases)

    generos = sorted(df["GENERO"].dropna().unique())
    generos_escolhidos = st.sidebar.multiselect("Gênero", generos, default=generos)

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


def grafico_sem_dados(mensagem: str = "Não há dados para os filtros selecionados.") -> None:
    st.info(mensagem)


def visao_geral(df: pd.DataFrame) -> None:
    st.subheader("Visão geral")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", formatar_numero(len(df)))
    col2.metric("Alunos únicos", formatar_numero(df["RA"].nunique()))
    col3.metric("INDE médio", f"{df['INDE'].mean():.2f}" if df["INDE"].notna().any() else "—")
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
        "Definição operacional usada: moderada = -1; severa -2."
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
        labels={"ANO_REFERENCIA": "Ano", "NIVEL_DEFASAGEM": "Nível"},
        title="Perfil de defasagem por ano",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


def questao_2(df: pd.DataFrame) -> None:
    st.subheader("2. Desempenho acadêmico — IDA")
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
        (col1, "IDA", "IEG × desempenho acadêmico"),
        (col2, "IPV", "IEG × ponto de virada"),
    ]:
        base = df.dropna(subset=["IEG", eixo])
        rho = base[["IEG", eixo]].corr(method="spearman").iloc[0, 1]
        fig = px.scatter(
            base,
            x="IEG",
            y=eixo,
            color="ANO_REFERENCIA",
            opacity=0.35,
            title=f"{titulo} · Spearman = {rho:.2f}",
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
            title="Diferença entre percepção e referência observada",
            labels={"DIFERENCA_IAA": "IAA − média(IDA, IEG)", "ANO_REFERENCIA": "Ano"},
        ),
        width="stretch",
    )
    st.caption(
        "Valores positivos indicam autoavaliação acima da média combinada de "
        "desempenho e engajamento; não representam, isoladamente, erro de percepção."
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
    st.subheader("5. Aspectos psicossociais — IPS antes de quedas")
    transicoes = construir_transicoes(df_completo)
    transicoes = transicoes[transicoes["RA"].isin(ids_filtrados)]
    if transicoes.empty:
        grafico_sem_dados("Não há transições anuais para os alunos selecionados.")
        return
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.scatter(
            transicoes,
            x="IPS",
            y="DELTA_IDA",
            color="ANO_REFERENCIA",
            opacity=0.4,
            title="IPS atual × variação do IDA no ano seguinte",
            labels={"DELTA_IDA": "IDA seguinte − IDA atual"},
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
            title="IPS atual × variação do IEG no ano seguinte",
            labels={"DELTA_IEG": "IEG seguinte − IEG atual"},
        ),
        width="stretch",
    )
    st.caption(
        "Esta visão temporal mais apropriada que uma correlação do mesmo ano, "
        "mas ainda mostra associação, não causalidade."
    )


def questao_6(df: pd.DataFrame) -> None:
    st.subheader("6. Aspectos psicopedagógicos — IPP e adequação")
    base = df.dropna(subset=["IPP", "DEFASAGEM"]).copy()
    if base.empty:
        grafico_sem_dados("O IPP não está disponível para os filtros selecionados.")
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
            title="Distribuição do IPP por nível de defasagem",
            labels={"NIVEL_DEFASAGEM": "Nível de defasagem"},
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
            title="IPP médio por nível e ano",
            labels={"ANO_REFERENCIA": "Ano", "NIVEL_DEFASAGEM": "Nível"},
        ),
        width="stretch",
    )
    st.info("O IPP não possui observações em 2022; comparações temporais começam em 2023.")


def questao_7(df: pd.DataFrame) -> None:
    st.subheader("7. Ponto de virada — fatores associados ao IPV")
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
    st.subheader("8. Multidimensionalidade — indicadores associados ao INDE")
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
                title="Combinação média de IDA, IEG, IPS e IPP × INDE",
                labels={"MEDIA_4_INDICADORES": "Média dos quatro indicadores"},
            ),
            width="stretch",
        )
    else:
        col2.info("Não há observações completas dos quatro indicadores.")


def modelo_risco(df: pd.DataFrame) -> None:
    st.subheader("9. Previsão de risco de defasagem")
    st.write(
        "Informe os dados atuais do aluno para estimar a probabilidade de "
        "defasagem negativa no ano seguinte."
    )
    if not MODEL_PATH.exists() or not MODEL_META_PATH.exists():
        st.warning(
            "O modelo treinado não foi encontrado. Execute "
            "`python src/modelo_preditivo.py` antes de utilizar esta aba."
        )
        return

    modelo, metadados = carregar_modelo()
    generos = sorted(df["GENERO"].dropna().astype(str).unique())
    instituicoes = sorted(df["INSTITUICAO_ENSINO"].dropna().astype(str).unique())
    qualidades = sorted(df["QUALIDADE_REGISTRO"].dropna().astype(str).unique())

    with st.form("formulario_risco"):
        st.markdown("#### Dados do aluno")
        col1, col2, col3 = st.columns(3)
        idade = col1.number_input("Idade", min_value=5, max_value=30, value=15)
        anos_pm = col2.number_input(
            "Anos na Passos Mágicos", min_value=0, max_value=15, value=3
        )
        fase = col3.number_input("Fase numérica", min_value=0, max_value=8, value=4)
        genero = col1.selectbox("Gênero", generos)
        instituicao = col2.selectbox("Instituição de ensino", instituicoes)
        qualidade = col3.selectbox("Qualidade do registro", qualidades)

        st.markdown("#### Indicadores atuais")
        c1, c2, c3, c4 = st.columns(4)
        defasagem = c1.number_input(
            "Defasagem atual", min_value=-5, max_value=3, value=0, step=1
        )
        ian = c2.slider("IAN", 0.0, 10.0, 10.0, 0.1)
        ida = c3.slider("IDA", 0.0, 10.0, 6.0, 0.1)
        ieg = c4.slider("IEG", 0.0, 10.0, 7.0, 0.1)
        iaa = c1.slider("IAA", 0.0, 10.0, 7.0, 0.1)
        ips = c2.slider("IPS", 0.0, 10.0, 7.0, 0.1)
        ipv = c3.slider("IPV", 0.0, 10.0, 7.0, 0.1)
        inde = c4.slider("INDE", 0.0, 10.0, 7.0, 0.1)
        calcular = st.form_submit_button("Calcular probabilidade", type="primary")

    if not calcular:
        st.caption("Preencha o formulário e clique em “Calcular probabilidade”.")
        return

    entrada = pd.DataFrame(
        [
            {
                "IDADE": idade,
                "ANOS_NA_PM": anos_pm,
                "FASE_NUMERICA": fase,
                "DEFASAGEM": defasagem,
                "IAN": ian,
                "IDA": ida,
                "IEG": ieg,
                "IAA": iaa,
                "IPS": ips,
                "IPV": ipv,
                "INDE": inde,
                "GENERO": genero,
                "INSTITUICAO_ENSINO": instituicao,
                "QUALIDADE_REGISTRO": qualidade,
            }
        ]
    )
    probabilidade = float(modelo.predict_proba(entrada)[0, 1])
    limiar = float(metadados["limiar"])
    classificacao = "Em risco" if probabilidade >= limiar else "Sem risco"
    if probabilidade >= limiar:
        faixa = "Alto"
    elif probabilidade >= limiar * 0.65:
        faixa = "Moderado"
    else:
        faixa = "Baixo"

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidade estimada", f"{probabilidade:.1%}")
    m2.metric("Classificação", classificacao)
    m3.metric("Nível de alerta", faixa)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probabilidade * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Risco estimado para o próximo ano"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, limiar * 65], "color": "#DCFCE7"},
                    {"range": [limiar * 65, limiar * 100], "color": "#FEF3C7"},
                    {"range": [limiar * 100, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#991B1B", "width": 4},
                    "value": limiar * 100,
                },
            },
        )
    )
    gauge.update_layout(height=330, margin=dict(l=30, r=30, t=70, b=20))
    st.plotly_chart(gauge, width="stretch")

    st.caption(
        f"O limiar operacional do modelo é {limiar:.1%}. A previsão serve como "
        "apoio à priorização e não substitui a avaliação pedagógica e psicossocial."
    )

    metricas = metadados["metricas_teste"]
    with st.expander("Desempenho e fatores gerais do modelo"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metricas['roc_auc']:.3f}")
        c2.metric("PR-AUC", f"{metricas['pr_auc']:.3f}")
        c3.metric("Recall", f"{metricas['recall_risco']:.1%}")
        c4.metric("Precisão", f"{metricas['precisao_risco']:.1%}")
        if MODEL_IMPORTANCE_PATH.exists():
            importancia = pd.read_csv(MODEL_IMPORTANCE_PATH).head(8)
            st.plotly_chart(
                px.bar(
                    importancia.sort_values("importancia"),
                    x="importancia",
                    y="variavel",
                    orientation="h",
                    title="Fatores gerais mais influentes no teste temporal",
                    labels={"importancia": "Importância por permutação", "variavel": "Variável"},
                ),
                width="stretch",
            )


def main() -> None:
    st.title("Datathon Passos Mágicos")
    st.write(
        "Análise interativa dos indicadores educacionais de 2022 a 2024. "
        "Use os filtros laterais para explorar segmentos específicos."
    )
    if DATA_PATH.exists():
        df_completo = carregar_dados(DATA_PATH)
    else:
        st.info(
            "Para preservar a privacidade, a base não fica armazenada no site. "
            "Carregue o arquivo Excel autorizado para iniciar a análise."
        )
        arquivo = st.file_uploader(
            "Base consolidada (.xlsx)",
            type=["xlsx"],
            help="O arquivo é processado somente durante esta sessão do Streamlit.",
        )
        if arquivo is None:
            st.stop()
        df_completo = carregar_dados_enviados(arquivo.getvalue())
    df = aplicar_filtros(df_completo)
    if df.empty:
        st.warning("Nenhum registro corresponde à combinação atual de filtros.")
        st.stop()

    abas = st.tabs(
        [
            "Visão geral",
            "1 · IAN",
            "2 · IDA",
            "3 · IEG",
            "4 · IAA",
            "5 · IPS",
            "6 · IPP",
            "7 · IPV",
            "8 · INDE",
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
        modelo_risco(df_completo)


if __name__ == "__main__":
    main()

