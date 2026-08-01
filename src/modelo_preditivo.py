"""Treina e avalia modelos de risco de defasagem no ano seguinte.

Execução na raiz do projeto:
    .venv\\Scripts\\python.exe src\\modelo_preditivo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "BASE_PEDE_CONSOLIDADA_TRATADA_V2.xlsx"
OUT = ROOT / "outputs" / "modelo"
FIG = OUT / "figuras"

NUMERICAS = [
    "IDADE",
    "ANOS_NA_PM",
    "FASE_NUMERICA",
    "DEFASAGEM",
    "IAN",
    "IDA",
    "IEG",
    "IAA",
    "IPS",
    "IPV",
    "INDE",
]
CATEGORICAS = ["GENERO", "INSTITUICAO_ENSINO", "QUALIDADE_REGISTRO"]
FEATURES = NUMERICAS + CATEGORICAS


def construir_transicoes(df: pd.DataFrame) -> pd.DataFrame:
    base = df.sort_values(["RA", "ANO_REFERENCIA"]).copy()
    base["PROX_ANO"] = base.groupby("RA")["ANO_REFERENCIA"].shift(-1)
    base["PROX_DEFASAGEM"] = base.groupby("RA")["DEFASAGEM"].shift(-1)
    base = base[base["PROX_ANO"] == base["ANO_REFERENCIA"] + 1].copy()
    base["RISCO_PROXIMO_ANO"] = (base["PROX_DEFASAGEM"] < 0).astype(int)
    return base


def criar_preprocessador() -> ColumnTransformer:
    numerico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numerico, NUMERICAS), ("cat", categorico, CATEGORICAS)],
        sparse_threshold=0,
    )


def escolher_limiar(y: pd.Series, probabilidades: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilidades)
    candidatos = np.where(recall[:-1] >= 0.80)[0]
    if len(candidatos) == 0:
        return 0.5
    melhor = candidatos[np.argmax(precision[:-1][candidatos])]
    return float(thresholds[melhor])


def calcular_metricas(y: pd.Series, prob: np.ndarray, limiar: float) -> dict:
    pred = (prob >= limiar).astype(int)
    return {
        "limiar": round(float(limiar), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "recall_risco": round(float(recall_score(y, pred)), 4),
        "precisao_risco": round(float(precision_score(y, pred, zero_division=0)), 4),
        "f1_risco": round(float(f1_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
    }


def salvar_curvas(resultados: dict, y_teste: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for nome, resultado in resultados.items():
        prob = resultado["prob_teste"]
        fpr, tpr, _ = roc_curve(y_teste, prob)
        precision, recall, _ = precision_recall_curve(y_teste, prob)
        axes[0].plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(y_teste, prob):.3f})")
        axes[1].plot(recall, precision, label=f"{nome} (AP={average_precision_score(y_teste, prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Curvas ROC — teste 2024", xlabel="Taxa de falso positivo", ylabel="Recall")
    axes[1].set(title="Curvas precisão-recall — teste 2024", xlabel="Recall", ylabel="Precisão")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "01_curvas_modelos.png", dpi=180, facecolor="white")
    plt.close(fig)


def salvar_diagnosticos(nome: str, resultado: dict, x_teste: pd.DataFrame, y_teste: pd.Series) -> None:
    modelo = resultado["modelo"]
    prob = resultado["prob_teste"]
    limiar = resultado["limiar"]
    pred = (prob >= limiar).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_teste,
        pred,
        display_labels=["Sem risco", "Em risco"],
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title(f"Matriz de confusão — {nome}")
    CalibrationDisplay.from_predictions(y_teste, prob, n_bins=8, strategy="quantile", ax=axes[1])
    axes[1].set_title("Calibração das probabilidades")
    fig.tight_layout()
    fig.savefig(FIG / "02_diagnostico_melhor_modelo.png", dpi=180, facecolor="white")
    plt.close(fig)

    importancia = permutation_importance(
        modelo,
        x_teste,
        y_teste,
        scoring="average_precision",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    tabela = pd.DataFrame(
        {"variavel": FEATURES, "importancia": importancia.importances_mean}
    ).sort_values("importancia", ascending=False)
    tabela.to_csv(OUT / "importancia_variaveis.csv", index=False, encoding="utf-8-sig")
    top = tabela.head(12).sort_values("importancia")
    plt.figure(figsize=(8, 6))
    plt.barh(top["variavel"], top["importancia"], color="#1D4ED8")
    plt.title(f"Importância por permutação — {nome}")
    plt.xlabel("Queda média na PR-AUC ao embaralhar a variável")
    plt.tight_layout()
    plt.savefig(FIG / "03_importancia_variaveis.png", dpi=180, facecolor="white")
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_PATH, sheet_name="BASE_CONSOLIDADA")
    transicoes = construir_transicoes(df)

    treino = transicoes[transicoes["ANO_REFERENCIA"] == 2022].copy()
    teste = transicoes[transicoes["ANO_REFERENCIA"] == 2023].copy()
    x_treino, y_treino = treino[FEATURES], treino["RISCO_PROXIMO_ANO"]
    x_teste, y_teste = teste[FEATURES], teste["RISCO_PROXIMO_ANO"]

    modelos = {
        "Regressão logística": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for nome, estimador in modelos.items():
        pipeline = Pipeline(
            [("preprocessador", criar_preprocessador()), ("modelo", estimador)]
        )
        prob_cv = cross_val_predict(
            pipeline,
            x_treino,
            y_treino,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        limiar = escolher_limiar(y_treino, prob_cv)
        pipeline.fit(x_treino, y_treino)
        prob_teste = pipeline.predict_proba(x_teste)[:, 1]
        resultados[nome] = {
            "modelo": pipeline,
            "limiar": limiar,
            "prob_teste": prob_teste,
            "metricas": calcular_metricas(y_teste, prob_teste, limiar),
        }

    metricas = pd.DataFrame(
        {nome: resultado["metricas"] for nome, resultado in resultados.items()}
    ).T.sort_values(["recall_risco", "pr_auc"], ascending=False)
    metricas.to_csv(OUT / "metricas_modelos.csv", encoding="utf-8-sig")
    melhor_nome = metricas.index[0]
    melhor = resultados[melhor_nome]

    salvar_curvas(resultados, y_teste)
    salvar_diagnosticos(melhor_nome, melhor, x_teste, y_teste)

    modelo_final = clone(melhor["modelo"])
    modelo_final.fit(transicoes[FEATURES], transicoes["RISCO_PROXIMO_ANO"])
    joblib.dump(modelo_final, OUT / "modelo_risco.joblib")
    metadados = {
        "modelo": melhor_nome,
        "limiar": round(float(melhor["limiar"]), 6),
        "features": FEATURES,
        "definicao_alvo": "DEFASAGEM no ano seguinte < 0",
        "treino": "2022 → 2023",
        "teste": "2023 → 2024",
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
        "metricas_teste": melhor["metricas"],
    }
    (OUT / "metadados_modelo.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMétricas no teste temporal 2023 -> 2024:\n")
    print(metricas.to_string())
    print(f"\nModelo selecionado: {melhor_nome}")
    print(f"Artefatos salvos em: {OUT}")


if __name__ == "__main__":
    main()
