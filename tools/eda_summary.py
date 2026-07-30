from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("BASE_PEDE_CONSOLIDADA_TRATADA_V2.xlsx")
OUTPUT = Path("tmp/eda_summary.json")


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def counts(series: pd.Series) -> dict:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def main() -> None:
    df = pd.read_excel(INPUT, sheet_name="BASE_CONSOLIDADA")
    years = sorted(df["ANO_REFERENCIA"].unique().tolist())
    indicators = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
    correlations = df[indicators].corr(method="spearman", min_periods=30).round(3)

    yearly = {}
    for year in years:
        part = df[df["ANO_REFERENCIA"] == year]
        yearly[str(year)] = {
            "rows": int(len(part)),
            "students": int(part["RA"].nunique()),
            "indicator_mean": {
                col: clean_value(part[col].mean()) for col in indicators
            },
            "indicator_coverage": {
                col: round(float(part[col].notna().mean()), 4) for col in indicators
            },
            "defasagem": counts(part["DEFASAGEM"]),
            "qualidade": counts(part["QUALIDADE_REGISTRO"]),
            "pedra": counts(part["PEDRA"]),
        }

    ordered = df.sort_values(["RA", "ANO_REFERENCIA"])
    ordered["NEXT_DEFASAGEM"] = ordered.groupby("RA")["DEFASAGEM"].shift(-1)
    ordered["NEXT_YEAR"] = ordered.groupby("RA")["ANO_REFERENCIA"].shift(-1)
    transitions = ordered[ordered["NEXT_YEAR"] == ordered["ANO_REFERENCIA"] + 1].copy()
    transitions["RISCO_PROXIMO_ANO"] = (transitions["NEXT_DEFASAGEM"] < 0).astype(int)

    missing = df.isna().mean().sort_values(ascending=False)
    report = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "students": int(df["RA"].nunique()),
        "years": counts(df["ANO_REFERENCIA"]),
        "students_by_observation_count": counts(df.groupby("RA").size()),
        "quality": counts(df["QUALIDADE_REGISTRO"]),
        "divergence_rows": int(df["DIVERGENCIAS_IDENTIFICADAS"].notna().sum()),
        "yearly": yearly,
        "spearman_correlations": correlations.to_dict(),
        "prediction_frame": {
            "rows": int(len(transitions)),
            "students": int(transitions["RA"].nunique()),
            "target_positive": int(transitions["RISCO_PROXIMO_ANO"].sum()),
            "target_rate": round(float(transitions["RISCO_PROXIMO_ANO"].mean()), 4),
            "origins": counts(transitions["ANO_REFERENCIA"]),
        },
        "columns_over_50pct_missing": {
            col: round(float(rate), 4) for col, rate in missing[missing > 0.5].items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
