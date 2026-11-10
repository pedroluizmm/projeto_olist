"""EDA somente do treino elegível; figuras estáticas exportáveis."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from olist_features.core import FEATURE_COLUMNS, TARGET
from olist_features.pipeline import write_csv, write_json


def plotting(root: Path):
    # Manter cache e configuração dentro do projeto.
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".venv/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 170})
    return plt


def save_plot(plt, fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", metadata={"Software": "Olist TASK 02"})
    plt.close(fig)


def categorical_profile(train: pd.DataFrame):
    from .protocol import CATEGORICAL
    rows = []
    for feature in CATEGORICAL:
        values = train[feature].astype("string").fillna("__MISSING__")
        for category, count in values.value_counts().items():
            target = train.loc[values.eq(category), TARGET]
            rows.append({"feature": feature, "category": category, "orders": int(count),
                         "orders_pct": count / len(train) * 100, "mean_target_days": target.mean(),
                         "median_target_days": target.median()})
    return pd.DataFrame(rows)


def run_eda(root: Path, train: pd.DataFrame, raw: dict):
    from .protocol import NUMERIC
    tables = root / "resultados/tabelas/task02"
    figures = root / "resultados/graficos/task02"
    plt = plotting(root)
    numeric = train[[*NUMERIC, TARGET]].astype(float)
    describe = numeric.describe(percentiles=[.01, .25, .5, .75, .95, .99]).T.reset_index(names="feature")
    write_csv(describe, tables / "eda_train_descriptive.csv")
    missing = train[FEATURE_COLUMNS].isna().sum().rename_axis("feature").reset_index(name="missing")
    missing["missing_pct"] = missing.missing / len(train) * 100
    write_csv(missing, tables / "eda_train_missing.csv")
    write_csv(categorical_profile(train), tables / "eda_train_categorical.csv")
    outliers = []
    for col in numeric:
        values = numeric[col]
        q1, q3 = values.quantile([.25, .75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = values.lt(low) | values.gt(high)
        outliers.append({"feature": col, "iqr_low": low, "iqr_high": high, "flagged": int(mask.sum()),
                         "flagged_pct": mask.mean() * 100, "action": "preservado; diagnóstico, sem corte"})
    write_csv(pd.DataFrame(outliers), tables / "eda_train_outliers.csv")
    spearman, pearson = numeric.corr(method="spearman"), numeric.corr(method="pearson")
    write_csv(spearman.rename_axis("feature").reset_index(), tables / "eda_train_spearman.csv")
    write_csv(pearson.rename_axis("feature").reset_index(), tables / "eda_train_pearson.csv")
    relations = pd.DataFrame({"feature": numeric.columns, "spearman": spearman[TARGET].values,
                              "pearson": pearson[TARGET].values}).query("feature != @TARGET")
    relations = relations.assign(abs_spearman=relations.spearman.abs()).sort_values("abs_spearman", ascending=False)
    write_csv(relations, tables / "eda_train_target_relations.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, label, log in zip(axes.flat, [TARGET, "order_value_total", "distance_km_mean", "weight_g_total"],
                                  ["Tempo real de entrega (dias)", "Valor do pedido (R$; eixo log)",
                                   "Distância média por item (km)", "Peso total (g; eixo log)"], [False, True, False, True]):
        vals = train[col].dropna().astype(float)
        bins = np.geomspace(vals[vals > 0].min(), vals.max(), 45) if log else 50
        ax.hist(vals, bins=bins, color="#256b87", edgecolor="white", linewidth=.3)
        if log:
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="Pedidos", title=f"n={len(vals):,}; ausentes={train[col].isna().sum()}")
    fig.suptitle("Distribuições no treino — extremos preservados", fontsize=15)
    fig.tight_layout()
    save_plot(plt, fig, figures / "01_distributions_train.png")

    varying = numeric.columns[numeric.nunique() > 1]
    corr = spearman.loc[varying, varying]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.set_title("Correlação de Spearman — treino; constantes omitidas", pad=16)
    fig.colorbar(im, ax=ax, shrink=.75, label="Correlação (sem unidade)")
    save_plot(plt, fig, figures / "02_correlations_train.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, label in zip(axes[:2], ["distance_km_mean", "freight_value_total"], ["Distância média (km)", "Frete total (R$)"]):
        valid = train[[col, TARGET]].dropna()
        hb = ax.hexbin(valid[col], valid[TARGET], gridsize=40, bins="log", mincnt=1, cmap="viridis")
        ax.set(xlabel=label, ylabel="Tempo real (dias)")
        fig.colorbar(hb, ax=ax, label="Pedidos (escala log)")
    same = train.same_state_items_fraction
    parts = [train.loc[same.eq(1), TARGET], train.loc[same.eq(0), TARGET], train.loc[same.gt(0) & same.lt(1), TARGET]]
    axes[2].boxplot(parts, tick_labels=["Mesma UF", "Outra UF", "Misto"], showfliers=False)
    axes[2].set(ylabel="Tempo real (dias)", title="Caixas sem pontos extremos*  ")
    fig.suptitle("Relações no treino — associação não implica causalidade", fontsize=14)
    fig.text(.5, -.02, "*Extremos permanecem nas estatísticas e no treinamento.", ha="center")
    fig.tight_layout()
    save_plot(plt, fig, figures / "03_feature_target_train.png")

    states = train.groupby("customer_state").agg(orders=(TARGET, "size"), mean_days=(TARGET, "mean"), median_days=(TARGET, "median"),
                                                  distance_mean_km=("distance_km_mean", "mean")).sort_values("median_days")
    write_csv(states.reset_index(), tables / "eda_train_by_state.csv")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(states.index, states.median_days, color="#256b87")
    for y, (_, row) in enumerate(states.iterrows()):
        ax.text(row.median_days + .2, y, f"n={int(row.orders):,}", va="center", fontsize=8)
    ax.set(xlabel="Mediana do tempo real (dias)", ylabel="UF do comprador", title="Entregas por UF — somente treino")
    ax.set_xlim(0, states.median_days.max() * 1.35)
    save_plot(plt, fig, figures / "04_state_train.png")

    buyer_orders = raw["orders"][["order_id", "customer_id"]].merge(
        raw["customers"][["customer_id", "customer_zip_code_prefix"]], on="customer_id", validate="many_to_one")
    locations = train[["order_id", TARGET]].merge(buyer_orders, on="order_id", validate="one_to_one")
    grouped = locations.groupby("customer_zip_code_prefix").agg(orders=(TARGET, "size"), median_days=(TARGET, "median")).reset_index()
    centers = pd.read_parquet(root / "resultados/tabelas/geolocation_centroids.parquet")
    grouped = grouped.merge(centers, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", validate="one_to_one")
    grouped = grouped.loc[grouped.orders.ge(20)]
    write_csv(grouped[["customer_zip_code_prefix", "orders", "median_days", "lat", "lon"]], tables / "eda_train_geographic_ceps.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    points = ax.scatter(grouped.lon, grouped.lat, c=grouped.median_days, s=np.sqrt(grouped.orders) * 4,
                        cmap="viridis", alpha=.8, edgecolor="white", linewidth=.25)
    fig.colorbar(points, ax=ax, label="Mediana de entrega (dias)")
    ax.set(xlabel="Longitude (graus)", ylabel="Latitude (graus)", title="Centroides de CEP do comprador — treino\nCEPs com ≥20 pedidos; área varia com √n; sem limites cartográficos")
    ax.set_aspect(1 / np.cos(np.radians(-15)))
    save_plot(plt, fig, figures / "05_geographic_train.png")

    monthly = train.assign(month=train.purchase_timestamp.dt.strftime("%Y-%m")).groupby("month").agg(
        orders=(TARGET, "size"), median_days=(TARGET, "median"), q25=(TARGET, lambda s: s.quantile(.25)), q75=(TARGET, lambda s: s.quantile(.75)))
    # Meses sem pedidos continuam no eixo como lacunas, sem interpolar resultados.
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M").astype(str)
    monthly = monthly.reindex(months).rename_axis("month")
    monthly["orders"] = monthly.orders.fillna(0).astype(int)
    write_csv(monthly.reset_index(), tables / "eda_train_by_month.csv")
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(12, 4.7))
    ax.plot(x, monthly.median_days, color="#256b87", marker="o", label="Mediana")
    ax.fill_between(x, monthly.q25, monthly.q75, alpha=.2, color="#256b87", label="Intervalo interquartil")
    ax.set_xticks(x, monthly.index, rotation=45, ha="right")
    ax.set(xlabel="Mês da compra", ylabel="Tempo real (dias)", title="Evolução temporal no treino\nÚltimo período sujeito à seleção por maturação do rótulo")
    ax2 = ax.twinx()
    ax2.plot(x, monthly.orders, color="#bb7840", ls="--", label="Pedidos")
    ax2.set_ylabel("Pedidos", color="#bb7840")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    save_plot(plt, fig, figures / "06_monthly_train.png")
    write_json({"scope": "somente treino com rótulo conhecido antes da validação", "rows": len(train),
                "purchase_min": str(train.purchase_timestamp.min()), "purchase_max": str(train.purchase_timestamp.max()),
                "test_used": False, "outliers_removed": 0, "geographic_min_orders": 20}, tables / "eda_scope.json")
