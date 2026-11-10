"""E0: importância na validação completa, sem treino ou acesso ao teste final.

Executar: python -B -m olist_modeling.importance --root .
Resultados novos ficam em resultados/tabelas/e0 e nunca sobrescrevem arquivos.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import fsum, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from olist_features.core import FEATURE_COLUMNS, TARGET
from .protocol import SEED, TEST_START, VALIDATION_START, prepare_X

CONFIG = {
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 3,
    "random_state": SEED,
    "n_jobs": 1,
    "max_samples": 1.0,
}


def validation_importance(model, X, y):
    """Modelo já ajustado; todas as avaliações usam a mesma população."""
    return permutation_importance(model, X, y, **CONFIG)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_metrics(y, prediction):
    """Fórmulas independentes, sem chamar os geradores históricos."""
    actual, predicted = list(map(float, y)), list(map(float, prediction))
    errors = [a - p for a, p in zip(actual, predicted, strict=True)]
    mean = fsum(actual) / len(actual)
    sse = fsum(e * e for e in errors)
    sst = fsum((a - mean) ** 2 for a in actual)
    return {
        "MAE_days": fsum(abs(e) for e in errors) / len(actual),
        "RMSE_days": sqrt(sse / len(actual)),
        "R2": 1 - sse / sst,
    }


def load_validation(root):
    """Lê somente IDs da validação no Parquet compartilhado; não carrega a coorte."""
    root = Path(root)
    tables = root / "resultados/tabelas/task02"
    selection = json.loads((tables / "selection_frozen.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "dados/tratados/feature_contract.json").read_text(encoding="utf-8"))
    if selection["selected_model"] != "Ridge" or contract["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("E0 exige a Ridge selecionada e o contrato original de 30 features")
    predictions_path = tables / "validation_predictions.parquet"
    if file_hash(predictions_path) != selection["validation_predictions_sha256"]:
        raise ValueError("As previsões de validação não correspondem à seleção selada")
    predictions = pd.read_parquet(predictions_path)
    ids = predictions.order_id.tolist()
    if len(ids) != 12675 or not predictions.order_id.is_unique:
        raise ValueError("E0 exige os 12.675 IDs únicos da validação histórica")
    filters = [("order_id", "in", ids)]
    data = pd.read_parquet(
        root / "dados/tratados/olist_orders_features.parquet",
        columns=["order_id", *FEATURE_COLUMNS, TARGET], filters=filters,
    ).set_index("order_id").loc[ids]
    dates = pd.read_parquet(
        root / "resultados/tabelas/order_selection_audit.parquet",
        columns=["order_id", "purchase_timestamp", "delivery_timestamp"], filters=filters,
    ).set_index("order_id").loc[ids]
    valid = (dates.purchase_timestamp.ge(VALIDATION_START)
             & dates.purchase_timestamp.lt(TEST_START)
             & dates.delivery_timestamp.lt(TEST_START))
    if not valid.all() or dates.reset_index().sort_values(["purchase_timestamp", "order_id"]).order_id.tolist() != ids:
        raise ValueError("IDs/ordem não correspondem à validação cronológica com rótulos maduros")
    np.testing.assert_allclose(data[TARGET], predictions.actual_days, rtol=0, atol=0)
    model = joblib.load(root / "resultados/modelos/task02/validation_Ridge.joblib")
    if model.named_steps["model"].estimator_.alpha != selection["best_params"]["model__estimator__alpha"]:
        raise ValueError("O modelo salvo não corresponde ao parâmetro selecionado")
    return model, prepare_X(data.reset_index()), predictions


def calculate_e0(root):
    """Diagnóstico puro em memória; não escreve nem chama fit/select/evaluate/report."""
    model, X, predictions = load_validation(root)
    model_before, data_before = joblib.hash(model), joblib.hash(X)
    with threadpool_limits(limits=2):
        current = model.predict(X)
        delta = float(np.max(np.abs(current - predictions.Ridge.to_numpy())))
        np.testing.assert_allclose(current, predictions.Ridge, rtol=0, atol=1e-9)
        scores = validation_metrics(predictions.actual_days, current)
        expected = pd.read_csv(Path(root) / "resultados/tabelas/task02/validation_metrics.csv").set_index("model").loc["Ridge"]
        for name, value in scores.items():
            np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-10)
        result = validation_importance(model, X, predictions.actual_days.to_numpy())
    if model_before != joblib.hash(model) or data_before != joblib.hash(X):
        raise AssertionError("O cálculo de importância alterou modelo ou entradas")
    table = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "increase_MAE_days_mean": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("increase_MAE_days_mean", ascending=False, kind="stable")
    historical = pd.read_csv(Path(root) / "resultados/tabelas/task02/importance_validation_permutation.csv")
    comparison = table.merge(historical, on="feature", validate="one_to_one", suffixes=("_e0", "_historical"))
    comparison["difference_e0_minus_historical"] = comparison.increase_MAE_days_mean_e0 - comparison.increase_MAE_days_mean_historical
    names = model.named_steps["preprocess"].get_feature_names_out()
    removed = names[~model.named_steps["select"].get_support()].tolist()
    controls = ["items_missing_weight_count", "items_missing_dimensions_count",
                "items_missing_price_count", "items_missing_freight_count"]
    for name in controls:
        values = result.importances[FEATURE_COLUMNS.index(name)]
        np.testing.assert_allclose(values, 0, rtol=0, atol=1e-12)
    root = Path(root)
    input_paths = [
        "resultados/modelos/task02/validation_Ridge.joblib",
        "dados/tratados/feature_contract.json", "dados/tratados/olist_orders_features.parquet",
        "resultados/tabelas/order_selection_audit.parquet",
        "resultados/tabelas/task02/selection_frozen.json",
        "resultados/tabelas/task02/validation_predictions.parquet",
        "resultados/tabelas/task02/validation_metrics.csv",
        "resultados/tabelas/task02/importance_validation_permutation.csv",
    ]
    metadata = {
        "correction": "E0: referência e permutações nos mesmos 12.675 pedidos; execução histórica preservada",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "validation_Ridge.joblib", "sklearn_version": sklearn.__version__,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "configuration": CONFIG.copy(), "rows": len(X), "feature_columns": FEATURE_COLUMNS,
        "ordered_order_ids_sha256": hashlib.sha256("\n".join(predictions.order_id).encode()).hexdigest(),
        "input_sha256": {name: file_hash(root / name) for name in input_paths},
        "code_sha256": {"src/olist_modeling/importance.py": file_hash(Path(__file__))},
        "validation_metrics": scores, "max_prediction_difference_days": delta,
        "removed_transformed_features": removed,
        "negative_controls": {name: result.importances[FEATURE_COLUMNS.index(name)].tolist() for name in controls},
        "fit_called": False, "final_test_used": False,
        "limitations": "Três repetições; desvio entre permutações não é intervalo de confiança; correlação limita interpretação; não é ablação.",
    }
    return table, comparison, metadata, predictions.order_id.tolist()


def save_e0(root, table, comparison, metadata, ids, *, figure=False):
    """Saídas exclusivas de E0, com criação exclusiva para evitar sobrescrita."""
    root = Path(root)
    folder = root / "resultados/tabelas/e0"
    names = ["importance_validation_permutation.csv", "comparison_historical.csv", "metadata.json", "validation_order_ids.csv"]
    paths = [folder / name for name in names]
    figure_path = root / "resultados/graficos/e0/importance_validation.png"
    if any(p.exists() for p in paths) or (figure and figure_path.exists()):
        raise FileExistsError("Resultados E0 já existem; nenhuma saída será sobrescrita")
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths[0], index=False, mode="x", float_format="%.17g", lineterminator="\n")
    comparison.to_csv(paths[1], index=False, mode="x", float_format="%.17g", lineterminator="\n")
    pd.DataFrame({"order_id": ids}).to_csv(paths[3], index=False, mode="x", lineterminator="\n")
    if figure:
        # Importação tardia: não usa o gerador de relatórios que consulta teste.
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(folder / "matplotlib-cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shown = table.head(12).sort_values("increase_MAE_days_mean")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(shown.feature, shown.increase_MAE_days_mean, xerr=shown["std"], capsize=3, color="#256b87")
        ax.axvline(0, color="#555", linewidth=.7)
        ax.set(xlabel="Aumento do MAE após permutação (dias)",
               title=f"E0 — Ridge: validação completa ({len(ids):,} pedidos)\n3 repetições; barras: desvio padrão; sem retreinamento")
        fig.tight_layout()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        with figure_path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, bbox_inches="tight", metadata={"Software": "Olist E0"})
        plt.close(fig)
    metadata = dict(metadata)
    metadata["output_sha256"] = {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in [paths[0], paths[1], paths[3], *([figure_path] if figure else [])]}
    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "resultados/tabelas/e0/metadata.json").exists():
        raise FileExistsError("E0 já calculado; recálculo não sobrescreve resultados")
    table, comparison, metadata, ids = calculate_e0(root)
    save_e0(root, table, comparison, metadata, ids, figure=args.figure)
    print(table.to_string(index=False))
    print(f"E0: {len(ids)} pedidos; diferença máxima nas previsões: {metadata['max_prediction_difference_days']:.3g} dia; nenhuma chamada de fit.")


if __name__ == "__main__":
    main()
