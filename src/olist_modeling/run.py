"""CLI da TASK 02; etapas prepare/select/evaluate/report, ou all (execução inicial)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from threadpoolctl import threadpool_limits

from olist_features.audit import sha256, zip_integrity
from olist_features.core import FEATURE_COLUMNS, TARGET
from olist_features.pipeline import write_csv, write_json
from .eda import run_eda
from .estimators import PARAM_GRIDS, fitted_feature_names, make_pipeline
from .importance import validation_importance
from .protocol import (CATEGORICAL, NUMERIC, SEED, TEST_START, VALIDATION_START, audit_source,
                       load_cohort, prepare_X, split_cohort, split_diagnostics, verify_preservation)


def paths(root):
    tables, models = root / "resultados/tabelas/task02", root / "resultados/modelos/task02"
    for path in (tables, models, root / "resultados/graficos/task02"):
        path.mkdir(parents=True, exist_ok=True)
    return tables, models


def metrics(y, pred):
    return {"MAE_days": float(mean_absolute_error(y, pred)),
            "RMSE_days": float(root_mean_squared_error(y, pred)), "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan")}


def independent_metrics(y, pred):
    # Fórmulas independentes, com floats nativos; não chama métricas do Scikit-learn.
    from math import fsum, sqrt
    actual, predicted = list(map(float, y)), list(map(float, pred))
    count = len(actual)
    mean = fsum(actual) / count
    error = [a - b for a, b in zip(actual, predicted)]
    sse = fsum(e * e for e in error)
    sst = fsum((a - mean) ** 2 for a in actual)
    return {"MAE_days": fsum(abs(e) for e in error) / count, "RMSE_days": sqrt(sse / count),
            "R2": (1 - sse / sst if sst else (1.0 if not sse else 0.0)) if count > 1 else float("nan")}


def checked_metrics(y, pred):
    result, independent = metrics(y, pred), independent_metrics(y, pred)
    for name in result:
        np.testing.assert_allclose(result[name], independent[name], rtol=1e-11, atol=1e-11)
    return result


def prepare(root):
    tables, _ = paths(root)
    print("READ: preservação e reconstrução em memória da TASK 01.", flush=True)
    data = load_cohort(root)
    raw = audit_source(root, data)
    meta, summary = split_cohort(data)
    write_csv(meta, tables / "split_assignments.csv")
    write_csv(summary, tables / "split_summary.csv")
    diagnostics = split_diagnostics(root, data, meta, raw)
    write_json({"seed": SEED, "validation_start": str(VALIDATION_START), "test_start": str(TEST_START),
                "features": FEATURE_COLUMNS, "categorical": CATEGORICAL, "numeric": NUMERIC,
                "search": PARAM_GRIDS, "candidate_count": sum(len(ParameterGrid(v)) for v in PARAM_GRIDS.values()),
                "cv": "PredefinedSplit único, treino/validação cronológicos com rótulos disponíveis",
                "selection": "menor MAE de validação entre baseline e modelos; desempate RMSE/nome",
                "forest_fixed": {"n_estimators": 80, "max_depth": 16, "max_samples": .85, "n_jobs": 2},
                "search_n_jobs": 1, "prediction_floor_days": 0, "outlier_policy": "preservar extremos válidos",
                "test_role": "avaliação única, após selar seleção", "maturation_diagnostics": diagnostics}, tables / "protocol.json")
    print("APPLY: EDA exclusivamente no treino elegível.", flush=True)
    run_eda(root, data.loc[meta.is_train].copy(), raw)
    versions = {name: importlib.metadata.version(name) for name in ("scikit-learn", "scipy", "numpy", "pandas", "matplotlib", "joblib")}
    write_json({"passed": True, "rows": len(data), "versions": versions,
                "dataset_sha256": sha256(root / "dados/tratados/olist_orders_features.parquet"),
                "protocol_sha256": sha256(tables / "protocol.json"), "source_audit_sha256": sha256(tables / "source_audit.json")}, tables / "prepared.json")
    print(summary[["split", "rows"]].to_string(index=False), flush=True)


def load_prepared(root):
    tables, _ = paths(root)
    prepared = json.loads((tables / "prepared.json").read_text(encoding="utf-8"))
    assert prepared["dataset_sha256"] == sha256(root / "dados/tratados/olist_orders_features.parquet")
    assert prepared["protocol_sha256"] == sha256(tables / "protocol.json")
    data = load_cohort(root)
    meta, _ = split_cohort(data)
    recorded = pd.read_csv(tables / "split_assignments.csv", dtype={"order_id": "string"})
    assert recorded.order_id.tolist() == meta.order_id.tolist()
    for flag in ("is_train", "is_validation", "is_test", "is_final_train"):
        assert np.array_equal(recorded[flag], meta[flag])
    return data, meta


def select(root):
    tables, models = paths(root)
    if (tables / "selection_frozen.json").exists():
        raise RuntimeError("Seleção já selada; use verify para refits reprodutíveis. Não sobrescrever decisões após o teste.")
    data, meta = load_prepared(root)
    X_train, y_train = prepare_X(data.loc[meta.is_train]), data.loc[meta.is_train, TARGET]
    X_val, y_val = prepare_X(data.loc[meta.is_validation]), data.loc[meta.is_validation, TARGET]
    X_dev, y_dev = pd.concat([X_train, X_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True)
    fold = PredefinedSplit(np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)])
    rows, candidates, selected_params = [], [], {}
    validation = pd.DataFrame({"order_id": data.loc[meta.is_validation, "order_id"].to_numpy(), "actual_days": y_val.to_numpy()})
    start = time.perf_counter()
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    pred = baseline.predict(X_val)
    rows.append({"model": "Baseline", **checked_metrics(y_val, pred), "fit_seconds": time.perf_counter() - start,
                 "best_params": json.dumps({"strategy": "median"}), "candidates": 1})
    selected_params["Baseline"] = {"strategy": "median"}
    validation["Baseline"] = pred
    joblib.dump(baseline, models / "validation_Baseline.joblib", compress=3)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    for name in PARAM_GRIDS:
        print(f"TRAIN: {name}, {len(ParameterGrid(PARAM_GRIDS[name]))} configurações; teste fechado.", flush=True)
        start = time.perf_counter()
        search = GridSearchCV(make_pipeline(name), PARAM_GRIDS[name], scoring=scoring, cv=fold,
                              refit=False, n_jobs=1, pre_dispatch=1, error_score="raise", return_train_score=False)
        with threadpool_limits(limits=2):
            search.fit(X_dev, y_dev)
        results = search.cv_results_
        indexes = sorted(range(len(results["params"])), key=lambda i: (-results["mean_test_mae"][i], -results["mean_test_rmse"][i], i))
        best_index = indexes[0]
        params = results["params"][best_index]
        for i, candidate in enumerate(results["params"]):
            candidates.append({"model": name, "params": json.dumps(candidate, sort_keys=True),
                               "MAE_days": -float(results["mean_test_mae"][i]), "RMSE_days": -float(results["mean_test_rmse"][i]),
                               "R2": float(results["mean_test_r2"][i]), "mean_fit_seconds": results["mean_fit_time"][i],
                               "selected_within_model": i == best_index})
        estimator = make_pipeline(name).set_params(**params)
        with threadpool_limits(limits=2):
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_val)
        result = checked_metrics(y_val, pred)
        np.testing.assert_allclose(result["MAE_days"], -results["mean_test_mae"][best_index], atol=1e-9)
        rows.append({"model": name, **result, "fit_seconds": time.perf_counter() - start,
                     "best_params": json.dumps(params, sort_keys=True), "candidates": len(results["params"])})
        selected_params[name] = params
        validation[name] = pred
        joblib.dump(estimator, models / f"validation_{name}.joblib", compress=3)
        # Medianas/categorias aprendidas só no treino; nomes removidos pelo seletor.
        pre = estimator.named_steps["preprocess"]
        all_names = pre.get_feature_names_out()
        mask = estimator.named_steps["select"].get_support()
        write_csv(pd.DataFrame({"transformed_feature": all_names, "kept": mask}), tables / f"features_{name}.csv")
        write_json({"numeric_imputation": dict(zip(NUMERIC, map(float, pre.named_transformers_["numeric"].named_steps["impute"].statistics_))),
                    "categories": {name: values.tolist() for name, values in zip(CATEGORICAL, pre.named_transformers_["categorical"].named_steps["encode"].categories_)},
                    "fit_rows": len(X_train), "transformed_before_selection": len(all_names), "transformed_after_selection": int(mask.sum()),
                    "removed": all_names[~mask].tolist()}, tables / f"preprocessing_{name}.json")
        write_csv(pd.DataFrame(rows), tables / "validation_metrics.csv")
        write_csv(pd.DataFrame(candidates), tables / "search_candidates.csv")
        print(f"VALIDATION: {name}, MAE={result['MAE_days']:.4f} dias.", flush=True)
    ranking = pd.DataFrame(rows).sort_values(["MAE_days", "RMSE_days", "model"]).reset_index(drop=True)
    write_csv(ranking, tables / "validation_metrics.csv")
    validation.to_parquet(tables / "validation_predictions.parquet", index=False)
    winner = ranking.iloc[0].model
    frozen = {"selected_model": winner, "best_params": selected_params[winner], "all_best_params": selected_params,
              "selection_metric": "MAE_days", "validation_metrics": ranking[["model", "MAE_days", "RMSE_days", "R2"]].to_dict("records"),
              "protocol_sha256": sha256(tables / "protocol.json"), "validation_predictions_sha256": sha256(tables / "validation_predictions.parquet"),
              "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluations_at_selection": 0,
              "estimator_code_sha256": {str(p.relative_to(root)): sha256(p) for p in
                  [root / "src/olist_modeling/estimators.py", root / "src/olist_modeling/protocol.py"]}}
    write_json(frozen, tables / "selection_frozen.json")
    print(f"SELECTION SEALED: {winner}; nenhum resultado do teste consultado.", flush=True)
    if winner != "Baseline":
        fitted = joblib.load(models / f"validation_{winner}.joblib")
        with threadpool_limits(limits=2):
            importance = validation_importance(fitted, X_val, y_val)
        table = pd.DataFrame({"feature": FEATURE_COLUMNS, "increase_MAE_days_mean": importance.importances_mean,
                              "std": importance.importances_std}).sort_values("increase_MAE_days_mean", ascending=False)
        write_csv(table, tables / "importance_validation_permutation.csv")


def evaluate(root):
    tables, models = paths(root)
    if (tables / "test_evaluation_started.json").exists():
        raise RuntimeError("Teste final já aberto. Não repetir seleção/avaliação; use report ou verify.")
    frozen_path = tables / "selection_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["protocol_sha256"] == sha256(tables / "protocol.json")
    for relative, expected in frozen.get("estimator_code_sha256", {}).items():
        assert sha256(root / relative) == expected, "Código do estimador mudou após selar seleção"
    data, meta = load_prepared(root)
    X_fit, y_fit = prepare_X(data.loc[meta.is_final_train]), data.loc[meta.is_final_train, TARGET]
    name = frozen["selected_model"]
    final = DummyRegressor(strategy="median") if name == "Baseline" else make_pipeline(name).set_params(**frozen["best_params"])
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        final.fit(X_fit, y_fit)
    baseline = DummyRegressor(strategy="median").fit(X_fit, y_fit)
    joblib.dump(final, models / "final_model.joblib", compress=3)
    joblib.dump(baseline, models / "final_baseline.joblib", compress=3)
    write_json({"model": name, "fit_rows": len(X_fit), "fit_seconds": time.perf_counter() - start,
                "selection_sha256": sha256(frozen_path), "label_cutoff_exclusive": str(TEST_START),
                "fit_order_ids_sha256": __import__('hashlib').sha256('\n'.join(data.loc[meta.is_final_train, 'order_id']).encode()).hexdigest()}, tables / "final_fit.json")
    final_contract = {"feature_columns": FEATURE_COLUMNS, "target": TARGET, "unit": "days", "prediction_floor": 0,
                      "selected_model": name, "categorical_columns": CATEGORICAL, "model_file": "final_model.joblib",
                      "prediction": "joblib.load(model_path).predict(dataframe[feature_columns]); tipos/imputação/seleção internos",
                      "scope": "pedidos eventualmente entregues; cadastros assumidos vigentes na compra"}
    write_json(final_contract, models / "inference_contract.json")
    # Só depois de selar escolha e reajustar modelos abrir o teste final.
    write_json({"selection_sha256": sha256(frozen_path), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_event": 1, "models": [name, "Baseline de referência"]}, tables / "test_evaluation_started.json")
    print(f"FINAL TEST: escolha {name} congelada; avaliação única.", flush=True)
    X_test, y_test = prepare_X(data.loc[meta.is_test]), data.loc[meta.is_test, TARGET]
    with threadpool_limits(limits=2):
        pred, base_pred = final.predict(X_test), baseline.predict(X_test)
    final_scores = pd.DataFrame([{"role": "selected", "model": name, **checked_metrics(y_test, pred)},
                                 {"role": "baseline_reference", "model": "Baseline", **checked_metrics(y_test, base_pred)}])
    write_csv(final_scores, tables / "test_metrics.csv")
    predictions = data.loc[meta.is_test, ["order_id", "purchase_timestamp", "customer_state", "items_count", "sellers_count", "items_missing_distance_count"]].reset_index(drop=True)
    predictions["actual_days"], predictions["predicted_days"], predictions["baseline_days"] = y_test.to_numpy(), pred, base_pred
    predictions["residual_days"] = predictions.actual_days - predictions.predicted_days
    predictions["absolute_error_days"] = predictions.residual_days.abs()
    predictions.to_parquet(tables / "test_predictions.parquet", index=False)
    write_csv(predictions.nlargest(30, "absolute_error_days"), tables / "largest_test_errors.csv")
    for column in ("customer_state", "sellers_count", "items_missing_distance_count"):
        groups = []
        for value, group in predictions.groupby(column):
            groups.append({column: value, "orders": len(group), **checked_metrics(group.actual_days, group.predicted_days)})
        write_csv(pd.DataFrame(groups), tables / f"test_errors_by_{column}.csv")
    by_time = predictions.assign(purchase_month=predictions.purchase_timestamp.dt.strftime("%Y-%m")).groupby("purchase_month")
    write_csv(pd.DataFrame([{"purchase_month": month, "orders": len(g), **checked_metrics(g.actual_days, g.predicted_days)} for month, g in by_time]), tables / "test_errors_by_month.csv")
    bins = pd.cut(predictions.actual_days, [0, 5, 10, 20, 40, np.inf], right=False)
    write_csv(predictions.assign(actual_range_days=bins.astype(str)).groupby("actual_range_days").agg(
        orders=("actual_days", "size"), mean_actual=("actual_days", "mean"), mean_predicted=("predicted_days", "mean"),
        MAE_days=("absolute_error_days", "mean")).reset_index(), tables / "test_errors_by_actual_range.csv")
    write_json({"test_evaluation_events": 1, "rows": len(predictions), "mean_residual_days": float(predictions.residual_days.mean()),
                "median_absolute_error_days": float(predictions.absolute_error_days.median()),
                "p90_absolute_error_days": float(predictions.absolute_error_days.quantile(.9)),
                "p95_absolute_error_days": float(predictions.absolute_error_days.quantile(.95)),
                "within_3_days_pct": float(predictions.absolute_error_days.le(3).mean() * 100),
                "within_5_days_pct": float(predictions.absolute_error_days.le(5).mean() * 100),
                "negative_predictions": int((pred < 0).sum()),
                "mae_improvement_over_baseline_pct": float((1 - final_scores.iloc[0].MAE_days / final_scores.iloc[1].MAE_days) * 100)}, tables / "test_diagnostics.json")
    if name != "Baseline":
        underlying = final.named_steps["model"].estimator_
        names = fitted_feature_names(final)
        values = underlying.feature_importances_ if hasattr(underlying, "feature_importances_") else underlying.coef_
        importance = pd.DataFrame({"transformed_feature": names, "importance_or_coefficient": values})
        importance["absolute_value"] = importance.importance_or_coefficient.abs()
        write_csv(importance.sort_values("absolute_value", ascending=False), tables / "importance_final_transformed.csv")
        write_csv(pd.DataFrame({"transformed_feature": final.named_steps["preprocess"].get_feature_names_out(),
                               "kept": final.named_steps["select"].get_support()}), tables / "features_final.csv")
    print(final_scores.to_string(index=False), flush=True)
    verify_preservation(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=["prepare", "select", "evaluate", "report", "all"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "all" and (root / "resultados/tabelas/task02/selection_frozen.json").exists():
        raise RuntimeError("Execução inicial já concluída/selada; use verify/report, sem reabrir seleção.")
    if args.stage in ("prepare", "all"):
        prepare(root)
    if args.stage in ("select", "all"):
        select(root)
    if args.stage in ("evaluate", "all"):
        evaluate(root)
    if args.stage in ("report", "all"):
        from .report import report
        report(root)


if __name__ == "__main__":
    main()
