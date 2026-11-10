"""Verificação técnica; refits com decisões seladas, sem nova seleção ou tuning."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from threadpoolctl import threadpool_limits

from olist_features.audit import sha256, zip_integrity
from olist_features.core import FEATURE_COLUMNS, TARGET
from olist_features.pipeline import write_csv, write_json
from .estimators import make_pipeline
from .protocol import CATEGORICAL, NUMERIC, TEST_START, VALIDATION_START, prepare_X, verify_preservation
from .run import checked_metrics, load_prepared


def verify(root: Path, refit: bool = True):
    folder, model_dir = root / "resultados/tabelas/task02", root / "resultados/modelos/task02"
    checks = []

    def check(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": str(evidence)})
        if not passed:
            raise AssertionError(f"{name}: {evidence}")

    integrity_before = zip_integrity(root)
    preservation = verify_preservation(root)
    check("task01_preserved", preservation["passed"], preservation["files_checked"])
    audit = json.loads((folder / "source_audit.json").read_text(encoding="utf-8"))
    check("raw_zip_integrity", integrity_before == audit["raw_integrity"], "CRC e SHA-256")
    check("engineering_recomputed_exactly", audit["passed"] and audit["features_compared"] == 30, audit["regenerated_rows"])
    data, meta = load_prepared(root)
    check("unique_order_id", data.order_id.is_unique, len(data))
    sets = [set(meta.loc[meta[col], "order_id"]) for col in ("is_train", "is_validation", "is_test")]
    check("disjoint_train_validation_test", not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)), "IDs sem sobreposição")
    check("train_labels_known_before_validation", meta.loc[meta.is_train, "delivery_timestamp"].lt(VALIDATION_START).all(), VALIDATION_START)
    check("validation_labels_known_before_test", meta.loc[meta.is_validation, "delivery_timestamp"].lt(TEST_START).all(), TEST_START)
    check("final_train_labels_known_before_test", meta.loc[meta.is_final_train, "delivery_timestamp"].lt(TEST_START).all(), TEST_START)
    check("final_train_test_disjoint", not (meta.is_final_train & meta.is_test).any(), "nenhum pedido de teste no ajuste final")
    check("chronological_purchase_windows", meta.loc[meta.is_train, "purchase_timestamp"].max() < meta.loc[meta.is_validation, "purchase_timestamp"].min()
          and meta.loc[meta.is_validation, "purchase_timestamp"].max() < meta.loc[meta.is_test, "purchase_timestamp"].min(), "janelas consecutivas")
    frozen_file = folder / "selection_frozen.json"
    frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
    fit = json.loads((folder / "final_fit.json").read_text(encoding="utf-8"))
    opening = json.loads((folder / "test_evaluation_started.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((folder / "test_diagnostics.json").read_text(encoding="utf-8"))
    check("selection_sealed_before_test", fit["selection_sha256"] == opening["selection_sha256"] == sha256(frozen_file)
          and frozen["sealed_at_utc"] < opening["started_at_utc"], frozen["selected_model"])
    check("one_test_evaluation_event", opening["evaluation_event"] == diagnostics["test_evaluation_events"] == 1, "recomputar fórmulas não é nova seleção")
    check("no_test_at_selection", frozen["test_evaluations_at_selection"] == 0, 0)
    ranking = pd.read_csv(folder / "validation_metrics.csv").sort_values(["MAE_days", "RMSE_days", "model"])
    check("selection_by_validation_only", ranking.iloc[0].model == frozen["selected_model"], ranking.iloc[0].model)
    validation = pd.read_parquet(folder / "validation_predictions.parquet")
    test = pd.read_parquet(folder / "test_predictions.parquet")
    check("validation_predictions_sealed", sha256(folder / "validation_predictions.parquet") == frozen["validation_predictions_sha256"], len(validation))
    check("validation_order_alignment", validation.order_id.tolist() == data.loc[meta.is_validation, "order_id"].tolist(), len(validation))
    check("test_order_alignment", test.order_id.tolist() == data.loc[meta.is_test, "order_id"].tolist(), len(test))
    X_train, y_train = prepare_X(data.loc[meta.is_train]), data.loc[meta.is_train, TARGET]
    X_val = prepare_X(data.loc[meta.is_validation])
    X_final, y_final = prepare_X(data.loc[meta.is_final_train]), data.loc[meta.is_final_train, TARGET]
    X_test = prepare_X(data.loc[meta.is_test])
    check("X_exact_allowlist", X_train.columns.tolist() == FEATURE_COLUMNS and TARGET not in X_train, len(X_train.columns))
    metric_rows, refits = [], []

    def fitted_state(model, X, label):
        pre = model.named_steps["preprocess"]
        learned = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
        expected = X[NUMERIC].median().fillna(0).to_numpy()
        check(f"{label}_train_only_medians", np.allclose(learned, expected), len(X))
        encoder = pre.named_transformers_["categorical"].named_steps["encode"]
        for col, categories in zip(CATEGORICAL, encoder.categories_):
            expected_categories = set(X[col].fillna("__MISSING__"))
            check(f"{label}_train_only_categories_{col}", set(categories) == expected_categories, len(categories))
        before = learned.copy()
        values = model.named_steps["select"].transform(pre.transform(X_val))
        check(f"{label}_finite_transformed_validation", np.isfinite(values.data if hasattr(values, "toarray") else values).all(), values.shape)
        check(f"{label}_transform_does_not_refit", np.array_equal(learned, before), "medianas preservadas")

    for name, params in frozen["all_best_params"].items():
        estimator = joblib.load(model_dir / f"validation_{name}.joblib")
        with threadpool_limits(limits=2):
            pred = estimator.predict(X_val)
        check(f"{name}_saved_validation_predictions", np.allclose(pred, validation[name], rtol=0, atol=1e-9), len(pred))
        actual_metrics = checked_metrics(validation.actual_days, validation[name])
        recorded = ranking.set_index("model").loc[name]
        for metric, value in actual_metrics.items():
            check(f"{name}_independent_validation_{metric}", np.isclose(value, recorded[metric], rtol=1e-10), value)
            metric_rows.append({"split": "validation", "model": name, "metric": metric, "recomputed": value, "recorded": recorded[metric]})
        if name != "Baseline":
            fitted_state(estimator, X_train, name)
        if refit:
            start = time.perf_counter()
            fresh = DummyRegressor(strategy="median") if name == "Baseline" else make_pipeline(name).set_params(**params)
            with threadpool_limits(limits=2):
                fresh.fit(X_train, y_train)
                repeated = fresh.predict(X_val)
            delta = float(np.max(np.abs(repeated - validation[name].to_numpy())))
            check(f"{name}_reproducible_refit", delta <= 1e-9, delta)
            refits.append({"stage": "validation", "model": name, "rows_fit": len(X_train), "max_abs_prediction_delta": delta,
                           "elapsed_seconds": time.perf_counter() - start, "passed": delta <= 1e-9})
    test_metrics = pd.read_csv(folder / "test_metrics.csv").set_index("role")
    for role, filename, column in [("selected", "final_model.joblib", "predicted_days"),
                                    ("baseline_reference", "final_baseline.joblib", "baseline_days")]:
        loaded = joblib.load(model_dir / filename)
        pred = loaded.predict(X_test)
        check(f"{role}_saved_test_predictions", np.allclose(pred, test[column], rtol=0, atol=1e-9), len(pred))
        for metric, value in checked_metrics(test.actual_days, test[column]).items():
            recorded = test_metrics.loc[role, metric]
            check(f"{role}_independent_test_{metric}", np.isclose(value, recorded, rtol=1e-10), value)
            metric_rows.append({"split": "test", "model": role, "metric": metric, "recomputed": value, "recorded": recorded})
    final_model = joblib.load(model_dir / "final_model.joblib")
    if frozen["selected_model"] != "Baseline":
        fitted_state(final_model, X_final, "final")
    if refit:
        name, params = frozen["selected_model"], frozen["best_params"]
        start = time.perf_counter()
        fresh = DummyRegressor(strategy="median") if name == "Baseline" else make_pipeline(name).set_params(**params)
        with threadpool_limits(limits=2):
            fresh.fit(X_final, y_final)
            repeated = fresh.predict(X_test)
        delta = float(np.max(np.abs(repeated - test.predicted_days.to_numpy())))
        check("final_reproducible_refit_predictions", delta <= 1e-9, delta)
        refits.append({"stage": "final_inference", "model": name, "rows_fit": len(X_final), "max_abs_prediction_delta": delta,
                       "elapsed_seconds": time.perf_counter() - start, "passed": delta <= 1e-9})
        write_csv(pd.DataFrame(refits), folder / "reproducibility_refits.csv")
    check("raw_zip_unchanged_after_verification", zip_integrity(root) == integrity_before, "CRC e SHA-256")
    check("task01_preserved_after_verification", verify_preservation(root)["passed"], preservation["files_checked"])
    write_csv(pd.DataFrame(metric_rows), folder / "independent_metrics.csv")
    write_csv(pd.DataFrame(checks), folder / "verification_checks.csv")
    write_json({"passed": True, "checks": len(checks), "full_data_refits": refit,
                "refit_prediction_tolerance_days": 1e-9, "test_selection_repeated": False,
                "test_evaluation_events": 1, "task01_files_preserved": preservation["files_checked"]}, folder / "verification_summary.json")
    print(f"VERIFY: {len(checks)} verificações passaram; refits={refit}; TASK 01 preservada.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--no-refit", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), not args.no_refit)


if __name__ == "__main__":
    main()
