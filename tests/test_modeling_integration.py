"""Evidências reais da TASK 02; executar após evaluate/verify/notebooks."""

import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd

from olist_features.audit import sha256, zip_integrity
from olist_features.core import FEATURE_COLUMNS, TARGET
from olist_modeling.estimators import fitted_feature_names
from olist_modeling.protocol import CATEGORICAL, NUMERIC, TEST_START, VALIDATION_START, prepare_X, verify_preservation
from olist_modeling.run import checked_metrics, load_prepared


def folder(root):
    return root / "resultados/tabelas/task02"


def read_json(root, name):
    return json.loads((folder(root) / name).read_text(encoding="utf-8"))


def test_modeling_real_source_rebuilt_and_prior_work_preserved(project_root):
    assert verify_preservation(project_root)["passed"]
    audit = read_json(project_root, "source_audit.json")
    assert audit["passed"] and audit["regenerated_rows"] == 96470
    assert audit["features_compared"] == 30
    assert audit["raw_integrity"] == zip_integrity(project_root)
    temporal = pd.read_csv(folder(project_root) / "feature_temporal_audit.csv")
    assert temporal.feature.tolist() == FEATURE_COLUMNS
    assert not temporal.future_outcome_used.any()
    assert not audit["historical_validity_provable"]


def test_modeling_real_splits_and_label_availability(project_root):
    data, meta = load_prepared(project_root)
    assert [meta[col].sum() for col in ("is_train", "is_validation", "is_test", "is_final_train")] == [64401, 12675, 14485, 80043]
    assert not (meta.is_train & meta.is_validation | meta.is_train & meta.is_test | meta.is_validation & meta.is_test).any()
    assert not (meta.is_final_train & meta.is_test).any()
    assert data.loc[meta.is_train, "delivery_timestamp"].max() < VALIDATION_START
    assert data.loc[meta.is_validation | meta.is_final_train, "delivery_timestamp"].max() < TEST_START
    assert data.loc[meta.is_test, "purchase_timestamp"].min() >= TEST_START
    assert meta.label_unavailable_at_cutoff.sum() == 4909


def test_modeling_real_eda_train_scope_only(project_root):
    scope = read_json(project_root, "eda_scope.json")
    assert scope["rows"] == 64401
    assert not scope["test_used"]
    assert scope["outliers_removed"] == 0
    monthly = pd.read_csv(folder(project_root) / "eda_train_by_month.csv")
    assert monthly.orders.sum() == scope["rows"]
    assert monthly.month.max() < "2018-05"
    missing_months = monthly.loc[monthly.orders.eq(0)]
    assert missing_months.median_days.isna().all()
    categories = pd.read_csv(folder(project_root) / "eda_train_categorical.csv")
    assert set(categories.feature) == set(CATEGORICAL)
    assert categories.groupby("feature").orders.sum().eq(scope["rows"]).all()


def test_modeling_real_search_budget_and_selected_decision(project_root):
    candidates = pd.read_csv(folder(project_root) / "search_candidates.csv")
    assert candidates.groupby("model").size().to_dict() == {"Ridge": 4, "DecisionTree": 6, "RandomForest": 4}
    assert candidates.groupby("model").selected_within_model.sum().eq(1).all()
    ranking = pd.read_csv(folder(project_root) / "validation_metrics.csv").sort_values(["MAE_days", "RMSE_days", "model"])
    frozen = read_json(project_root, "selection_frozen.json")
    assert frozen["selected_model"] == ranking.iloc[0].model
    assert frozen["test_evaluations_at_selection"] == 0
    opening = read_json(project_root, "test_evaluation_started.json")
    assert opening["selection_sha256"] == sha256(folder(project_root) / "selection_frozen.json")
    assert frozen["sealed_at_utc"] < opening["started_at_utc"]
    assert opening["evaluation_event"] == 1


def test_modeling_real_validation_metrics_and_baseline(project_root):
    data, meta = load_prepared(project_root)
    predictions = pd.read_parquet(folder(project_root) / "validation_predictions.parquet")
    metrics = pd.read_csv(folder(project_root) / "validation_metrics.csv").set_index("model")
    assert predictions.order_id.tolist() == data.loc[meta.is_validation, "order_id"].tolist()
    np.testing.assert_allclose(predictions.Baseline, data.loc[meta.is_train, TARGET].median())
    for name in metrics.index:
        for key, value in checked_metrics(predictions.actual_days, predictions[name]).items():
            np.testing.assert_allclose(value, metrics.loc[name, key], rtol=1e-10)


def test_modeling_real_preprocessing_fit_only_train(project_root):
    data, meta = load_prepared(project_root)
    X = prepare_X(data.loc[meta.is_train])
    for name in ("Ridge", "DecisionTree", "RandomForest"):
        model = joblib.load(project_root / f"resultados/modelos/task02/validation_{name}.joblib")
        pre = model.named_steps["preprocess"]
        learned = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
        np.testing.assert_allclose(learned, X[NUMERIC].median().fillna(0))
        encode = pre.named_transformers_["categorical"].named_steps["encode"]
        for col, values in zip(CATEGORICAL, encode.categories_):
            assert set(values) == set(X[col].fillna("__MISSING__"))
        selected = fitted_feature_names(model)
        assert "numeric__items_missing_price_count" not in selected
        assert "numeric__items_missing_freight_count" not in selected
        if name == "Ridge":
            scale = pre.named_transformers_["numeric"].named_steps["scale"]
            np.testing.assert_allclose(scale.center_, X[NUMERIC].median().fillna(0))


def test_modeling_real_test_metrics_and_refit_baseline(project_root):
    data, meta = load_prepared(project_root)
    pred = pd.read_parquet(folder(project_root) / "test_predictions.parquet")
    recorded = pd.read_csv(folder(project_root) / "test_metrics.csv").set_index("role")
    assert pred.order_id.tolist() == data.loc[meta.is_test, "order_id"].tolist()
    np.testing.assert_allclose(pred.actual_days, data.loc[meta.is_test, TARGET])
    np.testing.assert_allclose(pred.baseline_days, data.loc[meta.is_final_train, TARGET].median())
    for role, col in [("selected", "predicted_days"), ("baseline_reference", "baseline_days")]:
        for key, value in checked_metrics(pred.actual_days, pred[col]).items():
            np.testing.assert_allclose(value, recorded.loc[role, key], rtol=1e-10)
    assert pred.predicted_days.ge(0).all()
    assert np.isfinite(pred.predicted_days).all()


def test_modeling_real_final_model_inference_and_extra_future_columns(project_root):
    data, meta = load_prepared(project_root)
    sample = data.loc[meta.is_test].iloc[::251].copy()
    model = joblib.load(project_root / "resultados/modelos/task02/final_model.joblib")
    pred = model.predict(sample[FEATURE_COLUMNS])
    sample["order_delivered_customer_date"] = "2099-01-01"
    sample[TARGET] = -999
    np.testing.assert_allclose(model.predict(sample), pred)
    sample["customer_state"] = "UF_NOVA"
    sample["weight_g_total"] = np.nan
    assert np.isfinite(model.predict(sample)).all()


def test_modeling_real_refits_and_verification_passed(project_root):
    summary = read_json(project_root, "verification_summary.json")
    assert summary["passed"] and summary["full_data_refits"]
    assert summary["test_evaluation_events"] == 1
    checks = pd.read_csv(folder(project_root) / "verification_checks.csv")
    assert len(checks) >= 40 and checks.passed.all()
    refits = pd.read_csv(folder(project_root) / "reproducibility_refits.csv")
    assert len(refits) == 5
    assert refits.passed.all()
    assert refits.max_abs_prediction_delta.le(1e-9).all()


def test_modeling_real_executed_notebooks(project_root):
    for name in ("02_eda_olist.ipynb", "03_treinamento_olist.ipynb"):
        nb = nbformat.read(project_root / "notebooks" / name, as_version=4)
        nbformat.validate(nb)
        assert nb.metadata.kernelspec.name == "olist"
        cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(cells) >= 4
        assert all(c.execution_count is not None for c in cells)
        assert not any(output.output_type == "error" for cell in cells for output in cell.get("outputs", []))
