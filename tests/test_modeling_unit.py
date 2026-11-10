"""Casos artificiais apenas para testar transformações e protocolo da TASK 02."""

import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from olist_features.core import FEATURE_COLUMNS, TARGET
from olist_modeling.estimators import PARAM_GRIDS, make_pipeline
from olist_modeling.protocol import CATEGORICAL, NUMERIC, TEST_START, VALIDATION_START, prepare_X, split_cohort
from olist_modeling.run import checked_metrics, evaluate, select


@pytest.fixture
def modeling_frame():
    n = 30
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + 1 for name in NUMERIC})
    frame["customer_state"] = ["SP", "CE"] * (n // 2)
    frame["main_category"] = ["a", "b", None] * (n // 3)
    frame["purchase_month"] = [1, 2, 3] * (n // 3)
    frame["purchase_weekday"] = [0, 4, 6] * (n // 3)
    return frame[FEATURE_COLUMNS]


def test_maturation_cuts_and_disjoint_ids():
    frame = pd.DataFrame({"order_id": list("abcdef"),
                           "purchase_timestamp": pd.to_datetime(["2018-04-01", "2018-04-14", "2018-04-15", "2018-06-20", "2018-06-21", "2018-06-22"]),
                           "delivery_timestamp": pd.to_datetime(["2018-04-02", "2018-04-16", "2018-05-01", "2018-06-22", "2018-07-01", "2018-07-02"])})
    meta, summary = split_cohort(frame)
    assert meta.loc[meta.is_train, "order_id"].tolist() == ["a"]
    assert meta.loc[meta.is_validation, "order_id"].tolist() == ["c"]
    assert meta.loc[meta.is_test, "order_id"].tolist() == ["e", "f"]
    assert meta.loc[meta.is_final_train, "order_id"].tolist() == ["a", "b", "c"]
    assert meta.label_unavailable_at_cutoff.sum() == 2
    assert summary.set_index("split").loc["train_candidates", "rows"] == 2


@pytest.mark.parametrize("name", list(PARAM_GRIDS))
def test_preprocessing_train_only_unknowns_missing_and_roundtrip(modeling_frame, name, tmp_path):
    train, validation = modeling_frame.iloc[:20].copy(), modeling_frame.iloc[20:].copy()
    train.loc[0, "weight_g_total"] = np.nan
    train["items_missing_price_count"] = 0.0
    validation["items_missing_price_count"] = 100.0
    validation["customer_state"] = "XX"
    validation["weight_g_total"] = 1e12
    model = make_pipeline(name)
    if name == "RandomForest":
        model.set_params(model__estimator__n_estimators=3)
    model.fit(train, np.arange(20, dtype=float) + 2)
    pre = model.named_steps["preprocess"]
    impute = pre.named_transformers_["numeric"].named_steps["impute"]
    assert impute.statistics_[NUMERIC.index("weight_g_total")] == train.weight_g_total.median()
    old_stats = impute.statistics_.copy()
    old_variance = model.named_steps["select"].variances_.copy()
    encoder = pre.named_transformers_["categorical"].named_steps["encode"]
    assert "XX" not in encoder.categories_[CATEGORICAL.index("customer_state")]
    predictions = model.predict(validation)
    assert np.isfinite(predictions).all() and (predictions >= 0).all()
    np.testing.assert_array_equal(old_stats, impute.statistics_)
    np.testing.assert_array_equal(old_variance, model.named_steps["select"].variances_)
    names = pre.get_feature_names_out()
    index = list(names).index("numeric__items_missing_price_count")
    assert not model.named_steps["select"].get_support()[index]
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(validation), predictions)
    # Dados brutos são aceitos pelo pipeline salvo; eventos extras não entram em X.
    validation["order_approved_at"] = "2099-01-01"
    validation[TARGET] = -999.0
    np.testing.assert_allclose(model.predict(validation), predictions)


def test_all_missing_train_column_is_not_imputed_using_future(modeling_frame):
    frame = modeling_frame.copy()
    frame["weight_g_total"] = np.nan
    model = make_pipeline("DecisionTree").fit(frame, np.arange(len(frame)))
    stats = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
    assert stats[NUMERIC.index("weight_g_total")] == 0
    future = frame.copy()
    future["weight_g_total"] = 9999
    assert np.isfinite(model.predict(future)).all()


def test_robust_scale_and_imputation_fit_train(modeling_frame):
    model = make_pipeline("Ridge").fit(modeling_frame, np.arange(len(modeling_frame)))
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.center_, modeling_frame[NUMERIC].median())
    expected = modeling_frame[NUMERIC].quantile(.75) - modeling_frame[NUMERIC].quantile(.25)
    np.testing.assert_allclose(scaler.scale_, expected)


def test_metrics_independent_known_values():
    result = checked_metrics([2.0, 4.0], [1.0, 7.0])
    assert result["MAE_days"] == 2
    assert result["RMSE_days"] == sqrt(5)
    assert result["R2"] == -4


def test_reproducible_forest_small_fixture(modeling_frame):
    model = make_pipeline("RandomForest").set_params(model__estimator__n_estimators=4)
    a, b = clone(model), clone(model)
    y = np.arange(len(modeling_frame), dtype=float)
    np.testing.assert_allclose(a.fit(modeling_frame, y).predict(modeling_frame), b.fit(modeling_frame, y).predict(modeling_frame), atol=1e-12)


def test_only_allowlist_and_no_type_fitting(modeling_frame):
    frame = modeling_frame.copy()
    frame["order_status"] = "delivered"
    frame[TARGET] = 99
    frame.loc[0, "main_category"] = pd.NA
    x = prepare_X(frame)
    assert x.columns.tolist() == FEATURE_COLUMNS
    assert pd.isna(x.loc[0, "main_category"])
    assert frame.loc[0, TARGET] == 99


def test_guard_single_test_evaluation_and_frozen_selection(tmp_path):
    tables = tmp_path / "resultados/tabelas/task02"
    tables.mkdir(parents=True)
    (tables / "test_evaluation_started.json").write_text("{}")
    (tables / "selection_frozen.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Teste final já aberto"):
        evaluate(tmp_path)
    with pytest.raises(RuntimeError, match="Seleção já selada"):
        select(tmp_path)
