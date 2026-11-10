"""Regressões E0: somente inferência, validação e saídas temporárias isoladas."""

import ast
import builtins
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from olist_modeling import importance as e0
from olist_modeling.estimators import NonNegativeRegressor

ROOT = Path(__file__).resolve().parents[1]


class FixedPredictor(RegressorMixin, BaseEstimator):
    """Regressor definido diretamente: nenhuma preparação exige fit."""

    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        raise AssertionError("E0 não pode chamar fit")

    def predict(self, X):
        self.calls.append(tuple(X.index))
        return X.signal.to_numpy()


@pytest.fixture
def forbid_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit/fit_transform proibido em E0")
    for cls in [Pipeline, Ridge, NonNegativeRegressor, SimpleImputer, RobustScaler,
                OneHotEncoder, VarianceThreshold, GridSearchCV]:
        monkeypatch.setattr(cls, "fit", forbidden)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbidden)


def test_configuration_and_training_entrypoint_share_corrected_helper():
    assert e0.CONFIG == {"scoring": "neg_mean_absolute_error", "n_repeats": 3,
                         "random_state": 42, "n_jobs": 1, "max_samples": 1.0}
    tree = ast.parse((ROOT / "src/olist_modeling/run.py").read_text(encoding="utf-8"))
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "validation_importance" in calls
    assert "permutation_importance" not in calls


def test_same_orders_negative_controls_and_reproducibility(forbid_fit):
    # Erros deliberadamente heterogêneos: uma subamostra teria outro MAE.
    X = pd.DataFrame({"signal": np.arange(101, dtype=float),
                      "ignored": np.arange(101, dtype=float)[::-1], "constant": 7.0},
                     index=[f"order-{i}" for i in range(101)])
    y = X.signal.to_numpy() + np.r_[np.zeros(50), np.full(51, 50.0)]
    before = joblib.hash(X)
    model = FixedPredictor()
    result = e0.validation_importance(model, X, y)
    assert len(model.calls) == 1 + 3 * len(X.columns)
    assert all(ids == tuple(X.index) for ids in model.calls)
    np.testing.assert_array_equal(result.importances[1:], np.zeros((2, 3)))
    assert np.any(result.importances[0] != 0)
    repeated = e0.validation_importance(FixedPredictor(), X, y)
    np.testing.assert_array_equal(result.importances, repeated.importances)
    assert joblib.hash(X) == before


@pytest.fixture(scope="module")
def real_e0():
    # Escopo de monkeypatch manual permite compartilhar o cálculo sem fit.
    with pytest.MonkeyPatch.context() as patch:
        def no_fit(*args, **kwargs):
            raise AssertionError("fit proibido")
        for cls in [Pipeline, Ridge, NonNegativeRegressor, SimpleImputer, RobustScaler,
                    OneHotEncoder, VarianceThreshold, GridSearchCV]:
            patch.setattr(cls, "fit", no_fit)
            if hasattr(cls, "fit_transform"):
                patch.setattr(cls, "fit_transform", no_fit)
        opened = []
        original_open = builtins.open
        original_path_open = Path.open
        original_parquet = pd.read_parquet

        def check_path(path):
            if isinstance(path, (str, bytes, Path)):
                p = Path(path).resolve()
                if "resultados" in p.parts and (p.name.startswith("test_") or p.name == "largest_test_errors.csv"
                                               or p.name in {"final_model.joblib", "final_baseline.joblib"}):
                    raise AssertionError(f"Acesso ao teste/final proibido: {p}")
                opened.append(str(p))

        def guarded_open(path, *args, **kwargs):
            check_path(path)
            return original_open(path, *args, **kwargs)

        def guarded_path_open(path, *args, **kwargs):
            check_path(path)
            return original_path_open(path, *args, **kwargs)

        parquet_reads = []

        def guarded_parquet(path, *args, **kwargs):
            check_path(path)
            if Path(path).name in {"olist_orders_features.parquet", "order_selection_audit.parquet"}:
                filters = kwargs["filters"]
                assert filters[0][:2] == ("order_id", "in")
                assert len(filters[0][2]) == 12675
                assert kwargs.get("columns") is not None
            parquet_reads.append(Path(path).name)
            return original_parquet(path, *args, **kwargs)

        patch.setattr(builtins, "open", guarded_open)
        patch.setattr(Path, "open", guarded_path_open)
        patch.setattr(pd, "read_parquet", guarded_parquet)
        # Snapshot de conteúdo, sem abrir resultados de teste.
        protected = [*sorted((ROOT / "dados/tratados").glob("*")),
                     *sorted((ROOT / "resultados/modelos/task02").glob("validation_*.joblib")),
                     ROOT / "resultados/tabelas/task02/validation_predictions.parquet",
                     ROOT / "resultados/tabelas/task02/selection_frozen.json",
                     ROOT / "resultados/tabelas/task02/importance_validation_permutation.csv"]
        before = {str(p): e0.file_hash(p) for p in protected if p.is_file()}
        first = e0.calculate_e0(ROOT)
        second = e0.calculate_e0(ROOT)
        after = {str(p): e0.file_hash(p) for p in protected if p.is_file()}
    return first, second, before, after, opened, parquet_reads


def test_real_predictions_metrics_and_controls(real_e0):
    (table, comparison, metadata, ids), _, _, _, _, _ = real_e0
    assert len(table) == len(comparison) == 30
    assert len(ids) == len(set(ids)) == metadata["rows"] == 12675
    assert metadata["max_prediction_difference_days"] == 0
    np.testing.assert_allclose(
        list(metadata["validation_metrics"].values()),
        [4.259612014663884, 5.751332358545968, .322133438132609], rtol=0, atol=1e-12,
    )
    assert metadata["fit_called"] is False and metadata["final_test_used"] is False
    assert all(values == [0.0, 0.0, 0.0] for values in metadata["negative_controls"].values())


def test_real_reproduction_and_original_files_preserved(real_e0):
    first, second, before, after, _, _ = real_e0
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert before == after
    assert first[2]["ordered_order_ids_sha256"] == second[2]["ordered_order_ids_sha256"]


def test_real_access_is_validation_only(real_e0):
    _, _, _, _, opened, parquet_reads = real_e0
    assert parquet_reads == ["validation_predictions.parquet", "olist_orders_features.parquet",
                             "order_selection_audit.parquet"] * 2
    assert not any(Path(p).name.startswith("test_") for p in opened)


def test_save_is_separate_and_refuses_overwrite(tmp_path, real_e0):
    table, comparison, metadata, ids = real_e0[0]
    historical = tmp_path / "resultados/tabelas/task02/importance_validation_permutation.csv"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical sentinel\n", encoding="utf-8")
    e0.save_e0(tmp_path, table, comparison, metadata, ids)
    assert historical.read_text(encoding="utf-8") == "historical sentinel\n"
    saved = tmp_path / "resultados/tabelas/e0/importance_validation_permutation.csv"
    before = e0.file_hash(saved)
    with pytest.raises(FileExistsError, match="sobrescrita"):
        e0.save_e0(tmp_path, table, comparison, metadata, ids)
    assert e0.file_hash(saved) == before
    assert historical.read_text(encoding="utf-8") == "historical sentinel\n"
