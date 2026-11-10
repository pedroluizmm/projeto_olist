"""Pré-processamento ajustável por divisão e grades com orçamento limitado."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .protocol import CATEGORICAL, NUMERIC, SEED, prepare_X


class NonNegativeRegressor(RegressorMixin, BaseEstimator):
    """Piso físico em zero faz parte do estimador salvo e de todo scoring."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return np.maximum(self.estimator_.predict(X), 0.0)


def make_pipeline(name: str) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "Ridge":
        steps.append(("scale", RobustScaler(quantile_range=(25, 75))))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)),
    ])
    pre = ColumnTransformer([("numeric", Pipeline(steps), NUMERIC), ("categorical", categorical, CATEGORICAL)],
                            sparse_threshold=1.0, remainder="drop", verbose_feature_names_out=True)
    estimators = {
        "Ridge": Ridge(solver="lsqr", tol=1e-6, max_iter=3000),
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=16, bootstrap=True,
                                               max_samples=0.85, n_jobs=2, random_state=SEED),
    }
    return Pipeline([("types", FunctionTransformer(prepare_X, validate=False, feature_names_out="one-to-one")),
                     ("preprocess", pre), ("select", VarianceThreshold(threshold=0.0)),
                     ("model", NonNegativeRegressor(estimators[name]))])


PARAM_GRIDS = {
    "Ridge": {"model__estimator__alpha": [0.1, 1.0, 10.0, 100.0]},
    "DecisionTree": {"model__estimator__max_depth": [6, 10, 14], "model__estimator__min_samples_leaf": [20, 80]},
    "RandomForest": {"model__estimator__min_samples_leaf": [10, 40], "model__estimator__max_features": [0.5, 1.0]},
}


def fitted_feature_names(pipeline):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return names[pipeline.named_steps["select"].get_support()]
