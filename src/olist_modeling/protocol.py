"""Contrato de dados, auditoria e separação temporal, sem ajuste de modelos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from olist_features.audit import find_source, load_tables, sha256, zip_integrity
from olist_features.core import FEATURE_COLUMNS, FEATURE_SPECS, TARGET, build_features, select_target
from olist_features.pipeline import write_csv, write_json

SEED = 42
VALIDATION_START = pd.Timestamp("2018-04-15")
TEST_START = pd.Timestamp("2018-06-21")
CATEGORICAL = ["customer_state", "main_category", "purchase_weekday", "purchase_month"]
NUMERIC = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL]


def verify_preservation(root: Path) -> dict:
    path = root / "resultados/tabelas/task02/task01_preservation.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    changes = [name for name, expected in recorded.items() if not (root / name).is_file() or sha256(root / name) != expected]
    if changes:
        raise AssertionError(f"Arquivos da TASK 01 alterados: {changes}")
    return {"passed": True, "files_checked": len(recorded), "changed": []}


def load_cohort(root: Path) -> pd.DataFrame:
    folder = root / "dados/tratados"
    contract = json.loads((folder / "feature_contract.json").read_text(encoding="utf-8"))
    if contract["feature_columns"] != FEATURE_COLUMNS:
        raise AssertionError("O contrato de features mudou")
    data = pd.read_parquet(folder / "olist_orders_features.parquet")
    if list(data.columns) != ["order_id", *FEATURE_COLUMNS, TARGET] or not data.order_id.is_unique:
        raise AssertionError("Schema ou chave do dataset inválidos")
    audit = pd.read_parquet(root / "resultados/tabelas/order_selection_audit.parquet")
    audit = audit.loc[audit.selection_reason.eq("included"), ["order_id", "purchase_timestamp", "delivery_timestamp"]]
    if set(audit.order_id) != set(data.order_id) or not audit.order_id.is_unique:
        raise AssertionError("Coorte e auditoria não correspondem")
    data = data.merge(audit, on="order_id", how="left", validate="one_to_one")
    expected = (data.delivery_timestamp - data.purchase_timestamp).dt.total_seconds() / 86400
    np.testing.assert_allclose(expected, data[TARGET], rtol=0, atol=1e-12)
    if data.purchase_timestamp.isna().any() or data[TARGET].lt(0).any():
        raise AssertionError("Datas ou alvos inválidos")
    return data.sort_values(["purchase_timestamp", "order_id"]).reset_index(drop=True)


def audit_source(root: Path, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    verify_preservation(root)
    before = zip_integrity(root)
    raw, _ = load_tables(find_source(root))
    regenerated, _, _ = build_features(raw)
    labels, _, _ = select_target(raw["orders"], regenerated)
    rebuilt = regenerated.merge(labels, on="order_id", validate="one_to_one")
    expected = data[["order_id", *FEATURE_COLUMNS, TARGET]].sort_values("order_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, expected)
    if before != zip_integrity(root):
        raise AssertionError("O ZIP mudou")
    audit_rows = []
    for name in FEATURE_COLUMNS:
        source, description = FEATURE_SPECS[name]
        assumption = ("Referência estática por CEP, sem versão temporal; aproximação, não rota."
                      if "distance" in name else "Cadastro sem histórico: vigência na compra assumida."
                      if any(t in source for t in ("products.", "sellers.", "customers.")) else
                      "Carrinho no registro da compra; dump não registra revisões." if "items." in source else
                      "Derivação exclusiva do timestamp da compra.")
        audit_rows.append({"feature": name, "source": source, "definition": description,
                           "future_outcome_used": False, "availability": "compra, sob hipóteses documentadas",
                           "limitation": assumption})
    folder = root / "resultados/tabelas/task02"
    write_csv(pd.DataFrame(audit_rows), folder / "feature_temporal_audit.csv")
    write_json({"passed": True, "regenerated_rows": len(rebuilt), "features_compared": len(FEATURE_COLUMNS),
                "raw_integrity": before, "target_independently_validated_in_task01": True,
                "historical_validity_provable": False, "cohort_only_observed_deliveries": True}, folder / "source_audit.json")
    return raw


def split_cohort(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = data[["order_id", "purchase_timestamp", "delivery_timestamp"]].copy()
    p, d = meta.purchase_timestamp, meta.delivery_timestamp
    meta["segment"] = np.select([p.lt(VALIDATION_START), p.lt(TEST_START)], ["train", "validation"], default="test")
    meta["is_train"] = meta.segment.eq("train") & d.lt(VALIDATION_START)
    meta["is_validation"] = meta.segment.eq("validation") & d.lt(TEST_START)
    meta["is_test"] = meta.segment.eq("test")
    meta["is_final_train"] = p.lt(TEST_START) & d.lt(TEST_START)
    meta["label_unavailable_at_cutoff"] = ((meta.segment.eq("train") & ~meta.is_train) |
                                             (meta.segment.eq("validation") & ~meta.is_validation))
    summary = []
    for name, mask, cutoff in [
        ("train_candidates", meta.segment.eq("train"), VALIDATION_START),
        ("train", meta.is_train, VALIDATION_START),
        ("validation_candidates", meta.segment.eq("validation"), TEST_START),
        ("validation", meta.is_validation, TEST_START),
        ("test", meta.is_test, None), ("final_train", meta.is_final_train, TEST_START),
    ]:
        part = meta.loc[mask]
        if part.empty:
            raise ValueError(f"Divisão vazia: {name}")
        summary.append({"split": name, "rows": len(part), "cohort_pct": len(part) / len(data) * 100,
                        "purchase_min": str(part.purchase_timestamp.min()), "purchase_max": str(part.purchase_timestamp.max()),
                        "delivery_max": str(part.delivery_timestamp.max()), "label_cutoff_exclusive": str(cutoff) if cutoff is not None else "snapshot final; data de extração desconhecida"})
    assert not (meta.is_train & meta.is_validation | meta.is_train & meta.is_test | meta.is_validation & meta.is_test).any()
    assert not (meta.is_final_train & meta.is_test).any()
    assert meta.loc[meta.is_train, "delivery_timestamp"].lt(VALIDATION_START).all()
    assert meta.loc[meta.is_validation | meta.is_final_train, "delivery_timestamp"].lt(TEST_START).all()
    return meta, pd.DataFrame(summary)


def prepare_X(data: pd.DataFrame) -> pd.DataFrame:
    """Conversão determinística de tipos; não estima estatísticas nem aceita colunas extras."""
    result = data[FEATURE_COLUMNS].copy()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    for name in CATEGORICAL:
        values = result[name].astype("string").astype(object)
        result[name] = values.where(pd.notna(values), np.nan)
    if np.isinf(result[NUMERIC].to_numpy()).any():
        raise ValueError("Features numéricas infinitas")
    return result


def split_diagnostics(root: Path, data, meta, raw):
    folder = root / "resultados/tabelas/task02"
    counts = {}
    identities = raw["orders"][["order_id", "customer_id"]].merge(
        raw["customers"][["customer_id", "customer_unique_id"]], on="customer_id", validate="many_to_one")
    identities = meta.merge(identities, on="order_id", validate="one_to_one")
    train_customers = set(identities.loc[identities.is_train, "customer_unique_id"])
    final_customers = set(identities.loc[identities.is_final_train, "customer_unique_id"])
    for label, flag, prior in [("validation", "is_validation", train_customers), ("test", "is_test", final_customers)]:
        part = identities.loc[identities[flag]]
        counts[f"{label}_orders_with_customer_seen_before"] = int(part.customer_unique_id.isin(prior).sum())
        counts[f"{label}_unique_customers_seen_before"] = len(set(part.customer_unique_id) & prior)
    counts["train_purged_labels"] = int((meta.segment.eq("train") & meta.label_unavailable_at_cutoff).sum())
    counts["validation_purged_labels"] = int((meta.segment.eq("validation") & meta.label_unavailable_at_cutoff).sum())
    # Somente períodos de desenvolvimento: diagnosticar viés de maturação sem abrir alvos do teste.
    dev = data.loc[~meta.is_test, [TARGET]].copy()
    dev["segment"] = meta.loc[~meta.is_test, "segment"]
    dev["label_unavailable_at_cutoff"] = meta.loc[~meta.is_test, "label_unavailable_at_cutoff"]
    maturity = dev.groupby(["segment", "label_unavailable_at_cutoff"]).agg(
        orders=(TARGET, "size"), mean_days=(TARGET, "mean"), median_days=(TARGET, "median"), max_days=(TARGET, "max")).reset_index()
    write_csv(maturity, folder / "label_maturation_diagnostic.csv")
    write_json(counts, folder / "split_diagnostics.json")
    return counts
