"""Leitura somente do ZIP e relatórios de qualidade; sem EDA ou modelagem."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .core import parse_time

TABLE_FILES = {
    "customers": "olist_customers_dataset.csv", "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv", "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv", "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
KEYS = {"orders": ["order_id"], "customers": ["customer_id"], "products": ["product_id"],
        "sellers": ["seller_id"], "items": ["order_id", "order_item_id"],
        "payments": ["order_id", "payment_sequential"], "translation": ["product_category_name"]}
RELATIONS = [
    ("orders", "customer_id", "customers", "customer_id", "N:1 (neste dump 1:1)", True),
    ("items", "order_id", "orders", "order_id", "N:1", True),
    ("items", "product_id", "products", "product_id", "N:1", True),
    ("items", "seller_id", "sellers", "seller_id", "N:1", True),
    ("payments", "order_id", "orders", "order_id", "N:1", True),
    ("reviews", "order_id", "orders", "order_id", "N:1", False),
    ("products", "product_category_name", "translation", "product_category_name", "N:1 opcional", False),
    ("customers", "customer_zip_code_prefix", "geolocation", "geolocation_zip_code_prefix", "N:N bruto; N:1 após medianas", False),
    ("sellers", "seller_zip_code_prefix", "geolocation", "geolocation_zip_code_prefix", "N:N bruto; N:1 após medianas", False),
]
TIME_COLUMNS = {"order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date", "shipping_limit_date",
                "review_creation_date", "review_answer_timestamp"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def zip_integrity(root: Path) -> dict:
    paths = sorted((root / "dados/brutos").rglob("*.zip"))
    if not paths:
        raise FileNotFoundError("Nenhum ZIP em dados/brutos. Interrompido; nenhum dado foi fabricado.")
    result = {}
    for path in paths:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                raise ValueError(f"ZIP corrompido: {path.name}, membro {bad}")
            result[str(path.relative_to(root))] = {"sha256": sha256(path), "size_bytes": path.stat().st_size,
                                                    "crc_ok": True, "members": len(z.infolist())}
    return result


def find_source(root: Path) -> Path:
    candidates = []
    expected = set(TABLE_FILES.values())
    for path in sorted((root / "dados/brutos").rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            if expected.issubset({Path(name).name for name in z.namelist()}):
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Esperado um ZIP contendo os nove CSVs Olist; encontrados {len(candidates)}.")
    return candidates[0]


def load_tables(source: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables, inventory = {}, []
    by_name = {v: k for k, v in TABLE_FILES.items()}
    with zipfile.ZipFile(source) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if not info.filename.lower().endswith(".csv"):
                continue
            filename = Path(info.filename).name
            with z.open(info) as stream:
                header = pd.read_csv(stream, nrows=0).columns
            strings = {c: "string" for c in header if c.endswith("_id") or c.endswith("zip_code_prefix")
                       or c in ("order_status", "product_category_name", "product_category_name_english")}
            # O sequencial do item é numérico; order_item_id termina em _id.
            strings.pop("order_item_id", None)
            with z.open(info) as stream:
                frame = pd.read_csv(stream, dtype=strings)
            for col in frame:
                if col.endswith("zip_code_prefix"):
                    frame[col] = frame[col].str.zfill(5)
            name = by_name.get(filename, filename)
            if name in tables:
                raise ValueError(f"Nome de tabela ambíguo no ZIP: {name}")
            tables[name] = frame
            inventory.append({"table": name, "zip_member": info.filename, "rows": len(frame), "columns": len(frame.columns),
                              "exact_duplicate_rows": int(frame.duplicated().sum()), "uncompressed_bytes": info.file_size,
                              "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"})
    if not set(TABLE_FILES).issubset(tables):
        raise ValueError("Faltam tabelas obrigatórias no ZIP")
    return tables, pd.DataFrame(inventory)


def column_profile(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table, frame in tables.items():
        for col, series in frame.items():
            invalid = 0
            if col in TIME_COLUMNS:
                valid = parse_time(series)
                logical = "datetime (timezone não informado)"
                invalid = int((series.notna() & valid.isna()).sum())
            else:
                valid = series
                logical = "numeric" if pd.api.types.is_numeric_dtype(series) else "text"
                if col.endswith("zip_code_prefix"):
                    logical = "ZIP prefix (5 chars)"
                elif col.endswith("_id") and col != "order_item_id":
                    logical = "identifier"
            lo, hi = (valid.min(), valid.max()) if logical in ("numeric", "datetime (timezone não informado)") else (None, None)
            rows.append({"table": table, "column": col, "pandas_dtype": str(series.dtype), "logical_type": logical,
                         "rows": len(series), "missing": int(series.isna().sum()), "missing_pct": float(series.isna().mean() * 100),
                         "distinct_non_null": int(series.nunique()), "invalid_parsing": invalid, "min": lo, "max": hi})
    return pd.DataFrame(rows)


def key_and_relation_audit(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys, relations = [], []
    candidates = {**KEYS, "geolocation": ["geolocation_zip_code_prefix"], "reviews": ["review_id", "order_id"]}
    for name, columns in candidates.items():
        frame = tables[name]
        nulls = int(frame[columns].isna().any(axis=1).sum())
        duplicates = int(frame.duplicated(columns).sum())
        keys.append({"table": name, "key": " + ".join(columns), "rows": len(frame), "missing_key_rows": nulls,
                     "duplicate_key_rows": duplicates, "required_unique": name in KEYS})
        if name in KEYS and (duplicates or nulls):
            raise ValueError(f"Chave inválida: {name} {columns}")
    # Review_id não é uma PK; explicitar sua cardinalidade individual também.
    for col in ("review_id", "order_id"):
        frame = tables["reviews"]
        keys.append({"table": "reviews", "key": col, "rows": len(frame), "missing_key_rows": int(frame[col].isna().sum()),
                     "duplicate_key_rows": int(frame.duplicated(col).sum()), "required_unique": False})
    for child, fk, parent, pk, cardinality, required in RELATIONS:
        left, right = tables[child][fk], tables[parent][pk]
        unmatched = left.notna() & ~left.isin(right)
        counts = left.value_counts()
        relations.append({"child_table": child, "foreign_key": fk, "parent_table": parent, "parent_key": pk,
                          "cardinality": cardinality, "child_rows": len(left), "parent_distinct_keys": right.nunique(),
                          "missing_fk_rows": int(left.isna().sum()), "unmatched_rows": int(unmatched.sum()),
                          "max_child_rows_per_key": int(counts.max()), "required_match": required})
        if required and (unmatched.any() or left.isna().any()):
            raise ValueError(f"Relação obrigatória inválida: {child}.{fk} -> {parent}.{pk}")
    return pd.DataFrame(keys), pd.DataFrame(relations)


def timestamp_audit(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    times = {col: parse_time(orders[col]) for col in orders if col in TIME_COLUMNS}
    purchase, approved = times["order_purchase_timestamp"], times["order_approved_at"]
    carrier, delivery = times["order_delivered_carrier_date"], times["order_delivered_customer_date"]
    flags = pd.DataFrame({"order_id": orders.order_id, "order_status": orders.order_status})
    flags["delivered_missing_delivery"] = orders.order_status.eq("delivered") & delivery.isna()
    flags["not_delivered_with_delivery"] = ~orders.order_status.eq("delivered") & delivery.notna()
    flags["approval_before_purchase"] = approved.lt(purchase)
    flags["carrier_before_purchase"] = carrier.lt(purchase)
    flags["delivery_before_purchase"] = delivery.lt(purchase)
    flags["carrier_before_approval"] = carrier.lt(approved)
    flags["delivery_before_carrier"] = delivery.lt(carrier)
    flags["delivered_missing_approval"] = orders.order_status.eq("delivered") & approved.isna()
    flags["delivered_missing_carrier"] = orders.order_status.eq("delivered") & carrier.isna()
    for col, parsed in times.items():
        flags[f"invalid_parse_{col}"] = orders[col].notna() & parsed.isna()
    count = {col: int(flags[col].sum()) for col in flags if col not in ("order_id", "order_status")}
    return flags.loc[flags.drop(columns=["order_id", "order_status"]).any(axis=1)].copy(), count


def missing_profile(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": frame.columns, "missing": frame.isna().sum().values,
                         "missing_pct": frame.isna().mean().values * 100,
                         "distinct_non_null": frame.nunique().values})


def outlier_profile(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IQR apenas diagnóstico; não aprende transformações nem remove observações."""
    rows, flags = [], pd.DataFrame({"order_id": frame.order_id})
    numeric = frame.select_dtypes(include="number")
    for col in numeric:
        values = numeric[col].astype(float)
        valid = values.dropna()
        if not len(valid):
            continue
        q1, median, q3 = valid.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = values.lt(low) | values.gt(high)
        rows.append({"column": col, "non_null": len(valid), "min": valid.min(), "q1": q1, "median": median,
                     "q3": q3, "p99": valid.quantile(0.99), "max": valid.max(), "iqr_low": low, "iqr_high": high,
                     "flagged_iqr": int(flagged.sum()), "infinite_count": int(np.isinf(values).sum()),
                     "interpretation": "triagem; IQR=0/comportamento discreto pode marcar valores legítimos"})
        flags[f"iqr_{col}"] = flagged
    flags = flags.loc[flags.drop(columns="order_id").any(axis=1)].copy()
    return pd.DataFrame(rows), flags
