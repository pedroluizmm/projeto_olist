import json
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from olist_features.audit import key_and_relation_audit, sha256, zip_integrity
from olist_features.core import FEATURE_COLUMNS, TARGET, parse_time, select_target


def test_real_one_row_per_order_and_allowlist(real_dataset, project_root):
    assert real_dataset.order_id.is_unique
    assert real_dataset.order_id.notna().all()
    assert list(real_dataset.columns) == ["order_id", *FEATURE_COLUMNS, TARGET]
    contract = json.loads((project_root / "dados/tratados/feature_contract.json").read_text(encoding="utf-8"))
    assert contract["feature_columns"] == FEATURE_COLUMNS
    assert TARGET not in FEATURE_COLUMNS
    assert not any(token in col for token in ("approved", "delivered", "review", "payment", "estimated", "shipping_limit") for col in FEATURE_COLUMNS)


def test_real_coverage_and_dates(real_tables, real_dataset, project_root):
    features = pd.read_parquet(project_root / "dados/tratados/purchase_features_all_orders.parquet")
    labels, selection, coverage = select_target(real_tables["orders"], features)
    assert_frame_equal(labels, real_dataset[["order_id", TARGET]])
    assert len(real_dataset) == 96470  # Contagem da inspeção da fonte identificada pelo hash.
    assert len(features) == len(real_tables["orders"]) == 99441
    assert coverage.excluded_at_stage.sum() + len(labels) == len(features)
    assert len(selection) == len(features)
    assert real_dataset[TARGET].notna().all()
    assert real_dataset[TARGET].ge(0).all()
    assert real_dataset[TARGET].max() > 100  # Extremos legítimos não foram aparados.


def test_real_keys_and_foreign_keys(real_tables):
    keys, relations = key_and_relation_audit(real_tables)
    required = keys.loc[keys.required_unique]
    assert required.missing_key_rows.eq(0).all()
    assert required.duplicate_key_rows.eq(0).all()
    assert relations.loc[relations.required_match, "unmatched_rows"].eq(0).all()


def test_real_aggregates_multiple_items_and_payments(real_tables, real_dataset):
    items = real_tables["items"]
    expected = items.groupby("order_id").agg(products_value_total=("price", "sum"),
                                             freight_value_total=("freight_value", "sum"),
                                             items_count=("order_item_id", "size"),
                                             sellers_count=("seller_id", "nunique"))
    actual = real_dataset.set_index("order_id")
    expected = expected.reindex(actual.index)
    for col in expected:
        np.testing.assert_allclose(expected[col], actual[col])
    multi_payment_ids = real_tables["payments"].groupby("order_id").size().loc[lambda s: s > 1].index
    multi_item_ids = expected.index[expected.items_count.gt(1)]
    intersection = actual.index.intersection(multi_payment_ids).intersection(multi_item_ids)
    assert len(intersection) > 0
    np.testing.assert_allclose(actual.loc[intersection, "products_value_total"], expected.loc[intersection, "products_value_total"])


def test_real_target_independent_datetime(real_tables, real_dataset):
    orders = real_tables["orders"].set_index("order_id")
    # Casos determinísticos distribuídos pelo dataset, incluindo o maior tempo.
    sample = pd.concat([real_dataset.iloc[::997], real_dataset.nlargest(1, TARGET)])
    for row in sample.itertuples():
        raw = orders.loc[row.order_id]
        expected = (datetime.strptime(raw.order_delivered_customer_date, "%Y-%m-%d %H:%M:%S") -
                    datetime.strptime(raw.order_purchase_timestamp, "%Y-%m-%d %H:%M:%S")).total_seconds() / 86400
        assert abs(getattr(row, TARGET) - expected) < 1e-12
        assert raw.order_status == "delivered"


def test_real_zip_unchanged_and_output_hashes(project_root):
    manifest = json.loads((project_root / "resultados/tabelas/run_manifest.json").read_text(encoding="utf-8"))
    current = zip_integrity(project_root)
    assert current == manifest["input_integrity_before"] == manifest["input_integrity_after"]
    assert next(iter(current.values()))["sha256"] == "967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784"
    for relative, expected in manifest["output_sha256"].items():
        assert sha256(project_root / relative) == expected, relative


def test_real_csv_parquet_agree(project_root, real_dataset):
    csv = pd.read_csv(project_root / "dados/tratados/olist_orders_features.csv", dtype={"order_id": "string"})
    assert csv.order_id.tolist() == real_dataset.order_id.tolist()
    for col in real_dataset.select_dtypes(include="number"):
        np.testing.assert_allclose(csv[col], real_dataset[col].astype(float), equal_nan=True, rtol=1e-10, atol=1e-8)
    assert csv.customer_state.tolist() == real_dataset.customer_state.tolist()


def test_real_missing_and_bounds_consistent(project_root, real_dataset):
    missing = pd.read_csv(project_root / "resultados/tabelas/missing_final.csv").set_index("column")
    for col in real_dataset:
        assert missing.loc[col, "missing"] == real_dataset[col].isna().sum()
    assert real_dataset.items_missing_distance_count.ge(0).all()
    assert real_dataset.items_missing_distance_count.le(real_dataset.items_count).all()
    complete = real_dataset.items_missing_distance_count.eq(0)
    assert real_dataset.loc[complete, "distance_km_mean"].notna().all()
    assert real_dataset.distance_km_max.dropna().between(0, 20016).all()


def test_real_verification_table(project_root):
    checks = pd.read_csv(project_root / "resultados/tabelas/verification_checks.csv")
    assert len(checks) >= 15
    assert checks.passed.all()
    assert "target_datetime_independent" in checks.check.to_list()


def test_real_distances_independent_raw_relationships(project_root, real_tables, real_dataset):
    import math
    from olist_features.core import EARTH_RADIUS_KM

    centers = pd.read_parquet(project_root / "resultados/tabelas/geolocation_centroids.parquet").set_index("geolocation_zip_code_prefix")
    orders = real_tables["orders"].set_index("order_id")
    buyers = real_tables["customers"].set_index("customer_id")
    sellers = real_tables["sellers"].set_index("seller_id")
    items = real_tables["items"].groupby("order_id")
    sample = pd.concat([real_dataset.iloc[::997], real_dataset.loc[real_dataset.sellers_count.gt(1)].head(15)])
    for row in sample.itertuples():
        customer = buyers.loc[orders.loc[row.order_id, "customer_id"]]
        buyer_zip = customer.customer_zip_code_prefix
        distances = []
        missing = 0
        for item in items.get_group(row.order_id).itertuples():
            seller_zip = sellers.loc[item.seller_id, "seller_zip_code_prefix"]
            if buyer_zip not in centers.index or seller_zip not in centers.index:
                missing += 1
                continue
            a, b = centers.loc[buyer_zip], centers.loc[seller_zip]
            lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
            cos_angle = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            distances.append(EARTH_RADIUS_KM * math.acos(max(-1, min(1, cos_angle))))
        assert row.items_missing_distance_count == missing
        if distances:
            np.testing.assert_allclose([row.distance_km_min, row.distance_km_mean, row.distance_km_max],
                                       [min(distances), sum(distances) / len(distances), max(distances)], atol=0.001, rtol=0)
        else:
            assert np.isnan(row.distance_km_mean)
