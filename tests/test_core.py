from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from olist_features.audit import find_source, zip_integrity
from olist_features.core import (FEATURE_COLUMNS, TARGET, build_features, geo_centroids,
                                 haversine_km, payment_audit, select_target, stable_mode)


def test_haversine_reference_symmetry_missing():
    # Arco de um grau no equador é ~111,195 km; ponto idêntico deve ser zero.
    np.testing.assert_allclose(haversine_km([0, 0], [0, 0], [0, 0], [1, 0]), [111.19508, 0], atol=1e-5)
    sp_rio = haversine_km(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 350 < sp_rio < 370
    assert sp_rio == pytest.approx(haversine_km(-22.9068, -43.1729, -23.5505, -46.6333))
    assert np.isnan(haversine_km(np.nan, 0, 0, 1))
    assert np.isfinite(haversine_km(0, 0, 0, 180))


def test_multi_items_sellers_without_multiplication(tiny_tables):
    features, _, _ = build_features(tiny_tables)
    one = features.set_index("order_id").loc["o1"]
    assert len(features) == 2
    assert one.products_value_total == 45
    assert one.freight_value_total == 4
    assert one.order_value_total == 49
    assert one.items_count == 3
    assert one.products_count == 2
    assert one.sellers_count == 2
    assert one.weight_g_total == 400
    assert one.volume_cm3_total == 600
    assert one.main_category == "a"
    assert one.same_state_items_fraction == pytest.approx(2 / 3)
    d1, d2 = haversine_km(-23, -46, -22, -45), haversine_km(-23, -46, -21, -44)
    assert one.distance_km_mean == pytest.approx((2 * d1 + d2) / 3)
    assert one.distance_km_max == pytest.approx(d2)


def test_payments_aggregate_separately(tiny_tables):
    features, _, _ = build_features(tiny_tables)
    audit = payment_audit(tiny_tables["payments"], features).set_index("order_id")
    assert audit.loc["o1", "payment_records"] == 2
    assert audit.loc["o1", "payment_value_total"] == 49
    assert audit.loc["o1", "payment_minus_order_value"] == 0
    assert list(features.columns) == ["order_id", *FEATURE_COLUMNS]
    assert not any("payment" in col for col in FEATURE_COLUMNS)


def test_missing_distance_keeps_order(tiny_tables):
    features, _, _ = build_features(tiny_tables)
    second = features.set_index("order_id").loc["o2"]
    assert np.isnan(second.distance_km_mean)
    assert second.items_missing_distance_count == 1
    target, _, _ = select_target(tiny_tables["orders"], features)
    assert len(target) == 2


def test_partial_distance_has_counter(tiny_tables):
    tiny_tables["sellers"].loc[1, "seller_zip_code_prefix"] = "88888"
    features, _, _ = build_features(tiny_tables)
    first = features.set_index("order_id").loc["o1"]
    assert first.items_missing_distance_count == 1
    assert first.distance_km_mean == pytest.approx(haversine_km(-23, -46, -22, -45))


def test_invalid_measures_do_not_make_partial_totals(tiny_tables):
    tiny_tables["products"].loc[1, "product_weight_g"] = 0
    tiny_tables["products"].loc[1, "product_height_cm"] = np.nan
    tiny_tables["items"].loc[2, "price"] = -1
    features, _, info = build_features(tiny_tables)
    first = features.set_index("order_id").loc["o1"]
    assert np.isnan(first.weight_g_total)
    assert np.isnan(first.volume_cm3_total)
    assert np.isnan(first.products_value_total)
    assert np.isnan(first.order_value_total)
    assert first.weight_g_mean == 100
    assert first.items_missing_weight_count == 1
    assert first.items_missing_dimensions_count == 1
    assert first.items_missing_price_count == 1
    assert info["invalid_numeric_item_values"]["price"] == 1


def test_zero_price_and_freight_are_preserved(tiny_tables):
    tiny_tables["items"].loc[:, ["price", "freight_value"]] = 0
    features, _, _ = build_features(tiny_tables)
    assert features.order_value_total.eq(0).all()


def test_calendar_and_fractional_target(tiny_tables):
    features, _, _ = build_features(tiny_tables)
    target, audit, _ = select_target(tiny_tables["orders"], features)
    assert target.set_index("order_id").loc["o1", TARGET] == 1.5
    assert features.set_index("order_id").loc["o1", "purchase_weekday"] == 0
    assert features.set_index("order_id").loc["o2", "purchase_is_weekend"] == 1
    assert audit.selection_reason.eq("included").all()


def test_target_exclusions_and_zero_duration():
    orders = pd.DataFrame({"order_id": ["a", "b", "c", "d", "e", "f", "g"],
                           "order_status": ["canceled", "delivered", "delivered", "delivered", "delivered", "delivered", "delivered"],
                           "order_purchase_timestamp": ["2018-01-01 00:00:00", "bad", "2018-01-01 00:00:00", "2018-01-02 00:00:00", "2018-01-01 00:00:00", "2018-01-01 00:00:00", "2018-01-01 00:00:00"],
                           "order_delivered_customer_date": [None, "2018-01-03 00:00:00", None, "2018-01-01 00:00:00", "2018-01-01 00:00:00", "2018-01-02 00:00:00", "2018-12-31 00:00:00"]}, index=[9, 8, 7, 6, 5, 4, 3])
    features = pd.DataFrame({"order_id": orders.order_id.tolist(), "items_count": [1, 1, 1, 1, 1, 0, 1]})
    target, audit, coverage = select_target(orders, features)
    assert target.order_id.tolist() == ["e", "g"]
    assert target[TARGET].tolist() == [0.0, 364.0]
    assert coverage.excluded_at_stage.sum() == 5
    assert audit.set_index("order_id").loc["d", "selection_reason"] == "negative_duration"
    assert audit.set_index("order_id").loc["f", "selection_reason"] == "no_items"


def test_no_leakage_when_future_fields_changed_or_absent(tiny_tables):
    baseline, _, _ = build_features(tiny_tables)
    changed = deepcopy(tiny_tables)
    forbidden = [c for c in changed["orders"] if c not in ("order_id", "customer_id", "order_purchase_timestamp")]
    for col in forbidden:
        changed["orders"][col] = "changed future value"
    changed["items"]["shipping_limit_date"] = "2099-01-01 00:00:00"
    changed["payments"] = pd.concat([changed["payments"]] * 5, ignore_index=True)
    changed["reviews"]["review_score"] = 5
    same, _, _ = build_features(changed)
    pd.testing.assert_frame_equal(baseline, same)
    changed["orders"] = changed["orders"].drop(columns=forbidden)
    changed["items"] = changed["items"].drop(columns="shipping_limit_date")
    del changed["payments"], changed["reviews"]
    same, _, _ = build_features(changed)
    pd.testing.assert_frame_equal(baseline, same)


def test_order_and_input_frames_not_mutated(tiny_tables):
    original = deepcopy(tiny_tables)
    a, _, _ = build_features(tiny_tables)
    shuffled = {name: frame.sample(frac=1, random_state=42).reset_index(drop=True) for name, frame in tiny_tables.items()}
    b, _, _ = build_features(shuffled)
    pd.testing.assert_frame_equal(a, b)
    for name in tiny_tables:
        pd.testing.assert_frame_equal(tiny_tables[name], original[name])


@pytest.mark.parametrize("table,key", [("orders", "order_id"), ("customers", "customer_id"),
                                       ("products", "product_id"), ("sellers", "seller_id"), ("items", "order_id")])
def test_duplicate_keys_fail(tiny_tables, table, key):
    tiny_tables[table] = pd.concat([tiny_tables[table], tiny_tables[table].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Chave inválida"):
        build_features(tiny_tables)


def test_duplicate_payment_key_fails(tiny_tables):
    features, _, _ = build_features(tiny_tables)
    tiny_tables["payments"] = pd.concat([tiny_tables["payments"], tiny_tables["payments"].iloc[[0]]])
    with pytest.raises(ValueError, match="Chave inválida"):
        payment_audit(tiny_tables["payments"], features)


def test_geo_invalid_dedup_and_median(tiny_tables):
    geo = tiny_tables["geolocation"]
    extra = geo.iloc[[0, 0, 0]].copy()
    extra["geolocation_lat"] = [-25.0, 120.0, 40.0]
    combined = pd.concat([geo, extra], ignore_index=True)
    centroids, info = geo_centroids(combined)
    assert centroids.set_index("geolocation_zip_code_prefix").loc["01000", "lat"] == -24.0
    assert info["exact_duplicates_removed"] == 1
    assert info["globally_invalid_unique_rows"] == 1
    assert info["outside_brazil_box_unique_rows"] == 1


def test_category_tie_and_missing():
    assert stable_mode(pd.Series(["b", "a"])) == "a"
    assert pd.isna(stable_mode(pd.Series([None, None])))


def test_zip_absent_stops_without_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="nenhum dado foi fabricado"):
        zip_integrity(tmp_path)
    assert not (tmp_path / "dados/tratados").exists()
    with pytest.raises(ValueError, match="encontrados 0"):
        find_source(tmp_path)


@pytest.mark.parametrize("rounding_error,should_pass", [(1e-12, True), (1.0, False)])
def test_distance_validation_tolerance(tiny_tables, rounding_error, should_pass):
    from olist_features.pipeline import independent_checks

    features, centroids, _ = build_features(tiny_tables)
    labels, _, _ = select_target(tiny_tables["orders"], features)
    dataset = features.merge(labels, on="order_id", validate="one_to_one")
    dataset.loc[0, "distance_km_mean"] = dataset.loc[0, "distance_km_max"] + rounding_error
    if should_pass:
        assert independent_checks(tiny_tables, features, dataset, centroids, labels).passed.all()
    else:
        with pytest.raises(AssertionError, match="distance_order_bounds"):
            independent_checks(tiny_tables, features, dataset, centroids, labels)


def test_corrupt_zip_is_rejected(tmp_path):
    from zipfile import BadZipFile, ZIP_STORED, ZipFile

    folder = tmp_path / "dados/brutos"
    folder.mkdir(parents=True)
    path = folder / "corrupt.zip"
    with ZipFile(path, "w", compression=ZIP_STORED) as z:
        z.writestr("test.csv", "a,b\n123,456\n")
    content = path.read_bytes().replace(b"123,456", b"999,456")
    path.write_bytes(content)
    with pytest.raises((ValueError, BadZipFile)):
        zip_integrity(tmp_path)
