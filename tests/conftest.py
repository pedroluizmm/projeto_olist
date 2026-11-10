from pathlib import Path

import pandas as pd
import pytest

from olist_features.audit import load_tables


@pytest.fixture
def tiny_tables():
    """Dados artificiais exclusivos de testes; nunca são exportados como resultados."""
    return {
        "orders": pd.DataFrame({
            "order_id": ["o1", "o2"], "customer_id": ["c1", "c2"],
            "order_purchase_timestamp": ["2018-01-01 12:00:00", "2018-01-06 01:30:00"],
            "order_status": ["delivered", "delivered"],
            "order_delivered_customer_date": ["2018-01-03 00:00:00", "2018-01-07 01:30:00"],
            "order_approved_at": ["2018-01-01 13:00:00", "2018-01-06 02:00:00"],
            "order_delivered_carrier_date": ["2018-01-02 00:00:00", "2018-01-06 03:00:00"],
            "order_estimated_delivery_date": ["2018-01-05 00:00:00", "2018-01-08 00:00:00"],
        }),
        "customers": pd.DataFrame({"customer_id": ["c1", "c2"], "customer_zip_code_prefix": ["01000", "99999"],
                                   "customer_state": ["SP", "CE"]}),
        "sellers": pd.DataFrame({"seller_id": ["s1", "s2"], "seller_zip_code_prefix": ["02000", "03000"],
                                 "seller_state": ["SP", "RJ"]}),
        "products": pd.DataFrame({"product_id": ["p1", "p2"], "product_category_name": ["a", "b"],
                                  "product_weight_g": [100.0, 200.0], "product_length_cm": [10.0, 20.0],
                                  "product_height_cm": [2.0, 4.0], "product_width_cm": [3.0, 6.0]}),
        "items": pd.DataFrame({"order_id": ["o1", "o1", "o1", "o2"], "order_item_id": [1, 2, 3, 1],
                               "product_id": ["p1", "p1", "p2", "p2"], "seller_id": ["s1", "s1", "s2", "s2"],
                               "price": [10.0, 10.0, 25.0, 25.0], "freight_value": [1.0, 1.0, 2.0, 2.0],
                               "shipping_limit_date": ["2018-01-02 00:00:00"] * 4}),
        "geolocation": pd.DataFrame({"geolocation_zip_code_prefix": ["01000", "01000", "02000", "03000"],
                                     "geolocation_lat": [-23.0, -23.0, -22.0, -21.0],
                                     "geolocation_lng": [-46.0, -46.0, -45.0, -44.0],
                                     "geolocation_city": ["a", "a", "b", "c"],
                                     "geolocation_state": ["SP", "SP", "SP", "RJ"]}),
        "payments": pd.DataFrame({"order_id": ["o1", "o1", "o2"], "payment_sequential": [1, 2, 1],
                                  "payment_type": ["credit_card", "voucher", "credit_card"],
                                  "payment_installments": [1, 1, 2], "payment_value": [40.0, 9.0, 27.0]}),
        "reviews": pd.DataFrame({"order_id": ["o1"], "review_score": [1]}),
    }


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def real_tables(project_root):
    return load_tables(project_root / "dados/brutos/brazilian-ecommerce.zip")[0]


@pytest.fixture(scope="session")
def real_dataset(project_root):
    path = project_root / "dados/tratados/olist_orders_features.parquet"
    assert path.exists(), "Execute o pipeline real antes dos testes de integração"
    return pd.read_parquet(path)
