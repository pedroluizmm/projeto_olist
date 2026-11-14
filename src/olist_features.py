"""Leitura do dataset Olist e construção do dataset de regressão (uma linha por pedido).

Usado pelos notebooks 01, 02 e 03. Alvo: dias entre a compra e o recebimento.
Todas as features usam apenas informações disponíveis no momento da compra.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "dados/brutos/brazilian-ecommerce.zip"
DATASET_PATH = ROOT / "dados/tratados/olist_dataset.parquet"

TARGET = "delivery_time_days"
MAX_DELIVERY_DAYS = 60  # acima disso: extravio/anomalia, tratado como ruído
EARTH_RADIUS_KM = 6371.0088

TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

# nome -> (tabela de origem, descrição)
FEATURE_SPECS = {
    "products_value_total": ("order_items", "Soma dos preços dos itens (R$)"),
    "freight_value_total": ("order_items", "Soma do frete dos itens (R$)"),
    "order_value_total": ("order_items", "Produtos + frete (R$)"),
    "items_count": ("order_items", "Número de itens do pedido"),
    "products_count": ("order_items", "Produtos distintos no pedido"),
    "sellers_count": ("order_items", "Vendedores distintos no pedido"),
    "seller_states_count": ("sellers", "UFs distintas dos vendedores"),
    "same_state_items_fraction": ("sellers + customers", "Fração de itens com vendedor na UF do cliente"),
    "weight_g_total": ("products", "Peso total dos itens (g)"),
    "weight_g_mean": ("products", "Peso médio por item (g)"),
    "volume_cm3_total": ("products", "Volume total: comprimento × altura × largura (cm³)"),
    "length_cm_max": ("products", "Maior comprimento entre os itens (cm)"),
    "height_cm_max": ("products", "Maior altura entre os itens (cm)"),
    "width_cm_max": ("products", "Maior largura entre os itens (cm)"),
    "items_missing_weight_count": ("products", "Itens sem peso válido"),
    "items_missing_dimensions_count": ("products", "Itens sem dimensões válidas"),
    "items_missing_price_count": ("order_items", "Itens sem preço válido"),
    "items_missing_freight_count": ("order_items", "Itens sem frete válido"),
    "categories_count": ("products", "Categorias distintas no pedido"),
    "main_category": ("products", "Categoria mais frequente do pedido"),
    "items_missing_category_count": ("products", "Itens sem categoria"),
    "customer_state": ("customers", "UF do cliente"),
    "distance_km_mean": ("geolocation", "Distância média vendedor→cliente (Haversine, km)"),
    "distance_km_min": ("geolocation", "Menor distância vendedor→cliente (km)"),
    "distance_km_max": ("geolocation", "Maior distância vendedor→cliente (km)"),
    "items_missing_distance_count": ("geolocation", "Itens sem coordenadas válidas"),
    "purchase_weekday": ("orders", "Dia da semana da compra (0=seg … 6=dom)"),
    "purchase_month": ("orders", "Mês da compra (1–12)"),
    "purchase_hour": ("orders", "Hora da compra (0–23)"),
    "purchase_is_weekend": ("orders", "Compra no sábado/domingo (1/0)"),
}
FEATURES = list(FEATURE_SPECS)
CATEGORICAL = ["customer_state", "main_category", "purchase_weekday", "purchase_month"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# Colunas auxiliares para EDA/benchmark (NÃO entram no modelo)
AUX = ["order_id", "purchase_timestamp", "customer_lat", "customer_lon", "olist_estimated_days"]


def load_tables(zip_path: Path = ZIP_PATH) -> dict[str, pd.DataFrame]:
    """Lê os 9 CSVs do ZIP original. IDs e CEPs como texto (CEP com 5 dígitos)."""
    tables = {}
    with zipfile.ZipFile(zip_path) as z:
        for name, filename in TABLE_FILES.items():
            with z.open(filename) as f:
                header = pd.read_csv(f, nrows=0).columns
            dtypes = {c: "string" for c in header
                      if (c.endswith("_id") and c != "order_item_id") or c.endswith("zip_code_prefix")}
            with z.open(filename) as f:
                df = pd.read_csv(f, dtype=dtypes)
            for col in df.columns:
                if col.endswith("zip_code_prefix"):
                    df[col] = df[col].str.zfill(5)
            tables[name] = df
    return tables


def haversine_km(lat1, lon1, lat2, lon2):
    a1, o1, a2, o2 = [np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)]
    a = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geo_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    """Uma coordenada por prefixo de CEP (mediana), descartando pontos fora do Brasil."""
    geo = geo.drop_duplicates()
    inside = geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -28)
    return (geo[inside].groupby("geolocation_zip_code_prefix")
            .agg(lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median")).reset_index())


def _strict_sum(s: pd.Series):
    """Soma só se todos os itens têm valor (evita total parcial)."""
    return s.sum(min_count=len(s))


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def build_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Agrega itens por pedido e junta produtos, vendedores, clientes e geolocalização."""
    cent = geo_centroids(t["geolocation"])
    customers = t["customers"].merge(
        cent.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                             "lat": "customer_lat", "lon": "customer_lon"}),
        on="customer_zip_code_prefix", how="left", validate="many_to_one")
    sellers = t["sellers"].merge(
        cent.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                             "lat": "seller_lat", "lon": "seller_lon"}),
        on="seller_zip_code_prefix", how="left", validate="many_to_one")
    orders = t["orders"].merge(customers, on="customer_id", how="left", validate="many_to_one")

    it = (t["items"].merge(t["products"], on="product_id", how="left", validate="many_to_one")
          .merge(sellers, on="seller_id", how="left", validate="many_to_one")
          .merge(orders[["order_id", "customer_state", "customer_lat", "customer_lon"]],
                 on="order_id", how="left", validate="many_to_one"))
    assert len(it) == len(t["items"]), "junção multiplicou itens"

    # Limpeza de medidas: negativos/zeros/infinitos viram ausentes
    for col in ["price", "freight_value"]:
        it[col] = it[col].where(it[col] >= 0)
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        it[col] = it[col].where(it[col] > 0)
    dims = ["product_length_cm", "product_height_cm", "product_width_cm"]
    it["volume_cm3"] = it[dims[0]] * it[dims[1]] * it[dims[2]]
    it["miss_weight"] = it.product_weight_g.isna().astype(int)
    it["miss_dims"] = it[dims].isna().any(axis=1).astype(int)
    it["miss_price"] = it.price.isna().astype(int)
    it["miss_freight"] = it.freight_value.isna().astype(int)
    it["miss_cat"] = it.product_category_name.isna().astype(int)
    it["distance_km"] = haversine_km(it.customer_lat, it.customer_lon, it.seller_lat, it.seller_lon)
    it["miss_dist"] = it.distance_km.isna().astype(int)
    it["same_state"] = (it.customer_state == it.seller_state).astype(float)

    agg = it.groupby("order_id").agg(
        products_value_total=("price", _strict_sum), freight_value_total=("freight_value", _strict_sum),
        items_count=("order_item_id", "size"), products_count=("product_id", "nunique"),
        sellers_count=("seller_id", "nunique"), seller_states_count=("seller_state", "nunique"),
        same_state_items_fraction=("same_state", "mean"),
        weight_g_total=("product_weight_g", _strict_sum), weight_g_mean=("product_weight_g", "mean"),
        volume_cm3_total=("volume_cm3", _strict_sum), length_cm_max=("product_length_cm", "max"),
        height_cm_max=("product_height_cm", "max"), width_cm_max=("product_width_cm", "max"),
        items_missing_weight_count=("miss_weight", "sum"), items_missing_dimensions_count=("miss_dims", "sum"),
        items_missing_price_count=("miss_price", "sum"), items_missing_freight_count=("miss_freight", "sum"),
        categories_count=("product_category_name", "nunique"), main_category=("product_category_name", _mode),
        items_missing_category_count=("miss_cat", "sum"),
        distance_km_mean=("distance_km", "mean"), distance_km_min=("distance_km", "min"),
        distance_km_max=("distance_km", "max"), items_missing_distance_count=("miss_dist", "sum"),
    ).reset_index()
    agg["order_value_total"] = agg.products_value_total + agg.freight_value_total

    df = orders.merge(agg, on="order_id", how="left", validate="one_to_one")
    purchase = pd.to_datetime(df.order_purchase_timestamp)
    df["purchase_timestamp"] = purchase
    df["purchase_weekday"] = purchase.dt.dayofweek
    df["purchase_month"] = purchase.dt.month
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_is_weekend"] = (df.purchase_weekday >= 5).astype(int)
    delivered = pd.to_datetime(df.order_delivered_customer_date)
    estimated = pd.to_datetime(df.order_estimated_delivery_date)
    df[TARGET] = (delivered - purchase).dt.total_seconds() / 86400
    df["olist_estimated_days"] = (estimated - purchase).dt.total_seconds() / 86400
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtragem sequencial; devolve o dataset limpo e o funil de exclusões."""
    steps = [
        ("Pedidos originais", pd.Series(True, index=df.index)),
        ("Status 'delivered'", df.order_status.eq("delivered")),
        ("Com data de entrega", df[TARGET].notna()),
        ("Com itens", df.items_count.fillna(0) > 0),
        ("Tempo de entrega > 0", df[TARGET] > 0),
        (f"Entrega ≤ {MAX_DELIVERY_DAYS} dias (remove ruído)", df[TARGET] <= MAX_DELIVERY_DAYS),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = []
    for name, mask in steps:
        before = int(keep.sum())
        keep &= mask
        funnel.append({"etapa": name, "pedidos_restantes": int(keep.sum()), "removidos": before - int(keep.sum())})
    out = df.loc[keep, AUX + FEATURES + [TARGET]].reset_index(drop=True)
    for col in CATEGORICAL:  # texto; ausente = np.nan (compatível com SimpleImputer)
        values = out[col].astype("string")
        out[col] = values.astype(object).where(values.notna(), np.nan)
    out[NUMERIC] = out[NUMERIC].astype(float)
    return out, pd.DataFrame(funnel)


def feature_summary() -> pd.DataFrame:
    return pd.DataFrame([{"feature": k, "tipo": "categórica" if k in CATEGORICAL else "numérica",
                          "origem": v[0], "descrição": v[1]} for k, v in FEATURE_SPECS.items()])


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH)
