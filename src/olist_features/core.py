"""Transformações determinísticas. Nenhuma estatística do alvo é usada em X."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "delivery_time_days"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EARTH_RADIUS_KM = 6371.0088

# Este contrato é também a lista exclusiva de entradas da futura modelagem.
# order_id e timestamps ficam em artefatos separados, nunca em X.
FEATURE_SPECS = {
    "products_value_total": ("items.price", "Soma dos preços dos itens; BRL."),
    "freight_value_total": ("items.freight_value", "Soma do frete dos itens; BRL."),
    "order_value_total": ("items.price, items.freight_value", "Produtos + frete; não usa pagamentos."),
    "items_count": ("items.order_item_id", "Número de linhas de itens, não máximo do sequencial."),
    "products_count": ("items.product_id", "Quantidade de produtos distintos no carrinho."),
    "sellers_count": ("items.seller_id", "Quantidade de vendedores distintos."),
    "seller_states_count": ("sellers.seller_state", "Quantidade de UFs dos vendedores."),
    "same_state_items_fraction": ("customers.customer_state, sellers.seller_state", "Fração de itens com vendedor na UF do comprador."),
    "weight_g_total": ("products.product_weight_g", "Soma por item; ausente se houver peso ausente/inválido."),
    "weight_g_mean": ("products.product_weight_g", "Média dos pesos conhecidos por item."),
    "volume_cm3_total": ("products.product_*_cm", "Soma de comprimento × altura × largura; exige todos os itens."),
    "length_cm_max": ("products.product_length_cm", "Maior comprimento conhecido entre itens."),
    "height_cm_max": ("products.product_height_cm", "Maior altura conhecida entre itens."),
    "width_cm_max": ("products.product_width_cm", "Maior largura conhecida entre itens."),
    "items_missing_weight_count": ("products.product_weight_g", "Itens com peso ausente ou não positivo."),
    "items_missing_dimensions_count": ("products.product_*_cm", "Itens com alguma dimensão ausente ou não positiva."),
    "items_missing_price_count": ("items.price", "Itens com preço ausente, infinito ou negativo."),
    "items_missing_freight_count": ("items.freight_value", "Itens com frete ausente, infinito ou negativo."),
    "categories_count": ("products.product_category_name", "Número de categorias conhecidas distintas."),
    "main_category": ("products.product_category_name", "Moda por item; empate resolvido em ordem alfabética."),
    "items_missing_category_count": ("products.product_category_name", "Itens sem categoria."),
    "customer_state": ("customers.customer_state", "UF do endereço do pedido."),
    "distance_km_mean": ("customers/sellers ZIP, geolocation", "Haversine médio por item; apenas pares conhecidos."),
    "distance_km_min": ("customers/sellers ZIP, geolocation", "Menor distância conhecida entre itens."),
    "distance_km_max": ("customers/sellers ZIP, geolocation", "Maior distância conhecida entre itens."),
    "items_missing_distance_count": ("customers/sellers ZIP, geolocation", "Itens sem par de coordenadas válido."),
    "purchase_weekday": ("orders.order_purchase_timestamp", "Dia da semana: segunda=0, domingo=6."),
    "purchase_month": ("orders.order_purchase_timestamp", "Mês da compra: 1 a 12."),
    "purchase_hour": ("orders.order_purchase_timestamp", "Hora da compra: 0 a 23."),
    "purchase_is_weekend": ("orders.order_purchase_timestamp", "Sábado/domingo=1; demais=0."),
}
FEATURE_COLUMNS = list(FEATURE_SPECS)

ORDER_INPUTS = ["order_id", "customer_id", "order_purchase_timestamp"]
ITEM_INPUTS = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]
PRODUCT_INPUTS = ["product_id", "product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
CUSTOMER_INPUTS = ["customer_id", "customer_zip_code_prefix", "customer_state"]
SELLER_INPUTS = ["seller_id", "seller_zip_code_prefix", "seller_state"]


def parse_time(values: pd.Series) -> pd.Series:
    """O dump não informa timezone: preservar os horários locais, sem conversão."""
    return pd.to_datetime(values, format=TIME_FORMAT, errors="coerce")


def assert_key(df: pd.DataFrame, columns: list[str], name: str) -> None:
    if df[columns].isna().any().any() or df.duplicated(columns).any():
        raise ValueError(f"Chave inválida em {name}: {columns}")


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância esférica, em km; NaNs propagam. Inputs em graus."""
    a1, o1, a2, o2 = [np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)]
    a = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geo_centroids(geo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplica linhas idênticas; medianas marginais por prefixo de CEP."""
    clean = geo.drop_duplicates().copy()
    lat = pd.to_numeric(clean.geolocation_lat, errors="coerce")
    lng = pd.to_numeric(clean.geolocation_lng, errors="coerce")
    global_valid = lat.between(-90, 90) & lng.between(-180, 180)
    # Caixa ampla, não um teste de pertencimento ao território brasileiro.
    brazil_box = lat.between(-34, 6) & lng.between(-74, -28)
    valid_zip = clean.geolocation_zip_code_prefix.str.fullmatch(r"\d{5}", na=False)
    valid = global_valid & brazil_box & valid_zip
    clean["geolocation_lat"] = lat
    clean["geolocation_lng"] = lng
    centroids = clean.loc[valid].groupby("geolocation_zip_code_prefix", sort=True).agg(
        lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median"),
        geo_observations=("geolocation_lat", "size"),
    ).reset_index()
    assert_key(centroids, ["geolocation_zip_code_prefix"], "geo_centroids")
    return centroids, {
        "raw_rows": len(geo), "exact_duplicates_removed": len(geo) - len(clean),
        "unique_rows": len(clean), "globally_invalid_unique_rows": int((~global_valid).sum()),
        "outside_brazil_box_unique_rows": int((global_valid & ~brazil_box).sum()),
        "invalid_zip_unique_rows": int((~valid_zip).sum()),
        "valid_unique_rows": int(valid.sum()), "zip_centroids": len(centroids),
    }


def strict_sum(series: pd.Series):
    """Nunca apresenta uma soma parcial como se fosse um total completo."""
    return series.sum(min_count=len(series))


def stable_mode(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iloc[0] if len(modes) else pd.NA


def build_features(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Não lê status, pagamentos, avaliações nem eventos posteriores à compra."""
    orders = tables["orders"][ORDER_INPUTS].copy()
    customers = tables["customers"][CUSTOMER_INPUTS].copy()
    sellers = tables["sellers"][SELLER_INPUTS].copy()
    products = tables["products"][PRODUCT_INPUTS].copy()
    items = tables["items"][ITEM_INPUTS].copy()
    for name, frame, key in [
        ("orders", orders, ["order_id"]), ("customers", customers, ["customer_id"]),
        ("sellers", sellers, ["seller_id"]), ("products", products, ["product_id"]),
        ("items", items, ["order_id", "order_item_id"]),
    ]:
        assert_key(frame, key, name)

    centroids, geo_info = geo_centroids(tables["geolocation"])
    customer_geo = centroids.rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix", "lat": "buyer_lat", "lon": "buyer_lon",
    }).drop(columns="geo_observations")
    seller_geo = centroids.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_code_prefix", "lat": "seller_lat", "lon": "seller_lon",
    }).drop(columns="geo_observations")
    customers = customers.merge(customer_geo, on="customer_zip_code_prefix", how="left", validate="many_to_one")
    sellers = sellers.merge(seller_geo, on="seller_zip_code_prefix", how="left", validate="many_to_one")
    order_base = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    enriched = items.merge(products, on="product_id", how="left", validate="many_to_one")
    enriched = enriched.merge(sellers, on="seller_id", how="left", validate="many_to_one")
    enriched = enriched.merge(order_base[["order_id", "buyer_lat", "buyer_lon", "customer_state"]],
                              on="order_id", how="left", validate="many_to_one")
    if len(enriched) != len(items):
        raise ValueError("A junção multiplicou linhas de itens")

    invalid_numeric = {}
    for col in ["price", "freight_value", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        values = pd.to_numeric(enriched[col], errors="coerce")
        invalid = ~np.isfinite(values) | (values.lt(0) if col in ("price", "freight_value") else values.le(0))
        invalid_numeric[col] = int((invalid & enriched[col].notna()).sum())
        enriched[col] = values.mask(invalid)
    enriched["volume_cm3"] = enriched.product_length_cm * enriched.product_height_cm * enriched.product_width_cm
    enriched["missing_weight"] = enriched.product_weight_g.isna().astype(int)
    enriched["missing_dimensions"] = enriched[["product_length_cm", "product_height_cm", "product_width_cm"]].isna().any(axis=1).astype(int)
    enriched["missing_price"] = enriched.price.isna().astype(int)
    enriched["missing_freight"] = enriched.freight_value.isna().astype(int)
    enriched["missing_category"] = enriched.product_category_name.isna().astype(int)
    enriched["distance_km"] = haversine_km(enriched.buyer_lat, enriched.buyer_lon, enriched.seller_lat, enriched.seller_lon)
    enriched["missing_distance"] = enriched.distance_km.isna().astype(int)
    enriched["same_state"] = (enriched.customer_state == enriched.seller_state).astype("Float64").mask(
        enriched.customer_state.isna() | enriched.seller_state.isna())

    aggregates = enriched.groupby("order_id", sort=True).agg(
        products_value_total=("price", strict_sum), freight_value_total=("freight_value", strict_sum),
        items_count=("order_item_id", "size"), products_count=("product_id", "nunique"),
        sellers_count=("seller_id", "nunique"), seller_states_count=("seller_state", "nunique"),
        same_state_items_fraction=("same_state", "mean"),
        weight_g_total=("product_weight_g", strict_sum), weight_g_mean=("product_weight_g", "mean"),
        volume_cm3_total=("volume_cm3", strict_sum), length_cm_max=("product_length_cm", "max"),
        height_cm_max=("product_height_cm", "max"), width_cm_max=("product_width_cm", "max"),
        items_missing_weight_count=("missing_weight", "sum"),
        items_missing_dimensions_count=("missing_dimensions", "sum"),
        items_missing_price_count=("missing_price", "sum"), items_missing_freight_count=("missing_freight", "sum"),
        categories_count=("product_category_name", "nunique"), main_category=("product_category_name", stable_mode),
        items_missing_category_count=("missing_category", "sum"),
        distance_km_mean=("distance_km", "mean"), distance_km_min=("distance_km", "min"), distance_km_max=("distance_km", "max"),
        items_missing_distance_count=("missing_distance", "sum"),
    ).reset_index()
    aggregates["order_value_total"] = aggregates.products_value_total + aggregates.freight_value_total
    result = order_base.merge(aggregates, on="order_id", how="left", validate="one_to_one")
    purchase = parse_time(result.order_purchase_timestamp)
    result["purchase_weekday"] = purchase.dt.day_of_week.astype("Int64")
    result["purchase_month"] = purchase.dt.month.astype("Int64")
    result["purchase_hour"] = purchase.dt.hour.astype("Int64")
    result["purchase_is_weekend"] = result.purchase_weekday.ge(5).astype("Int64")
    result = result[["order_id", *FEATURE_COLUMNS]].sort_values("order_id").reset_index(drop=True)
    assert_key(result, ["order_id"], "features")
    return result, centroids, {"geolocation": geo_info, "invalid_numeric_item_values": invalid_numeric}


def select_target(orders: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """O status futuro é permitido exclusivamente para seleção da coorte rotulada."""
    assert_key(orders, ["order_id"], "orders")
    assert_key(features, ["order_id"], "features")
    purchase = parse_time(orders.order_purchase_timestamp)
    delivery = parse_time(orders.order_delivered_customer_date)
    duration = (delivery - purchase).dt.total_seconds() / 86400.0
    labels = pd.DataFrame({"order_id": orders.order_id, "purchase_timestamp": purchase,
                           "delivery_timestamp": delivery, TARGET: duration})
    labels = labels.merge(features[["order_id", "items_count"]], on="order_id", how="left", validate="one_to_one")
    # Construir máscaras após a junção evita depender do índice externo do DataFrame.
    status = orders.set_index("order_id").order_status
    masks = [
        ("status_not_delivered", ~labels.order_id.map(status).eq("delivered")),
        ("invalid_purchase_timestamp", labels.purchase_timestamp.isna()),
        ("invalid_delivery_timestamp", labels.delivery_timestamp.isna()),
        ("negative_duration", labels[TARGET].lt(0)),
        ("no_items", labels.items_count.fillna(0).le(0)),
    ]
    remaining = pd.Series(True, index=labels.index)
    reasons = pd.Series("included", index=labels.index, dtype="string")
    coverage = [{"stage": "raw_orders", "remaining": len(labels), "excluded_at_stage": 0}]
    for name, mask in masks:
        removed = remaining & mask
        reasons.loc[removed] = name
        remaining &= ~mask
        coverage.append({"stage": name, "remaining": int(remaining.sum()), "excluded_at_stage": int(removed.sum())})
    labels["selection_reason"] = reasons
    audit = labels.drop(columns="items_count").sort_values("order_id").reset_index(drop=True)
    selected = labels.loc[remaining, ["order_id", TARGET]].sort_values("order_id").reset_index(drop=True)
    return selected, audit, pd.DataFrame(coverage)


def payment_audit(payments: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Agregação separada: jamais juntar pagamentos diretamente às linhas de itens."""
    assert_key(payments, ["order_id", "payment_sequential"], "payments")
    summary = payments.groupby("order_id", sort=True).agg(
        payment_records=("payment_sequential", "size"), payment_types=("payment_type", "nunique"),
        payment_value_total=("payment_value", strict_sum), max_installments=("payment_installments", "max"),
    ).reset_index()
    summary = features[["order_id", "order_value_total"]].merge(summary, on="order_id", how="left", validate="one_to_one")
    summary["payment_minus_order_value"] = summary.payment_value_total - summary.order_value_total
    summary["difference_over_0_05"] = summary.payment_minus_order_value.abs().gt(0.05 + 1e-9)
    return summary.sort_values("order_id").reset_index(drop=True)
