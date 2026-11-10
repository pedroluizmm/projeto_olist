"""CLI: python -m olist_features.pipeline --root <diretório do projeto>."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .audit import (column_profile, find_source, key_and_relation_audit, load_tables, missing_profile,
                    outlier_profile, sha256, timestamp_audit, zip_integrity)
from .core import (EARTH_RADIUS_KM, FEATURE_COLUMNS, FEATURE_SPECS, TARGET, build_features,
                   haversine_km, parse_time, payment_audit, select_target)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def independent_checks(tables, features, dataset, centroids, labels) -> pd.DataFrame:
    checks = []

    def record(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(f"Verificação falhou: {name}: {evidence}")

    record("unique_order_id", dataset.order_id.is_unique and dataset.order_id.notna().all(), len(dataset))
    record("features_exact_allowlist", list(features.columns) == ["order_id", *FEATURE_COLUMNS], len(FEATURE_COLUMNS))
    record("dataset_exact_allowlist", list(dataset.columns) == ["order_id", *FEATURE_COLUMNS, TARGET], len(dataset.columns))
    record("finite_nonnegative_target", np.isfinite(dataset[TARGET]).all() and dataset[TARGET].ge(0).all(), "dias fracionários")
    actual = dataset.set_index("order_id")
    counts = tables["items"].groupby("order_id").size().reindex(actual.index)
    record("items_count_independent", np.array_equal(counts.to_numpy(), actual.items_count.to_numpy()), "groupby.size")
    totals = tables["items"].groupby("order_id")[["price", "freight_value"]].sum().reindex(actual.index)
    # O dump observado não tem valores inválidos de preço/frete. As regras de ausência
    # e negativos são verificadas em fixtures sintéticas nos testes unitários.
    for source, feature in [("price", "products_value_total"), ("freight_value", "freight_value_total")]:
        record(f"{feature}_independent", np.allclose(totals[source], actual[feature], equal_nan=True), "soma independente antes das junções")
    record("order_value_identity", np.allclose(actual.order_value_total, actual.products_value_total + actual.freight_value_total, equal_nan=True), "produtos + frete")
    orders = tables["orders"].set_index("order_id")
    # datetime nativo em vez da operação vetorizada do pipeline: todos os rótulos.
    expected = [(datetime.strptime(orders.loc[oid, "order_delivered_customer_date"], "%Y-%m-%d %H:%M:%S") -
                 datetime.strptime(orders.loc[oid, "order_purchase_timestamp"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 86400
                for oid in dataset.order_id]
    record("target_datetime_independent", np.allclose(expected, dataset[TARGET], rtol=0, atol=1e-12), len(expected))
    record("label_alignment", np.array_equal(labels.order_id.to_numpy(), dataset.order_id.to_numpy()), "merge one_to_one e ordenação")
    record("centroids_unique", centroids.geolocation_zip_code_prefix.is_unique, len(centroids))
    record("centroids_brazil_box", centroids.lat.between(-34, 6).all() and centroids.lon.between(-74, -28).all(), "caixa ampla")
    # Amostra determinística de pares; lei esférica dos cossenos como fórmula independente.
    points = centroids.iloc[::max(1, len(centroids) // 100)].head(100)
    other = points.iloc[::-1].reset_index(drop=True)
    direct = haversine_km(points.lat, points.lon, other.lat, other.lon)
    independent = []
    for a, b, c, d in zip(points.lat, points.lon, other.lat, other.lon):
        a, b, c, d = map(math.radians, (a, b, c, d))
        cos_angle = math.sin(a) * math.sin(c) + math.cos(a) * math.cos(c) * math.cos(d - b)
        independent.append(EARTH_RADIUS_KM * math.acos(max(-1, min(1, cos_angle))))
    record("distance_independent_formula", np.allclose(direct, independent, rtol=0, atol=0.001), len(points))
    # A média de valores iguais pode ultrapassar o máximo por arredondamento binário.
    # Tolerância de 1e-9 km = um micrômetro; não altera nenhuma feature.
    tolerance_km = 1e-9
    max_violation = max(float((dataset.distance_km_min - dataset.distance_km_mean).max()),
                        float((dataset.distance_km_mean - dataset.distance_km_max).max()), 0.0)
    record("distance_order_bounds", max_violation <= tolerance_km,
           f"mínimo <= média <= máximo; erro máximo={max_violation:.3g} km; tolerância={tolerance_km} km")
    record("weekday_range", dataset.purchase_weekday.between(0, 6).all(), "segunda=0")
    record("month_range", dataset.purchase_month.between(1, 12).all(), "1..12")
    record("no_infinite_features", not np.isinf(dataset.select_dtypes(include="number").astype(float)).any().any(), "ausências preservadas")
    return pd.DataFrame(checks)


def technical_report(root, dataset, coverage, info, timestamp_counts, payments, manifest):
    def n(value):
        return f"{int(value):,}".replace(",", ".")

    rows = "\n".join(f"| {r.stage} | {n(r.remaining)} | {n(r.excluded_at_stage)} |" for r in coverage.itertuples())
    missing = missing_profile(dataset).query("missing > 0")
    nulls = "\n".join(f"| {r.column} | {n(r.missing)} | {r.missing_pct:.3f}% |" for r in missing.itertuples())
    time_flags = "; ".join(f"`{key}`: {n(value)}" for key, value in timestamp_counts.items() if value)
    dependencies = ", ".join(f"{key} {value}" for key, value in manifest["versions"].items())
    tests = "Execução registrada em `resultados/tabelas/pytest.xml` e `pytest.txt` após a geração."
    junit = root / "resultados/tabelas/pytest.xml"
    if junit.exists():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        tests = (f"{sum(int(s.get('tests', 0)) for s in suites)} testes; "
                 f"{sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)} falhas/erros; "
                 f"{sum(int(s.get('skipped', 0)) for s in suites)} ignorados. "
                 "Evidência: `resultados/tabelas/pytest.xml` e `pytest.txt`.")
    text = f"""# Engenharia de features — Olist (TASK 01)

## Problema e momento da previsão

Regressão supervisionada com **um pedido por linha**. O alvo `delivery_time_days`
é `(order_delivered_customer_date - order_purchase_timestamp).total_seconds()/86400`,
em dias fracionários (sem arredondamento ou truncamento). Prevê-se no registro da compra,
com carrinho e endereço definidos. Datas do dump não informam fuso; não se inventou conversão.
O alvo mede compra até recebimento, incluindo espera de aprovação.

## Fonte e relacionamentos

Os nove CSVs são lidos diretamente do ZIP, sem extração. `orders` → `customers` por
`customer_id` (N:1 conceitualmente, 1:1 neste dump); `items` → `orders`, `products` e
`sellers` por seus IDs (N:1). O `customer_unique_id` identifica pessoas recorrentes,
mas não é chave de junção do pedido. `payments` e `reviews` têm múltiplas linhas por
pedido; `review_id` também não é único. Produtos → tradução de categoria é uma relação
opcional. Clientes e vendedores usam prefixos de CEP (texto de cinco caracteres).
Geolocalização é N:N no bruto; somente suas medianas por CEP entram em junções N:1.
Inventário e perfil de **todas** as colunas: `raw_inventory.csv` e `raw_columns.csv`.
Chaves e órfãos: `key_checks.csv` e `relationships.csv`, em `resultados/tabelas/`.

## Seleção e cobertura

Selecionam-se `order_status == delivered`, compra e recebimento válidos, duração ≥ 0
e ao menos um item. O status e o recebimento são usados apenas para seleção/rótulo.
Razões sequenciais, mutuamente exclusivas:

| Etapa | Restantes | Excluídos nesta etapa |
|---|---:|---:|
{rows}

Cobertura: **{len(dataset) / int(coverage.iloc[0].remaining) * 100:.4f}%** dos pedidos originais.
Não se exigem aprovação ou despacho válidos para um alvo compra–recebimento válido;
suas inconsistências ficam sinalizadas. Não entregues podem incluir censura à direita,
cancelamento e indisponibilidade: esta coorte não estima entrega de todo pedido criado.
`order_selection_audit.parquet` preserva datas/razões fora das features; cobertura por
status, mês e UF está nas tabelas `coverage_by_*.csv`.

## Features e disponibilidade

São **{len(FEATURE_COLUMNS)} features**, explicitadas em `feature_dictionary.csv` e
`feature_contract.json`. Valores e frete vêm dos itens do carrinho; quantidades de itens,
produtos, vendedores e UFs representam complexidade logística. Peso, dimensões máximas,
volume, categorias, UF do comprador e proporção de itens na mesma UF usam cadastros.
Calendário: dia da semana, mês, hora e fim de semana, todos derivados da compra.

Cada linha de item corresponde a uma unidade; não há coluna de quantidade separada.
Preço/frete/peso/volume são somados **por item**, respeitando produtos repetidos. Totais
ficam ausentes se algum componente faltar. Médias/máximos usam componentes conhecidos,
com contadores de ausência. Categoria principal é a moda por item; empate alfabético.

Distância: Haversine com raio {EARTH_RADIUS_KM} km, entre medianas marginais de latitude/
longitude por CEP. Removem-se {n(info['geolocation']['exact_duplicates_removed'])} duplicatas integrais em memória;
{n(info['geolocation']['globally_invalid_unique_rows'])} coordenadas únicas globalmente inválidas e
{n(info['geolocation']['outside_brazil_box_unique_rows'])} fora da caixa ampla [-34,6] × [-74,-28] são descartadas
apenas da referência geográfica. Restam {n(info['geolocation']['zip_centroids'])} CEPs válidos.
Calculam-se média (ponderada pela quantidade de itens), mínimo e máximo; distâncias
parciais têm contador explícito. É aproximação esférica de localização, não rota de transporte.

O dump é um retrato estático, sem histórico dos cadastros. Assume-se que produto,
vendedor, endereço, preço e frete correspondem à compra; a vigência histórica desses
valores não pode ser provada. Coordenadas são referência estática, sem uso do alvo.
Essa hipótese deve ser validada com o professor antes de afirmar disponibilidade operacional.

## Proteção contra vazamento e multiplicação de linhas

`build_features` lê somente colunas permitidas de compra, itens e cadastros. O alvo é
construído em função separada. IDs, timestamps brutos, `order_status`, aprovação,
despacho, entrega, prazo estimado, `shipping_limit_date`, avaliações e pagamentos não
integram X. O contrato usa lista positiva, e os testes removem/alteram os eventos
proibidos, verificando que X permanece igual. O modelo futuro deverá ler **somente**
as colunas de `feature_contract.json`, nunca inferir X removendo apenas o alvo.

Itens são agregados antes da junção por pedido, com `validate=one_to_one`/
`many_to_one`. Pagamentos são agregados independentemente por pedido **só para auditoria**:
{n(payments.payment_records.gt(1).sum())} pedidos têm múltiplos registros,
{n(payments.payment_records.isna().sum())} não têm registro e
{n(payments.difference_over_0_05.sum())} diferem de produtos+frete em mais de R$ 0,05
(diagnóstico, sem correção automática). Não há timestamps de pagamentos que sustentem
sua disponibilidade exata na compra; parcelas/tipo/valor não são usados como entradas.

## Ausências, inconsistências e outliers

No dataset final:

| Coluna | Ausentes | Percentual |
|---|---:|---:|
{nulls}

Medidas não positivas e valores negativos/infinitos tornam-se ausentes nas cópias;
frete zero é permitido, preço zero é preservado para revisão. Ocorrências por coluna:
`quality_summary.json`. Não há imputação, escalonamento ou codificação aprendidos.
IQR (1,5×) sinaliza extremos sem remover/capar valores: `outliers_numeric.csv` e
`outlier_flags.parquet`. IQR=0 e variáveis discretas podem sinalizar valores legítimos.
Tempos longos são preservados; não se adota limite arbitrário de entrega.

Alertas temporais observados (podem se sobrepor): {time_flags or 'nenhum'}.
IDs afetados: `timestamp_inconsistencies.csv`. Casos de ordem intermediária incoerente
ficam na coorte quando compra e recebimento sustentam um alvo válido.

## Verificação e reprodutibilidade

ZIPs originais: CRC e SHA-256 iguais antes/depois; SHA-256 da fonte:
`{manifest['input_integrity_after'][str(Path(manifest['source']))]['sha256']}`.
Asserções de unicidade, chaves, junções, allowlist, somas, datas e distância passaram;
`verification_checks.csv` contém as evidências. Todos os alvos foram recalculados
com `datetime` nativo e a distância foi comparada à lei esférica dos cossenos.
{tests}

Ambiente: Python {platform.python_version()}; {dependencies}. Instalação e comandos
em `README.md`, versões diretas em `requirements.txt`, transitivas em
`requirements-lock.txt`; kernel `olist` instalado em `.venv/share/jupyter/kernels/olist`.
`run_manifest.json` registra hashes de entradas, código e artefatos. CSV e Parquet são
gerados em ordem determinística de `order_id`. Arquivos de auditoria contêm informação
posterior e ficam separados de X.

## Pendências para decisão humana

1. Confirmar que o problema será condicionado à entrega observada e discutir censura
   dos pedidos ainda em andamento; oito entregues sem recebimento foram excluídos.
2. Validar a hipótese de vigência dos cadastros e do carrinho na compra e a aproximação
   por CEP, inclusive a caixa geográfica adotada e distâncias parciais.
3. Revisar extremos legítimos, pesos zero, lacunas de medidas/categorias e divergências
   de pagamento; nenhuma correção baseada em suposição foi aplicada ao original.
4. Na próxima tarefa, definir separação temporal (e possível agrupamento por cliente),
   data de corte e janela de observação dos rótulos. Ajustar imputação, codificação e
   outras transformações apenas no treino; não usar estatísticas deste relatório para
   selecionar features com base no conjunto de teste.

**Escopo encerrado em preparação e engenharia: sem EDA completa, treinamento ou otimização.**
"""
    (root / "docs/metodologia_detalhada.md").write_text(text, encoding="utf-8")
    brief = f"""# Engenharia de features — Olist (TASK 01)

**Ciência dos Dados (T326), UNIFOR — primeiro acompanhamento.**

## Problema e seleção

Regressão supervisionada com **um pedido por linha**, prevendo na compra o tempo até
recebimento. Alvo: `delivery_time_days = (entrega - compra).total_seconds()/86400`,
em dias fracionários. Datas sem fuso declarado foram preservadas sem conversão.
Usam-se entregues com compra/recebimento válidos, duração ≥ 0 e ao menos um item.
Não se exige aprovação ou despacho válidos quando o alvo compra–recebimento é válido.

| Etapa sequencial | Restantes | Excluídos na etapa |
|---|---:|---:|
{rows}

Cobertura: **{len(dataset) / int(coverage.iloc[0].remaining) * 100:.4f}%** dos originais.
Exclusões não se sobrepõem. Não entregues podem estar censurados ou cancelados;
a coorte representa entregas observadas, não todos os pedidos criados.

## Tabelas e junções

O ZIP contém nove CSVs, lidos sem extração. `orders` → `customers` por `customer_id`;
`items` → `orders`, `products` e `sellers` pelos IDs (N:1). `customer_unique_id`
identifica recorrência de pessoas, mas não é chave da junção. `payments` e `reviews`
têm múltiplos registros por pedido; `review_id` não é único. Tradução de categorias
é opcional. Geolocalização tem múltiplas linhas por CEP; só medianas por CEP entram
em junções N:1. As chaves e relações obrigatórias foram verificadas sem órfãos.

## Features e prevenção de vazamento

São **{len(FEATURE_COLUMNS)} entradas**, descritas em `resultados/tabelas/feature_dictionary.csv`:

- Valores de produtos, frete e total: somas dos itens do carrinho, em reais.
- Quantidades de itens, produtos, vendedores e UFs de vendedores; fração de itens
  com vendedor na mesma UF do comprador: complexidade e contexto logístico.
- Peso total/médio, volume total e dimensões máximas: características físicas.
  Totais exigem todos os componentes; médias/máximos usam os conhecidos.
- Categoria principal (moda por item; empate alfabético), número de categorias e
  UF do comprador: composição do pedido e localização.
- Distância Haversine mínima/média/máxima por item; calendário da compra
  (dia da semana, mês, hora e fim de semana); contadores explícitos de ausência.

Uma linha de item representa uma unidade. Os itens são agregados **antes** da junção
por pedido; pagamentos são agregados separadamente, apenas para auditoria. Há
{n(payments.payment_records.gt(1).sum())} pedidos com múltiplos pagamentos e
{n(payments.difference_over_0_05.sum())} divergências superiores a R$ 0,05 em relação
a produtos+frete, sem correção automática ou inclusão dessas variáveis em X.

Geolocalização: removidas {n(info['geolocation']['exact_duplicates_removed'])} duplicatas integrais em memória;
coordenadas impossíveis e fora da caixa ampla [-34,6] × [-74,-28] são invalidadas.
Restam {n(info['geolocation']['zip_centroids'])} prefixos válidos, representados por medianas marginais.
Haversine usa raio {EARTH_RADIUS_KM} km; a média é ponderada pela quantidade de itens.
É aproximação por CEP, não distância de rota. Pares ausentes não eliminam pedidos.

X usa lista positiva em `dados/tratados/feature_contract.json`. IDs, timestamps brutos,
alvo, status, aprovação, despacho, recebimento, prazo estimado, limite de despacho,
pagamentos e avaliações ficam fora. Eventos futuros servem apenas para rótulo/auditoria.
**Hipótese:** cadastros e valores dos itens representam o instante da compra; o dump
estático não permite comprovar sua vigência histórica.

## Qualidade e verificação

Ausências no dataset final:

| Coluna | Ausentes | Percentual |
|---|---:|---:|
{nulls}

Medidas não positivas e valores negativos/infinitos tornam-se ausentes nas cópias;
frete/preço zero são preservados. Extremos são sinalizados por IQR, sem remoção ou
limite arbitrário para o alvo. Não houve imputação ou transformação aprendida.
Alertas temporais (sobrepostos): {time_flags or 'nenhum'}.

Passaram as verificações de chaves, unicidade, somas, datas, distância e contrato.
Todos os alvos foram recalculados com `datetime`; Haversine foi comparado a uma
fórmula independente. {tests}
Testes também alteram/removem eventos posteriores e confirmam invariância de X.
CRC e SHA-256 dos ZIPs são iguais antes/depois; hashes e evidências constam em
`resultados/tabelas/run_manifest.json` e `verification_checks.csv`.

## Ambiente, entrega e decisões humanas

Python {platform.python_version()} em `.venv`, dependências fixadas e kernel
**Python (Olist .venv)**. `README.md` contém estrutura e comandos reproduzíveis.
Dataset: `dados/tratados/olist_orders_features.parquet` e `.csv`; auditorias,
ausências, cobertura por mês/UF/status e outliers: `resultados/tabelas/`.
Método completo: [metodologia_detalhada.md](metodologia_detalhada.md).

Validar com o professor: (1) coorte condicionada à entrega e censura; (2) vigência
dos cadastros e aproximação por CEP; (3) extremos, lacunas e divergências de pagamento;
(4) divisão temporal e janela de observação dos rótulos na próxima etapa. Imputação
e codificação deverão ser ajustadas somente no treino futuro.

**Etapa encerrada: sem EDA completa, treinamento ou otimização.**
"""
    (root / "docs/engenharia_features.md").write_text(brief, encoding="utf-8")


def run(root: Path) -> dict:
    root = root.resolve()
    # Interromper por ausência/corrupção antes de criar saídas.
    integrity_before = zip_integrity(root)
    source = find_source(root)
    print(f"READ: {source.name}; CRC e SHA-256 registrados.", flush=True)
    tables, inventory = load_tables(source)
    reports, treated = root / "resultados/tabelas", root / "dados/tratados"
    reports.mkdir(parents=True, exist_ok=True)
    treated.mkdir(parents=True, exist_ok=True)
    columns = column_profile(tables)
    keys, relationships = key_and_relation_audit(tables)
    print("APPLY: chaves verificadas; construindo features por pedido.", flush=True)
    features, centroids, quality = build_features(tables)
    labels, selection, coverage = select_target(tables["orders"], features)
    dataset = features.merge(labels, on="order_id", how="inner", validate="one_to_one")
    dataset = dataset[["order_id", *FEATURE_COLUMNS, TARGET]].sort_values("order_id").reset_index(drop=True)
    checks = independent_checks(tables, features, dataset, centroids, labels)
    payments = payment_audit(tables["payments"], features)
    timestamps, timestamp_counts = timestamp_audit(tables["orders"])
    outliers, outlier_flags = outlier_profile(dataset)
    quality["timestamp_inconsistencies"] = timestamp_counts
    quality["payments"] = {"multiple_payment_orders": int(payments.payment_records.gt(1).sum()),
                           "orders_without_payment": int(payments.payment_records.isna().sum()),
                           "differences_over_0_05": int(payments.difference_over_0_05.sum())}
    quality["raw_zero_product_weights"] = int(tables["products"].product_weight_g.eq(0).sum())
    quality["raw_zero_item_prices"] = int(tables["items"].price.eq(0).sum())
    quality["raw_zero_item_freights"] = int(tables["items"].freight_value.eq(0).sum())
    quality["final_rows"] = len(dataset)
    quality["feature_count"] = len(FEATURE_COLUMNS)

    csvs = {"raw_inventory.csv": inventory, "raw_columns.csv": columns, "key_checks.csv": keys,
            "relationships.csv": relationships, "coverage.csv": coverage,
            "missing_final.csv": missing_profile(dataset), "outliers_numeric.csv": outliers,
            "timestamp_inconsistencies.csv": timestamps, "payments_by_order_audit.csv": payments,
            "verification_checks.csv": checks,
            "feature_dictionary.csv": pd.DataFrame([{"feature": col, "source": FEATURE_SPECS[col][0],
                 "description": FEATURE_SPECS[col][1], "dtype": str(dataset[col].dtype),
                 "availability": "compra/cadastro; hipótese de vigência histórica"} for col in FEATURE_COLUMNS])}
    coverage_base = tables["orders"][["order_id", "customer_id", "order_status", "order_purchase_timestamp"]].copy()
    coverage_base = coverage_base.merge(tables["customers"][["customer_id", "customer_state"]], on="customer_id", validate="many_to_one")
    coverage_base["purchase_month"] = parse_time(coverage_base.order_purchase_timestamp).dt.strftime("%Y-%m")
    coverage_base["included"] = coverage_base.order_id.isin(labels.order_id)
    for col in ("order_status", "purchase_month", "customer_state"):
        summary = coverage_base.groupby(col, dropna=False, sort=True).agg(raw_orders=("order_id", "size"), included_orders=("included", "sum")).reset_index()
        summary["coverage_pct"] = summary.included_orders / summary.raw_orders * 100
        csvs[f"coverage_by_{col}.csv"] = summary
    for name, frame in csvs.items():
        write_csv(frame, reports / name)
    write_csv(dataset, treated / "olist_orders_features.csv")
    dataset.to_parquet(treated / "olist_orders_features.parquet", index=False)
    features.to_parquet(treated / "purchase_features_all_orders.parquet", index=False)
    selection.to_parquet(reports / "order_selection_audit.parquet", index=False)
    centroids.to_parquet(reports / "geolocation_centroids.parquet", index=False)
    outlier_flags.to_parquet(reports / "outlier_flags.parquet", index=False)
    write_json(quality, reports / "quality_summary.json")
    contract = {"id_column": "order_id", "target_column": TARGET, "feature_columns": FEATURE_COLUMNS,
                "excluded_from_X": ["order_id", TARGET, "customer_id", "customer_unique_id", "order_status",
                                    "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
                                    "order_estimated_delivery_date", "shipping_limit_date", "payments.*", "reviews.*"],
                "prediction_time": "order_purchase_timestamp; carrinho e endereço definidos",
                "historical_snapshot_assumption": "cadastros e itens representam valores disponíveis na compra"}
    write_json(contract, treated / "feature_contract.json")
    integrity_after = zip_integrity(root)
    if integrity_before != integrity_after:
        raise AssertionError("A integridade dos arquivos brutos mudou durante a execução")
    versions = {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow", "ipykernel", "jupyterlab", "pytest")}
    output_paths = [reports / name for name in csvs]
    output_paths += [treated / name for name in ("olist_orders_features.csv", "olist_orders_features.parquet",
                      "purchase_features_all_orders.parquet", "feature_contract.json")]
    output_paths += [reports / name for name in ("order_selection_audit.parquet", "geolocation_centroids.parquet",
                      "outlier_flags.parquet", "quality_summary.json")]
    code_paths = sorted((root / "src").rglob("*.py"))
    code_paths += [root / "pyproject.toml", root / "requirements.txt", root / "requirements-lock.txt"]
    manifest = {"source": str(source.relative_to(root)), "python": platform.python_version(), "versions": versions,
                "input_integrity_before": integrity_before, "input_integrity_after": integrity_after,
                "code_sha256": {str(p.relative_to(root)): sha256(p) for p in code_paths if p.exists()},
                "output_sha256": {str(p.relative_to(root)): sha256(p) for p in output_paths},
                "rows_before": len(tables["orders"]), "rows_after": len(dataset), "features": FEATURE_COLUMNS,
                "trained_models": 0}
    write_json(manifest, reports / "run_manifest.json")
    technical_report(root, dataset, coverage, quality, timestamp_counts, payments, manifest)
    print(f"VERIFY: {len(checks)} verificações; {len(dataset):,} pedidos; {len(FEATURE_COLUMNS)} features; originais intactos.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
