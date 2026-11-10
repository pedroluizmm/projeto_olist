# Engenharia de features — Olist (TASK 01)

**Ciência dos Dados (T326), UNIFOR — primeiro acompanhamento.**

> Escopo histórico da TASK 01: preparação, sem ajuste de modelos naquela etapa.
> A TASK 02 posteriormente realizou EDA, pré-processamento e treinamento; a TASK 04
> corrigiu a documentação. Estado atual: [README](../README.md) e
> [relatório de treinamento](treinamento_avaliacao.md). Os números abaixo são os
> registros dessa preparação, não novos resultados desta revisão.

## Problema e seleção

Regressão supervisionada com **um pedido por linha**, prevendo na compra o tempo até
recebimento. Alvo: `delivery_time_days = (entrega - compra).total_seconds()/86400`,
em dias fracionários. Datas sem fuso declarado foram preservadas sem conversão.
Usam-se entregues com compra/recebimento válidos, duração ≥ 0 e ao menos um item.
Não se exige aprovação ou despacho válidos quando o alvo compra–recebimento é válido.

| Etapa sequencial | Restantes | Excluídos na etapa |
|---|---:|---:|
| raw_orders | 99.441 | 0 |
| status_not_delivered | 96.478 | 2.963 |
| invalid_purchase_timestamp | 96.478 | 0 |
| invalid_delivery_timestamp | 96.470 | 8 |
| negative_duration | 96.470 | 0 |
| no_items | 96.470 | 0 |

Cobertura: **97.0123%** dos originais.
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

São **30 entradas**, descritas em `resultados/tabelas/feature_dictionary.csv`:

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
2.961 pedidos com múltiplos pagamentos e
260 divergências superiores a R$ 0,05 em relação
a produtos+frete, sem correção automática ou inclusão dessas variáveis em X.

Geolocalização: removidas 261.831 duplicatas integrais em memória;
coordenadas impossíveis e fora da caixa ampla [-34,6] × [-74,-28] são invalidadas.
Restam 19.011 prefixos válidos, representados por medianas marginais.
Haversine usa raio 6371.0088 km; a média é ponderada pela quantidade de itens.
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
| weight_g_total | 22 | 0.023% |
| weight_g_mean | 22 | 0.023% |
| volume_cm3_total | 16 | 0.017% |
| length_cm_max | 16 | 0.017% |
| height_cm_max | 16 | 0.017% |
| width_cm_max | 16 | 0.017% |
| main_category | 1.332 | 1.381% |
| distance_km_mean | 477 | 0.494% |
| distance_km_min | 477 | 0.494% |
| distance_km_max | 477 | 0.494% |

Medidas não positivas e valores negativos/infinitos tornam-se ausentes nas cópias;
frete/preço zero são preservados. Extremos são sinalizados por IQR, sem remoção ou
limite arbitrário para o alvo. Não houve imputação ou transformação aprendida.
Alertas temporais (sobrepostos): `delivered_missing_delivery`: 8; `not_delivered_with_delivery`: 6; `carrier_before_purchase`: 166; `carrier_before_approval`: 1.359; `delivery_before_carrier`: 23; `delivered_missing_approval`: 14; `delivered_missing_carrier`: 2.

Passaram as verificações de chaves, unicidade, somas, datas, distância e contrato.
Todos os alvos foram recalculados com `datetime`; Haversine foi comparado a uma
fórmula independente. 33 testes; 0 falhas/erros; 0 ignorados. Evidência: `resultados/tabelas/pytest.xml` e `pytest.txt`.
Testes também alteram/removem eventos posteriores e confirmam invariância de X.
CRC e SHA-256 dos ZIPs são iguais antes/depois; hashes e evidências constam em
`resultados/tabelas/run_manifest.json` e `verification_checks.csv`.

## Ambiente, entrega e decisões humanas

Python 3.13.6 em `.venv`, dependências fixadas e kernel
**Python (Olist .venv)**. `README.md` contém estrutura e comandos reproduzíveis.
Dataset: `dados/tratados/olist_orders_features.parquet` e `.csv`; auditorias,
ausências, cobertura por mês/UF/status e outliers: `resultados/tabelas/`.
Método completo: [metodologia_detalhada.md](metodologia_detalhada.md).

Validar com o professor: (1) coorte condicionada à entrega e censura; (2) vigência
dos cadastros e aproximação por CEP; (3) extremos, lacunas e divergências de pagamento;
(4) divisão temporal e janela de observação dos rótulos na próxima etapa. Imputação
e codificação deverão ser ajustadas somente no treino futuro.

**A TASK 01 encerrou na preparação. EDA, treinamento e busca limitada ocorreram
posteriormente na TASK 02; a revisão TASK 04 não alterou esses experimentos.**
