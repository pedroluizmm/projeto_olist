# Engenharia de features — Olist (TASK 01)

> Documento metodológico da preparação histórica. A ausência de modelagem descrita
> abaixo se refere à TASK 01. A TASK 02 realizou EDA, pré-processamento e treinamento;
> a TASK 04 apenas revisou a documentação. Veja o [estado atual](../README.md) e
> as [limitações e decisões](limitacoes_decisoes.md).

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
| raw_orders | 99.441 | 0 |
| status_not_delivered | 96.478 | 2.963 |
| invalid_purchase_timestamp | 96.478 | 0 |
| invalid_delivery_timestamp | 96.470 | 8 |
| negative_duration | 96.470 | 0 |
| no_items | 96.470 | 0 |

Cobertura: **97.0123%** dos pedidos originais.
Não se exigem aprovação ou despacho válidos para um alvo compra–recebimento válido;
suas inconsistências ficam sinalizadas. Não entregues podem incluir censura à direita,
cancelamento e indisponibilidade: esta coorte não estima entrega de todo pedido criado.
`order_selection_audit.parquet` preserva datas/razões fora das features; cobertura por
status, mês e UF está nas tabelas `coverage_by_*.csv`.

## Features e disponibilidade

São **30 features**, explicitadas em `feature_dictionary.csv` e
`feature_contract.json`. Valores e frete vêm dos itens do carrinho; quantidades de itens,
produtos, vendedores e UFs representam complexidade logística. Peso, dimensões máximas,
volume, categorias, UF do comprador e proporção de itens na mesma UF usam cadastros.
Calendário: dia da semana, mês, hora e fim de semana, todos derivados da compra.

Cada linha de item corresponde a uma unidade; não há coluna de quantidade separada.
Preço/frete/peso/volume são somados **por item**, respeitando produtos repetidos. Totais
ficam ausentes se algum componente faltar. Médias/máximos usam componentes conhecidos,
com contadores de ausência. Categoria principal é a moda por item; empate alfabético.

Distância: Haversine com raio 6371.0088 km, entre medianas marginais de latitude/
longitude por CEP. Removem-se 261.831 duplicatas integrais em memória;
0 coordenadas únicas globalmente inválidas e
27 fora da caixa ampla [-34,6] × [-74,-28] são descartadas
apenas da referência geográfica. Restam 19.011 CEPs válidos.
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
2.961 pedidos têm múltiplos registros,
1 não têm registro e
260 diferem de produtos+frete em mais de R$ 0,05
(diagnóstico, sem correção automática). Não há timestamps de pagamentos que sustentem
sua disponibilidade exata na compra; parcelas/tipo/valor não são usados como entradas.

## Ausências, inconsistências e outliers

No dataset final:

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
frete zero é permitido, preço zero é preservado para revisão. Ocorrências por coluna:
`quality_summary.json`. Não há imputação, escalonamento ou codificação aprendidos.
IQR (1,5×) sinaliza extremos sem remover/capar valores: `outliers_numeric.csv` e
`outlier_flags.parquet`. IQR=0 e variáveis discretas podem sinalizar valores legítimos.
Tempos longos são preservados; não se adota limite arbitrário de entrega.

Alertas temporais observados (podem se sobrepor): `delivered_missing_delivery`: 8; `not_delivered_with_delivery`: 6; `carrier_before_purchase`: 166; `carrier_before_approval`: 1.359; `delivery_before_carrier`: 23; `delivered_missing_approval`: 14; `delivered_missing_carrier`: 2.
IDs afetados: `timestamp_inconsistencies.csv`. Casos de ordem intermediária incoerente
ficam na coorte quando compra e recebimento sustentam um alvo válido.

## Verificação e reprodutibilidade

ZIPs originais: CRC e SHA-256 iguais antes/depois; SHA-256 da fonte:
`967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784`.
Asserções de unicidade, chaves, junções, allowlist, somas, datas e distância passaram;
`verification_checks.csv` contém as evidências. Todos os alvos foram recalculados
com `datetime` nativo e a distância foi comparada à lei esférica dos cossenos.
33 testes; 0 falhas/erros; 0 ignorados. Evidência: `resultados/tabelas/pytest.xml` e `pytest.txt`.

Ambiente: Python 3.13.6; pandas 3.1.0, numpy 2.5.4, pyarrow 26.0.1, ipykernel 7.4.0, jupyterlab 4.6.5, pytest 9.2.0. Instalação e comandos
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

**A TASK 01 encerrou em preparação e engenharia. EDA, treinamento e busca limitada
foram realizados na TASK 02; não foram repetidos na revisão documental TASK 04.**
