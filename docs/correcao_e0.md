# E0 — Correção da importância por permutação

Aplicação autorizada em 08/11/2026. Escopo: corrigir somente o diagnóstico de
importância da Ridge de validação. As 30 features, seus cálculos, a imputação,
os splits, os hiperparâmetros e os modelos treinados permanecem iguais.

## READ — Estado e causa

A cópia local não contém repositório Git reconhecido. `git status --short`
retornou `fatal: not a git repository`; não é possível identificar diferenças
prévias por Git. Nenhum commit foi feito.

Antes da edição, o inventário
[`preservation_before.json`](../resultados/tabelas/e0/preservation_before.json)
registrou 273 arquivos preexistentes fora de ambientes e caches: 263 por SHA-256
e dez artefatos de teste final somente por tamanho e timestamp, sem abri-los.

A execução histórica chamava `permutation_importance` com a referência nos
12.675 pedidos da validação e `max_samples=3000` nas permutações. Na versão
instalada, scikit-learn 1.9.1, isso acrescentava a diferença de MAE entre a
subamostra e a população: **+0,014930298556740773 dia**. Por exemplo, as colunas
de preço/frete ausentes, removidas pelo pipeline, recebiam importância positiva.

Os registros históricos não foram sobrescritos. A correção não altera previsões,
métricas ou seleção; o diagnóstico é calculado depois dessas etapas.

## APPLY — Alterações

- `src/olist_modeling/run.py`: usa o cálculo compartilhado de importância com
  `max_samples=1.0`, preservando scoring MAE negativo, três repetições,
  semente 42 e `n_jobs=1`.
- `src/olist_modeling/importance.py`: procedimento isolado de recálculo com a
  Ridge já salva. Seleciona somente os IDs da validação nos Parquets
  compartilhados, verifica ordem cronológica/maturação, target e previsões,
  calcula a importância e recusa sobrescrever suas saídas.
- `tests/test_importance_e0.py`: regressões sintéticas sem treino e integração
  somente na validação, com bloqueios de `fit`/`fit_transform` e de acesso aos
  arquivos de teste e aos modelos finais.
- Documentos de treinamento, guia de execução, dicionário de features,
  perguntas do professor e galeria: referências diretas passam a distinguir
  o diagnóstico histórico do resultado E0.

## Reproduzir E0 isoladamente

Na raiz do projeto, com as dependências existentes:

```powershell
.\.venv\Scripts\python.exe -B -m olist_modeling.importance --root . --figure
```

O comando carrega `validation_Ridge.joblib` e utiliza os 12.675 pedidos da
validação. Não chama `fit`, seleção, engenharia ou relatório completo, nem
abre o conjunto de teste final. Se as saídas E0 já existirem, recusa a gravação;
para conferir novamente em memória, use `calculate_e0(root)` sem `save_e0`.

Os IDs são selecionados a partir das previsões históricas de validação, cujo
hash é validado contra `selection_frozen.json`. O Parquet compartilhado contém
outras coortes, mas o filtro de leitura materializa apenas esses IDs e as
colunas necessárias. A auditoria de datas também é filtrada pelos mesmos IDs.
Os hashes dos arquivos compartilhados são calculados como bytes para
integridade, sem analisar as linhas fora da validação.

### Saídas novas

- [Tabela completa corrigida](../resultados/tabelas/e0/importance_validation_permutation.csv).
- [Comparação completa com o histórico](../resultados/tabelas/e0/comparison_historical.csv).
- [Metadados, hashes, métricas e controles](../resultados/tabelas/e0/metadata.json).
- [IDs em ordem](../resultados/tabelas/e0/validation_order_ids.csv).
- [Figura E0](../resultados/graficos/e0/importance_validation.png).
- [Resultados dos testes](../resultados/tabelas/e0/pytest.xml).
- [Verificação final de preservação](../resultados/tabelas/e0/preservation_after.json).

Modelos, datasets, previsões, registros de seleção, resultados e figuras
históricos permanecem preservados. O notebook e os PDFs anteriores continuam
documentos históricos; suas saídas não foram regeneradas. A galeria HTML
mantém a figura antiga identificada como histórica e oferece uma opção E0.
O gerador de relatório completo é histórico e não foi executado nesta correção.

## VERIFY — Resultados

Configuração: `scoring="neg_mean_absolute_error"`, `n_repeats=3`,
`random_state=42`, `n_jobs=1`, **`max_samples=1.0`**.

| Feature | Histórico (3.000 nas permutações) | E0 (12.675 em ambas) |
|---|---:|---:|
| `distance_km_min` | 0,9222791971 | **0,9301540366** |
| `same_state_items_fraction` | 0,3118211723 | **0,3110156892** |
| `customer_state` | 0,3106905657 | **0,2874415714** |
| `main_category` | 0,1278795308 | **0,1135744248** |
| `distance_km_mean` | 0,0650367375 | **0,0490981555** |
| `sellers_count` | 0,0306579401 | **0,0073481655** |
| `distance_km_max` | 0,0087178164 | **−0,0056880846** |
| `items_missing_weight_count` | 0,0149302986 | **0** |
| `items_missing_dimensions_count` | 0,0149302986 | **0** |
| `items_missing_price_count` | 0,0149302986 | **0** |
| `items_missing_freight_count` | 0,0149302986 | **0** |

Os quatro controles negativos são exatamente zero nas três repetições. Peso
e dimensões são constantes na validação; preço/frete ausentes são removidos
pelo seletor. Isso não demonstra inutilidade das primeiras em outra população.

**E0 não é a simples subtração do deslocamento histórico.** A subamostra
reconciliada teria, por exemplo, importância 0,9073488986 para a distância mínima;
a nova execução completa resulta em 0,9301540366. População e permutações mudaram.

### Métricas e previsões preservadas

| Ridge na validação | Valor reproduzido |
|---|---:|
| MAE | 4,259612014663884 dias |
| RMSE | 5,751332358545968 dias |
| R² | 0,322133438132609 |
| Diferença máxima de previsão contra o arquivo salvo | **0 dia** |

O modelo e X também foram comparados em memória antes/depois do diagnóstico.
Os arquivos de modelo, dados e previsões foram verificados por hash.

### Testes executados

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_importance_e0.py -p no:cacheprovider --basetemp resultados/tabelas/e0/pytest-tmp --junitxml resultados/tabelas/e0/pytest.xml -q
```

**Seis testes passaram.** Cobrem a população idêntica em cada chamada,
controles negativos, erros heterogêneos, reprodução numérica, inferência igual
à salva, ausência de treino, leitura filtrada de validação, bloqueio do teste,
preservação dos arquivos originais e recusa de sobrescrita. O cálculo real foi
repetido em memória e produziu tabelas idênticas. O recálculo gravado também
foi executado com bloqueios de ajuste e de abertura de arquivos de teste/finais.
Não foram executados os testes históricos que treinam modelos ou leem teste.

## Limites e próximos passos

Três repetições fornecem apenas um desvio descritivo das permutações, não um
intervalo de confiança. Distâncias são correlacionadas; permutação isolada
pode produzir combinações geográficas incoerentes. Importância negativa não
autoriza retirar uma feature sem ablação com reajuste e validação apropriada.

Não houve E1–E6, mudança de features, hiperparâmetros, imputação ou seleção.
Esses experimentos dependem de autorização posterior.
