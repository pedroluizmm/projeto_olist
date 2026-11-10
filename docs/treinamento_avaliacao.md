# TASK 02 — EDA, pré-processamento e treinamento Olist

> Revisão documental TASK 04 após auditoria independente relatada pelo usuário:
> **APROVADO COM RESSALVAS**. Métricas e experimentos permanecem os históricos da
> TASK 02. As médias adicionais são as fornecidas pela auditoria; não houve novo
> acesso aos dados/previsões de teste, inferência, ajuste ou busca nesta revisão.

## Resultado e alcance

Modelo escolhido **Ridge**, pelo menor MAE de validação, antes de abrir o teste.
No teste: **MAE 4.1081 dias**, RMSE 5.5279 dias e
R² -0.0246. O modelo reduziu o MAE em 14.87% frente à mediana do ajuste final. R² negativo indica desempenho quadrático inferior à média observada do teste; isso limita o resultado.
MAE significa desvio absoluto médio da previsão em relação à entrega real; RMSE
penaliza mais os erros grandes. Esses números não são garantia por pedido.

### Interpretação corrigida de R² e calibração temporal

O R² da validação é **0,3221**; no teste, **−0,0246**. O MAE de teste da Ridge é
**4,1081 dias**, com redução de **14,87%** frente ao baseline mediana. O R² negativo
não invalida esse ganho de MAE: as duas métricas têm perdas e referências distintas.

Na definição convencional do Scikit-learn,
`R² = 1 − Σ(y − previsão)² / Σ(y − média_do_conjunto_avaliado)²`. A média do
denominador no teste é a **média do teste**. Não foi substituída pela média do
treino, nem se apresenta uma normalização alternativa como R² convencional.
Ver [definição oficial](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

Segundo os valores fornecidos pela auditoria, a média do alvo no **ajuste final**
foi **13,27 dias**, contra **8,37 dias no teste**; a previsão média foi **10,51 dias**,
com superestimação aproximada de **2,15 dias**. A média de **13,36 dias** mencionada
na EDA refere-se ao **treino inicial**, não ao ajuste final. As médias publicadas
são arredondadas: o viés de 2,15 dias não deve ser recalculado pela subtração dos
arredondamentos 10,51 e 8,37.

A auditoria também aponta redução da variância do alvo no teste. Como `SST/n`
corresponde à variância com divisor n, menor dispersão torna mais exigente a
referência quadrática do R². Assim, o RMSE pode diminuir de 5,7513 para 5,5279 dias
e ainda assim o R² cair. Queda de média e redução de variância são fenômenos
distintos; não foram recalculadas variâncias nesta tarefa.

O R² negativo indica erro quadrático acima da referência retrospectiva que prevê
a média observada do teste. Essa referência não é o benchmark operacional mediana
do ajuste final, cujo MAE foi 4,8258 dias. O resultado evidencia limitações de
generalização e calibração temporal, inclusive superestimação média. Não motivou
recalibração com dados de teste ou alteração dos experimentos.

O alvo é compra–recebimento em segundos/86.400, sem truncamento. A coorte contém
96.470 pedidos efetivamente entregues, com datas válidas e itens; não entregues não
foram tratados como entregas rápidas nem como tempo zero. O modelo estima pedidos
eventualmente entregues observados neste dump, sob hipótese de vigência dos cadastros.
Não estima probabilidade de cancelamento, entrega ou censura.

## Auditoria das 30 features

Features e alvo da TASK 01 foram reconstruídos em memória e coincidem exatamente
com os artefatos preservados. Não se encontrou uso direto de eventos posteriores
nas entradas. `feature_temporal_audit.csv` documenta cada atributo individualmente.
Status, aprovação, despacho, recebimento, limite de despacho, prazo estimado,
pagamentos, avaliações e IDs não entram em X. Datas posteriores são usadas apenas
para formar o alvo e verificar disponibilidade do rótulo nos cortes.

Cadastros de produtos/vendedores e valores dos itens não têm versões históricas;
o dump não comprova que nunca foram revisados. Coordenadas são medianas por prefixo
de CEP, deduplicadas e filtradas pela caixa ampla documentada na TASK 01: podem
representar áreas extensas, sem rota ou localização exata. São hipóteses explícitas,
não uma prova de disponibilidade operacional histórica.

## Divisão cronológica e maturação

Os cortes próximos aos percentis 70%/85% de compra foram arredondados ao início do
dia: validação desde **15/04/2018**, teste desde **21/06/2018**. Nenhum dia foi
partido. Antes de purgar rótulos tardios, são 67.379/14.606/14.485 pedidos.

| split | rows | purchase_min | purchase_max |
| --- | --- | --- | --- |
| train_candidates | 67379 | 2016-09-15 12:16:38 | 2018-04-14 23:28:11 |
| train | 64401 | 2016-09-15 12:16:38 | 2018-04-13 13:01:55 |
| validation_candidates | 14606 | 2018-04-15 00:08:41 | 2018-06-20 23:39:24 |
| validation | 12675 | 2018-04-15 00:08:41 | 2018-06-19 13:09:39 |
| test | 14485 | 2018-06-21 00:22:02 | 2018-08-29 15:00:37 |
| final_train | 80043 | 2016-09-15 12:16:38 | 2018-06-19 13:09:39 |

Foram excluídos 2978 rótulos ainda desconhecidos no
início da validação e 1931 da validação ainda
desconhecidos no início do teste. Por isso os tamanhos efetivos se afastam de 70/15/15;
não foram substituídos por sorteio. Todos os modelos comparam os mesmos IDs elegíveis.
O reajuste final usa 80043 pedidos anteriores ao teste, já entregues
antes desse corte, incluindo rótulos de treino que amadureceram após 15/04/2018.

Reconciliação explícita, sem alteração das divisões originais:

```text
96.470 = 91.561 + 4.909
91.561 = 64.401 + 12.675 + 14.485
 4.909 =  2.978 + 1.931
77.076 = 64.401 + 12.675
80.043 = 77.076 + 2.967
 2.978 =  2.967 + 11
```

Os 4.909 representam rótulos indisponíveis nos cortes do desenvolvimento, não
pedidos apagados da coorte. Dos 2.978 do período inicial, 2.967 puderam entrar no
ajuste final quando seus rótulos amadureceram. **Onze pedidos iniciais continuavam
sem entrega conhecida no corte de 21/06/2018**, permanecendo fora desse ajuste,
além dos 1.931 da janela de validação. Os limites de entrega são exclusivos
(`delivery_timestamp < corte`). A reincorporação de 2.967 não altera
retroativamente o treino inicial nem a validação de seleção.

Purgar evita treinamento/seleção com rótulos futuros, mas favorece entregas rápidas
perto dos cortes: `label_maturation_diagnostic.csv` quantifica esse viés. A avaliação
é retrospectiva, condicionada à entrega observada, e não representa um experimento
prospectivo completo. O teste mantém o conjunto final de entregues do dump, cuja data
de extração oficial é desconhecida; possível censura no fim do período permanece.

Há 282 pedidos de validação
e 320 do teste de clientes já vistos
no respectivo treino. Clientes não foram agrupados entre cortes; IDs não são entradas.
O cenário corresponde à mistura de clientes novos/recorrentes da operação, sem
garantia de generalização para clientes inéditos.

## EDA (somente treino elegível)

O alvo tem mediana 11.26 dias, média 13.36 e máximo
209.63 no treino, com cauda à direita. Spearman entre distância média
e alvo: 0.533; mesma UF e alvo:
-0.490; frete e alvo:
0.405. As três distâncias resumem uma
informação semelhante e são fortemente correlacionadas; não são evidências independentes.
Medianas por UF variam de 8.00 a 26.89
dias, com tamanhos amostrais exibidos nos gráficos. Diferenças de composição e período
podem explicar associações; não se atribui causalidade.

Figuras 01–06: distribuições (eixos log para valor/peso), Spearman, relações com o alvo,
medianas por UF, dispersão geográfica de CEPs com ≥20 pedidos e evolução mensal.
O gráfico geográfico usa graus e não requer limites cartográficos externos. IQR é
diagnóstico: extremos válidos continuam na EDA, ajuste e avaliação. Caixas ocultam
apenas a marcação visual de extremos, não os removem dos cálculos.

## Pré-processamento e features finais

Contrato de entrada: as 30 features da TASK 01, listadas em `docs/features_task02.md`.
Numéricas: mediana ajustada apenas no treino; os contadores de ausência originais
permanecem. Categóricas: `customer_state, main_category, purchase_weekday, purchase_month`; ausência vira `__MISSING__` e
one-hot com categorias desconhecidas ignoradas. Mês e dia da semana são categorias,
evitando tratá-los como uma ordem linear. Hora/fim de semana mantêm a definição
original. Tipos são normalizados por função determinística dentro do pipeline salvo.

Ridge utiliza escala robusta por mediana/IQR; árvores não recebem escala. Não houve
ajuste de winsorização, corte do alvo ou remoção automática de caudas legítimas.
RobustScaler, regularização e limites de profundidade/folha reduzem sensibilidade;
não eliminam a dificuldade de prever extremos. `VarianceThreshold(0)` seleciona
colunas não constantes após transformação, sem seleção supervisionada por correlação.
As constantes removidas na validação do escolhido foram: `numeric__items_missing_price_count, numeric__items_missing_freight_count`.
O reajuste final produziu **144 colunas transformadas selecionadas**.
Vocabulário/constantes podem mudar ao reajustar, sempre antes do teste.

Todas as etapas ajustáveis pertencem a `Pipeline`/`ColumnTransformer`, conforme a
[documentação de prevenção de leakage do Scikit-learn](https://scikit-learn.org/stable/common_pitfalls).
Não se ajustou nenhum transformador sobre o dataset completo para avaliar modelos.

## Busca, orçamento e decisão

Baseline: mediana do treino. Ridge, DecisionTree e RandomForest usam as mesmas
amostras e MAE como critério principal. `GridSearchCV` com um `PredefinedSplit`
temporal e `refit=False` avalia 4 + 6 + 4 configurações. Uma única janela de
validação é reutilizada para hiperparâmetros/modelo; seus resultados têm viés de
seleção e não são uma avaliação independente. Não se usou CV aleatória.
O comportamento de `refit`/múltiplas métricas segue o
[GridSearchCV oficial](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV).

A vantagem em MAE do escolhido para o segundo colocado (RandomForest) foi apenas
0.0241 dia. Não foram estimados intervalos para essa diferença; ela não demonstra
superioridade estável entre janelas. O critério foi mantido conforme o plano.

Ridge e Random Forest devem ser apresentados como **modelos de desempenho muito
próximo na validação, sem vantagem estatisticamente demonstrada para a Ridge**.
A floresta teve RMSE menor nessa janela. A escolha da Ridge respeita o menor MAE
previamente definido; sua simplicidade relativa é uma justificativa prática
adicional, não um novo critério aplicado ao teste.

Semente 42. Busca sequencial; floresta: 80 árvores, profundidade máxima 16,
bootstrap 85%, no máximo duas threads. Não houve busca extensa ou garantia de
ótimo global. Previsões de todos os regressores têm piso físico zero no estimador
salvo, fixado antes das métricas. Grades completas e tempos: `search_candidates.csv`.

Na configuração efetivamente salva da floresta, `max_features=1.0` é **float**:
considera todas as features transformadas disponíveis em cada divisão de um nó.
`max_features=1`, inteiro, consideraria apenas uma feature. Com bootstrap e todas
as features candidatas, a configuração funciona essencialmente como **bagging de
árvores**, conforme a
[documentação oficial](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor).
A grade de quatro configurações da floresta foi válida, porém limitada, e não
demonstra ótimo global. Os parâmetros foram conferidos no modelo salvo sem
treinamento/inferência. Não foi feita nova otimização.

Hiperparâmetros escolhidos por modelo:

- **Baseline:** `{"strategy": "median"}`.
- **DecisionTree:** `{"model__estimator__max_depth": 10, "model__estimator__min_samples_leaf": 20}`.
- **RandomForest:** `{"model__estimator__max_features": 1.0, "model__estimator__min_samples_leaf": 40}`.
- **Ridge:** `{"model__estimator__alpha": 1.0}`.

## Validação e teste final

| model | MAE_days | RMSE_days | R2 |
| --- | --- | --- | --- |
| Ridge | 4.2596 | 5.7513 | 0.3221 |
| RandomForest | 4.2837 | 5.6991 | 0.3344 |
| DecisionTree | 4.3972 | 5.8639 | 0.2953 |
| Baseline | 5.4554 | 7.0402 | -0.0157 |

Os registros relatam a escolha antes do teste, mas `selection_frozen.json` foi
**regravado após a selagem inicial**, para acrescentar atestação de código. Seu
hash posterior aparece no ajuste final e no registro de abertura; não comprova
imutabilidade da versão inicial. Após reajustar o escolhido e o baseline com os
rótulos disponíveis em 21/06/2018, realizou-se um evento de avaliação final:

| role | model | MAE_days | RMSE_days | R2 |
| --- | --- | --- | --- | --- |
| selected | Ridge | 4.1081 | 5.5279 | -0.0246 |
| baseline_reference | Baseline | 4.8258 | 6.0796 | -0.2394 |

A EDA não usou validação/teste. Importância foi calculada na validação após selar
a escolha; erros do teste são análise descritiva posterior, sem mudança de modelo.

## Erros e importância

45.45% das previsões ficaram a até três dias do real e
68.66% a até cinco. Erro absoluto P90:
8.23 dias; P95: 9.88.
Resíduo médio (real − previsto): -2.15 dias; sinal positivo
representa subestimação. Gráficos 08–09 mantêm todos os pontos em densidade e
evidenciam caudas/erros sistemáticos; não indicam intervalos de confiança por pedido.
O sinal médio negativo observado indica superestimação média no período de teste.
Nos 38 pedidos com duração ≥40 dias, o MAE foi 39.42 dias; os atrasos raros continuam mal previstos.
As 30 maiores falhas e erros por UF, mês, vendedores, distância ausente e faixa
do alvo estão nas tabelas `test_errors_*`/`largest_test_errors.csv`. Faixas do alvo
são usadas somente para diagnóstico posterior, nunca como features.

**Correção E0:** a tabela histórica de importância comparava a referência nos
12.675 pedidos com permutações em 3.000 pedidos, introduzindo deslocamento de
+0,014930298556740773 dia. Ela permanece preservada como registro histórico.
O recálculo E0 usa o modelo de validação salvo e `max_samples=1.0`, com três
repetições e semente 42: `distance_km_min` **0,930154 dia**,
`same_state_items_fraction` **0,311016 dia** e `customer_state` **0,287442 dia**.
São novos resultados da validação completa, não a subtração do deslocamento da
subamostra. As quatro colunas de controle negativo têm importância zero.
Modelos, previsões, métricas e seleção permanecem inalterados; não houve treino
ou acesso ao teste. [Procedimento, tabela, figura e verificações E0](correcao_e0.md).
As três repetições e a correlação entre features limitam a interpretação;
o desvio entre permutações não é intervalo de confiança nem evidência causal.
Importâncias/coefs internos do modelo final também foram exportados. Importância
por impureza pode favorecer variáveis contínuas/com muitas categorias; coeficientes
dependem da escala e regularização. Nenhuma importância foi usada para retunar no teste.

## Verificações históricas, rastreabilidade e limitações

Na TASK 02 foram registrados 53 testes, 0 falhas/erros e 0 ignorados, e 77 verificações de integridade, splits, métricas,
pré-processamento, recarga e reprodutibilidade registradas; refits completos de
configurações seladas: True. As métricas foram
conferidas com somas/fórmulas independentes, e as previsões de refits devem coincidir
a 1e-9 dia. Refits técnicos repetem inferência, sem nova decisão ou nova seleção.
Naquela etapa, 47 arquivos anteriores foram preservados byte a byte; CRC e
SHA-256 dos ZIPs foram verificados. Esses são resultados históricos, não testes
reexecutados na TASK 04. A revisão documental atual tem sua própria
[verificação de preservação](task04/verificacao.md). O Projeto ENADE não foi modificado.

A ressalva de rastreabilidade permanece: houve regravação de `selection_frozen.json`
após a selagem e **não há snapshot exato da versão anterior de `run.py`**. Os registros
atuais informam selagem às 17:00:24.313019 UTC, atestação de código às
17:03:17.261299 UTC e início da avaliação às 17:03:20.360192 UTC, em 03/11/2026.
Esses horários e hashes não reconstituem integralmente a execução anterior.

Modelos, previsões, logs e cinco refits técnicos registrados sustentam
reprodutibilidade numérica (diferença máxima cerca de 7,1×10⁻¹⁵ dia na floresta,
zero nos demais ajustes registrados). Não provam custódia perfeita da selagem.
A auditoria independente relatada pelo usuário **não encontrou sinal de consulta
antecipada ao teste**, vazamento confirmado ou erro nas métricas. Não se deve
transformar essa conclusão em prova absoluta; detalhes e fontes em
[limitações e decisões](limitacoes_decisoes.md). Nenhum registro foi regravado
retroativamente na TASK 04.

Limitações: seleção de entregues/censura, viés de maturação, uma janela temporal,
cadastros sem vigência, centroides aproximados, correlação entre features, dependência
entre clientes recorrentes e ausência de fatores operacionais como transportadora,
rota, clima, capacidade e incidentes. O desempenho não autoriza inferência causal,
intervalos calibrados ou compromisso operacional. Não se conhece a estabilidade
em outros anos; o teste não foi reutilizado para corrigir esse problema.

Pendências: validar alcance condicionado à entrega; obter snapshot/histórico dos
cadastros; planejar novas janelas temporais e rótulos maduros; avaliar estabilidade
geográfica/temporal e incerteza antes de produção. Nenhuma nova busca foi disparada
para melhorar o resultado após conhecer o teste.

Reprodução histórica e arquivos: [README_task02.md](README_task02.md). A versão
oficial da rubrica não foi localizada no projeto. A
[conferência acadêmica](rubrica_academica.md) utiliza os requisitos fornecidos pelo
usuário: problema 10%, pré-processamento 30% (mínimo quatro procedimentos), EDA 10%
(mínimo quatro análises), treinamento 30% (três modelos e otimização) e avaliação/
conclusão 20% (MAE, RMSE, R² e benchmark). O conteúdo técnico cobre esses itens com
as ressalvas descritas; a nota depende da avaliação do professor.

A entrega final informada exige vídeo de **10 a 20 minutos**, por slides, com
participação de todos os integrantes e envio do link do **YouTube pelo AVA**.
Esses materiais e o envio continuam pendentes; não foram criados nesta revisão.
