# Projeto Olist — tempo de entrega

Ciência dos Dados (T326), UNIFOR — Projeto 2: regressão supervisionada do tempo
entre compra e recebimento, em dias, usando informações disponíveis na compra.
Unidade de análise: um pedido. O alcance é condicionado aos pedidos entregues
observados no dataset; não inclui previsão de cancelamento ou probabilidade de entrega.

## Estado da entrega

TASK 01: preparação e 30 features. TASK 02: EDA, pré-processamento, baseline,
Ridge, DecisionTree e RandomForest treinados, com busca limitada de hiperparâmetros.
TASK 04: revisão documental após auditoria independente, relatada pelo usuário como
**APROVADO COM RESSALVAS**. Nesta revisão não houve treinamento, inferência, nova
consulta ao teste, alteração de métricas ou modificação de registros históricos.

| Modelo | MAE validação (dias) | RMSE validação (dias) | R² validação |
|---|---:|---:|---:|
| Ridge | 4,2596 | 5,7513 | 0,3221 |
| Random Forest | 4,2837 | 5,6991 | 0,3344 |
| Árvore | 4,3972 | 5,8639 | 0,2953 |
| Baseline mediana | 5,4554 | 7,0402 | −0,0157 |

Ridge (`alpha=1`) e Random Forest tiveram desempenho muito próximo na validação,
sem vantagem estatisticamente demonstrada para a Ridge. A diferença de MAE foi
0,0241 dia. A escolha respeitou o critério de MAE previamente definido; sua
simplicidade relativa também favorece interpretação e custo computacional.

No teste já avaliado: **MAE 4,1081 dias**, RMSE **5,5279 dias** e **R² −0,0246**.
O MAE diminuiu **14,87%** frente ao baseline mediana (4,8258 dias). O R² convencional
usa a média do alvo do próprio conjunto avaliado: o valor negativo não anula a
melhoria de MAE, mas revela limitações de generalização e calibração temporal.
Segundo os números da auditoria fornecidos pelo usuário, a média do alvo passou de
**13,27 dias no ajuste final** para **8,37 dias no teste**, enquanto a previsão média
foi **10,51 dias**, com superestimação média aproximada de **2,15 dias**. A menor
variância do alvo no teste reduz a referência por observação do denominador do R². Não se recalculou o teste
para esta revisão; esses números são arredondados e suas diferenças não devem
substituir estatísticas calculadas originalmente com precisão completa.

## Amostras e processamento

- **96.470 = 91.561 + 4.909**: coorte rotulada = amostras das divisões efetivas +
  rótulos indisponíveis nos cortes de desenvolvimento.
- **91.561 = 64.401 + 12.675 + 14.485**: treino inicial + validação + teste reservado.
- **4.909 = 2.978 + 1.931**: exclusões por maturação no treino inicial e na validação.
- **80.043 = 77.076 + 2.967**: ajuste final = treino inicial e validação
  (**77.076 = 64.401 + 12.675**) + rótulos iniciais que amadureceram.
- **2.978 = 2.967 + 11**: onze pedidos do período inicial ainda não estavam
  entregues no corte final de 21/06/2018; continuaram fora do ajuste final.

Cortes por compra: validação desde 15/04/2018; teste desde 21/06/2018. Treino inicial
exige entrega anterior a 15/04/2018; validação e ajuste final exigem entrega anterior
a 21/06/2018, sempre com limite exclusivo. A purga evita usar rótulos futuros, mas
favorece entregas rápidas perto dos cortes. Os 4.909 pedidos não foram apagados da
coorte, e 2.967 foram reincorporados apenas quando seus rótulos estavam disponíveis.

Mediana numérica, categoria ausente explícita, one-hot, escala robusta na Ridge e
remoção de constantes foram ajustados dentro dos pipelines, somente no treino de
cada divisão. Extremos válidos foram preservados. O contrato mantém 30 entradas;
duas constantes são removidas. São **143 colunas transformadas no treino inicial**
e **144 no ajuste final**, devido ao vocabulário aprendido nos respectivos treinos.

Na configuração salva da Random Forest, **`max_features=1.0` é float e considera
todas as features em cada divisão**; `max_features=1` seria o inteiro de contagem
de uma feature. Com bootstrap, essa configuração funciona essencialmente como
bagging de árvores. A busca foi válida, porém limitada: quatro configurações da
floresta, além de quatro da Ridge e seis da árvore. Não houve nova otimização.

## Documentação e artefatos

- [Resumo técnico para o professor](docs/resumo_task02.md).
- [Treinamento, avaliação e interpretação corrigida](docs/treinamento_avaliacao.md).
- [Limitações, decisões e ressalva de rastreabilidade](docs/limitacoes_decisoes.md).
- [Reconciliação com a rubrica acadêmica](docs/rubrica_academica.md).
- [Features finais](docs/features_task02.md) e [engenharia da TASK 01](docs/engenharia_features.md).
- [Verificação documental da TASK 04](docs/task04/verificacao.md).

```text
dados/brutos/                    ZIP original, somente leitura
dados/tratados/                  CSV/Parquet e contrato das 30 entradas
src/olist_features/              engenharia e auditoria da TASK 01
src/olist_modeling/              EDA, pipelines e protocolo da TASK 02
notebooks/                      01 preparação; 02 EDA; 03 treinamento/avaliação
resultados/graficos/task02/      10 gráficos existentes
resultados/tabelas/task02/       métricas, previsões, splits e registros históricos
resultados/modelos/task02/       pipelines salvos e contrato de inferência
docs/                           relatórios atualizados e planos históricos
tests/                          testes dos experimentos
.venv/                          Python 3.13.6 e kernel Olist local
requirements*.txt               versões fixadas por etapa
```

## Consulta e reprodução histórica

Os resultados históricos registram **53 testes e 77 verificações aprovados**, além
de cinco refits técnicos reprodutíveis. Não foram reexecutados na TASK 04. Modelos,
previsões, métricas, divisões, gráficos, notebooks, código e logs permanecem intactos.
`selection_frozen.json` foi regravado após a selagem inicial para acrescentar
atestação de código; não há snapshot exato do `run.py` anterior. A auditoria relatada
pelo usuário não encontrou sinal de consulta antecipada ao teste. Essa evidência
não elimina a ressalva de rastreabilidade, detalhada no documento de limitações.

O [guia da TASK 02](docs/README_task02.md) preserva os comandos históricos de
reprodução. Eles não devem ser executados sobre esta entrega para esta revisão:
alguns refazem ajustes, leem o teste ou sobrescrevem documentos e manifestos. Os
geradores antigos não incorporam estas correções textuais. Os manifestos originais
continuam sendo snapshots das versões anteriores; não foram regravados para
legitimar retrospectivamente a nova documentação.

Verificação exclusivamente documental, sem carregar dados tabulares de teste ou
executar modelos:

```powershell
.\.venv\Scripts\python.exe -B docs/task04/verificar_documentacao.py
```

## Entrega acadêmica

O conteúdo técnico está pronto para orientar a produção dos slides, com as
ressalvas explicitadas. O documento oficial do professor não foi localizado no
projeto; a conferência utiliza os requisitos fornecidos pelo usuário, cuja versão
oficial ainda deve ser confirmada. A entrega final informada é **vídeo de 10 a 20
minutos, apresentado por slides, participação de todos os integrantes e envio do
link do YouTube pelo AVA**. Slides, vídeo e envio não foram produzidos nesta tarefa.
