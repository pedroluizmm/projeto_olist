# Limitações e decisões metodológicas — revisão TASK 04

Revisão de documentação das TASKS 01 e 02, sem alterar os experimentos. As médias
adicionais e o veredito **APROVADO COM RESSALVAS** provêm do relato de auditoria
independente fornecido pelo usuário (Claude Opus 5.5). O laudo integral e a rubrica
oficial do professor não foram localizados entre os arquivos do projeto. Não se
consultaram novamente os dados ou as previsões do teste para obter números.

## R², erro absoluto e mudança temporal

| Estatística registrada ou fornecida pela auditoria | Valor |
|---|---:|
| R² da Ridge na validação | 0,3221 |
| R² da Ridge no teste | −0,0246 |
| MAE da Ridge no teste | 4,1081 dias |
| Redução do MAE frente ao baseline mediana | 14,87% |
| Média do alvo no ajuste final (80.043 pedidos) | 13,27 dias |
| Média do alvo no teste | 8,37 dias |
| Previsão média da Ridge no teste | 10,51 dias |
| Superestimação média aproximada | 2,15 dias |

As quatro últimas linhas foram fornecidas pela auditoria; não são novas medições
da TASK 04. O treino inicial de 64.401 pedidos tem média histórica **13,36 dias**:
não confundir esse treino com o conjunto de **ajuste final**, de média **13,27 dias**.
Os valores arredondados 10,51 e 8,37 não reproduzem exatamente o viés de 2,15 dias;
o viés informado deriva da precisão completa, não da subtração desses arredondamentos.

Para alvo não constante, o R² convencional é
`1 − Σ(y − previsão)² / Σ(y − média_do_conjunto_avaliado)²`. No teste, o centro do
denominador é **a média do teste**, não a do treino. A definição é a do
[Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
Não se reporta como R² convencional nenhuma normalização pela média do treino.

Além da queda da média, a auditoria aponta **redução da variância do alvo** no
teste. Como `SST/n` é a variância com divisor n, menor dispersão reduz a referência
por observação do denominador. Assim, mesmo com RMSE menor que na validação
(5,5279 contra 5,7513 dias), a razão entre erro quadrático e dispersão pode aumentar
e o R² cair. A mudança de média, isoladamente, não determina a variância. Não foram
recalculadas variâncias nem inventados valores de dispersão nesta revisão.

R² −0,0246 significa que o erro quadrático ficou ligeiramente acima da referência
constante que usa a média **observada no próprio teste**. Essa média é referência
retrospectiva da métrica, não uma previsão disponível antes de abrir o conjunto.
O benchmark operacional usado foi a **mediana do ajuste final**, com MAE 4,8258 dias,
e não essa média retrospectiva. Portanto, o ganho de MAE de 14,87% e o R² negativo
podem coexistir: têm perdas e referências distintas. O ganho de MAE permanece válido,
mas não prova boa calibração ou generalização temporal. A previsão média de 10,51
dias diante de entregas médias de 8,37 dias evidencia superestimação aproximada de
2,15 dias no período. Isso sugere mudança temporal de distribuição; não identifica
causa nem justifica corrigir previsões com informação do teste.

## Reconciliação completa das amostras

```text
96.470 = 91.561 + 4.909
91.561 = 64.401 + 12.675 + 14.485
 4.909 =  2.978 + 1.931
77.076 = 64.401 + 12.675
80.043 = 77.076 + 2.967
 2.978 =  2.967 + 11
81.985 = 67.379 + 14.606 = 80.043 + 1.931 + 11
```

- Coorte: 96.470 entregues com datas válidas e itens, provenientes dos 99.441 pedidos
  originais; 2.963 sem status de entregue e oito entregues sem recebimento válido
  haviam sido excluídos na TASK 01.
- Compra anterior a **15/04/2018**: 67.379 candidatos. Somente 64.401 já tinham
  recebido a entrega **antes** desse corte; 2.978 rótulos ainda não eram conhecidos.
- Compra de **15/04/2018 até antes de 21/06/2018**: 14.606 candidatos à validação.
  Somente 12.675 estavam entregues **antes de 21/06/2018**; 1.931 ficaram fora da
  validação usada para seleção.
- Compra desde **21/06/2018**: 14.485 pedidos no teste já avaliado. A soma das três
  divisões efetivas é 91.561; os 4.909 restantes permanecem no dataset original.
- Ajuste final: os 77.076 pedidos do treino inicial mais validação, acrescidos de
  2.967 pedidos do período inicial cujos rótulos amadureceram até antes de 21/06/2018.
  Total: 80.043, sem incluir pedidos de teste.
- **Onze pedidos do período inicial** ainda não estavam entregues no corte final:
  não foram reincorporados. Somam-se aos 1.931 da janela de validação ainda sem
  rótulo nesse corte, totalizando 1.942 compras anteriores ao teste fora do ajuste.

Os limites de recebimento são exclusivos (`delivery_timestamp < corte`). Pedido
excluído por maturação significa rótulo ainda indisponível naquele instante, não
pedido apagado ou necessariamente nunca entregue. Reincorporação posterior não
altera retroativamente as amostras da validação. A purga impede uso de rótulos
futuros, mas favorece entregas rápidas perto dos cortes e limita comparabilidade
com uma avaliação prospectiva não condicionada ao desfecho.

## Random Forest e seleção

A configuração foi conferida por leitura dos parâmetros do modelo salvo
`validation_RandomForest.joblib`, sem chamar `fit` ou `predict`, e corresponde ao
JSON histórico: **`max_features=1.0`, tipo float**, 80 árvores, profundidade 16,
folha mínima 40, bootstrap ativado, `max_samples=0.85`, semente 42 e duas threads.

O float **1.0** representa 100% das colunas transformadas disponíveis em cada
divisão do nó. O inteiro **1** representaria uma coluna, e não 100%. Com todas as
features consideradas e bootstrap, a configuração funciona essencialmente como
**bagging de árvores**, sem a redução aleatória do conjunto candidato de features
que ocorre com frações menores. Isso segue a
[documentação oficial da RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor).

A busca foi válida, porém limitada: quatro configurações da floresta (folha mínima
10/40 × fração de features 0,5/1,0), com número de árvores/profundidade fixados por
orçamento. Somam-se quatro configurações da Ridge e seis da árvore: 14 no total.
Não representa otimização exaustiva ou garantia de melhor configuração possível.
Não foi realizada nova otimização após a auditoria.

Ridge e Random Forest tiveram desempenho muito próximo na validação: MAEs de
4,2596 e 4,2837 dias, diferença de 0,0241 dia. A floresta teve RMSE de validação
menor (5,6991 contra 5,7513). **Não há vantagem estatisticamente demonstrada para a
Ridge.** O vencedor é o escolhido pelo MAE previamente definido para essa janela;
a simplicidade relativa da Ridge reforça a justificativa prática, sem mudar
retroativamente o critério ou alegar teste estatístico que não foi realizado.

## Rastreabilidade: o que os registros permitem afirmar

O `selection_frozen.json` **foi regravado após a selagem inicial**, acrescentando
hashes de código e o campo `code_attested_at_utc`. O histórico da execução relata
que escolha, parâmetros e métricas de validação permaneceram iguais; não há uma
cópia exata do arquivo anterior que permita provar essa identidade byte a byte.
Portanto, o nome “frozen” não deve ser interpretado como garantia de imutabilidade.

Os campos atualmente preservados registram:

| Registro | Horário UTC de 03/11/2026 |
|---|---|
| Selagem informada (`sealed_at_utc`) | 17:00:24.313019 |
| Atestação posterior (`code_attested_at_utc`) | 17:03:17.261299 |
| Início da avaliação (`started_at_utc`) | 17:03:20.360192 |

Esses horários descrevem os registros atuais; não são prova independente de toda
a sequência de execução. O hash da versão atual é referenciado por `final_fit.json`
e `test_evaluation_started.json`. Isso vincula esses registros à versão **posterior
à regravação**, não recupera a versão original.

Também **não existe snapshot exato da versão anterior de `run.py`** usada durante
parte da execução. O manifesto e o código atual não permitem reconstruir esse
estado anterior integralmente. Não foi criado um snapshot retroativo ou uma nova
selagem para encobrir a lacuna.

Evidências disponíveis, preservadas: protocolo, grades/candidatos, modelos e
previsões salvos, hashes atuais, registros de ajuste e abertura, 53 testes e 77
verificações históricos, e cinco refits técnicos registrados. Esses refits
reproduziram as previsões: diferença máxima zero nos ajustes registrados da Ridge,
árvore e baseline; cerca de 7,1×10⁻¹⁵ dia na floresta. Demonstram reprodutibilidade
numérica das configurações disponíveis, **não a custódia perfeita da execução
original nem ausência absoluta de consulta ao teste**.

A auditoria independente **relatada pelo usuário** não encontrou vazamento
confirmado, erro nas métricas ou sinal de consulta antecipada ao teste. Essa
conclusão convive com a ressalva de rastreabilidade. Na TASK 04 os registros foram
apenas lidos ou comparados por hash; nenhum foi alterado retroativamente.

## Demais limites e decisão de encerramento

Permanecem: seleção de entregues e possível censura final; cadastros sem histórico
de vigência; centroides por CEP em lugar de rotas; uma única janela de validação;
clientes recorrentes entre períodos; correlação entre atributos; atrasos raros mal
previstos; ausência de fatores operacionais e de intervalos calibrados. Não há
garantia de desempenho em outros anos nem interpretação causal das importâncias.

A decisão desta tarefa é preservar os experimentos e apresentar resultados com
essas ressalvas. Futuras avaliações exigiriam outro protocolo e autorização própria;
não foram executadas nesta revisão. O conteúdo técnico pode orientar slides,
mantendo a [rubrica e suas pendências](rubrica_academica.md) e a limitação de
rastreabilidade visíveis.
