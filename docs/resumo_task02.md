# Olist — resumo técnico corrigido após auditoria

**Ciência dos Dados (T326), UNIFOR — TASKS 01/02; revisão documental TASK 04.**
Veredito da auditoria independente relatado pelo usuário: **APROVADO COM RESSALVAS**.
Não houve novos experimentos, consulta ao teste ou alteração dos registros nesta revisão.

## Problema, entradas e processamento

Prever o tempo compra–recebimento, em dias, no instante da compra. Unidade: pedido;
coorte: 96.470 entregues com datas válidas. São 30 entradas de carrinho, produtos,
vendedores, UF, distância e calendário. Eventos posteriores e IDs ficam fora de X.
Cadastros sem histórico e distâncias aproximadas por CEP limitam a disponibilidade
histórica comprovável e a interpretação operacional.

Mediana numérica, categoria ausente explícita, one-hot, escala robusta na Ridge e
remoção de constantes foram ajustados somente no treino, dentro de pipelines.
Extremos válidos foram preservados. Duas entradas constantes foram removidas na
transformação: 143 colunas selecionadas no treino inicial e 144 no ajuste final.
EDA restrita ao treino incluiu distribuições, outliers, correlações, relações com
o alvo, geografia e evolução mensal, sem atribuir causalidade.

## Reconciliação e maturação dos rótulos

```text
96.470 = 91.561 + 4.909
91.561 = 64.401 + 12.675 + 14.485
 4.909 =  2.978 + 1.931
77.076 = 64.401 + 12.675
80.043 = 77.076 + 2.967
 2.978 =  2.967 + 11
```

Treino inicial: 64.401 pedidos, com compra e entrega antes de 15/04/2018.
Validação: 12.675 compras de 15/04 até antes de 21/06/2018, entregues antes do
segundo corte. Teste reservado: 14.485 compras desde 21/06/2018, já avaliadas.
Ficaram fora do desenvolvimento 2.978 + 1.931 rótulos ainda desconhecidos nos
respectivos cortes; não foram apagados da coorte. A purga evita rótulos futuros,
mas favorece entregas rápidas perto dos limites.

O ajuste final usou 80.043 pedidos: os 77.076 do treino inicial mais validação e
2.967 pedidos iniciais cujos rótulos amadureceram antes de 21/06/2018. **Onze
pedidos do período inicial ainda não estavam entregues nesse corte** e continuaram
fora, além dos 1.931 da janela de validação. Todos os limites de entrega são exclusivos.

## Modelos e decisão

Baseline mediana e três modelos; 14 configurações em `GridSearchCV` cronológico,
sementes 42 e orçamento limitado. Resultados históricos de validação:

| Modelo | MAE (dias) | RMSE (dias) | R² |
|---|---:|---:|---:|
| Ridge | 4,2596 | 5,7513 | 0,3221 |
| Random Forest | 4,2837 | 5,6991 | 0,3344 |
| Árvore | 4,3972 | 5,8639 | 0,2953 |
| Baseline mediana | 5,4554 | 7,0402 | −0,0157 |

Ridge e Random Forest tiveram desempenho **muito próximo**, sem vantagem
estatisticamente demonstrada para a Ridge. A diferença de MAE foi 0,0241 dia;
a floresta teve RMSE menor. A escolha da **Ridge, alpha=1**, respeitou o critério
de MAE previamente definido e é compatível com sua simplicidade relativa.

Na floresta salva, `max_features=1.0` é **float: todas as features em cada divisão**;
o inteiro `max_features=1` significaria uma feature. Com bootstrap, a configuração
funciona essencialmente como **bagging de árvores**: 80 árvores, profundidade 16,
folha mínima 40 e amostra bootstrap de 85%. A busca foi válida, porém limitada;
nenhuma nova otimização foi realizada.

## Avaliação e interpretação corrigida

| Modelo no teste histórico | MAE (dias) | RMSE (dias) | R² |
|---|---:|---:|---:|
| Ridge | **4,1081** | 5,5279 | **−0,0246** |
| Baseline mediana do ajuste final | 4,8258 | 6,0796 | −0,2394 |

O MAE diminuiu **14,87%** frente ao baseline. O R² negativo **não invalida esse
ganho**, mas evidencia limitações de generalização e calibração temporal. O R²
convencional usa a média do **próprio conjunto avaliado**, não a média do treino;
sua referência não é o baseline mediana usado na comparação de MAE.

Segundo a auditoria fornecida, a média do alvo foi **13,27 dias no ajuste final**,
**8,37 dias no teste**, e a previsão média **10,51 dias**, com superestimação média
aproximada de **2,15 dias**. Os valores são arredondados; o viés não é a simples
subtração desses arredondamentos. A redução da variância do alvo no teste torna
menor o denominador por observação do R²: mesmo um RMSE inferior ao da validação
pode resultar em R² pior. Não houve recálculo ou recalibração com dados do teste.

## Rastreabilidade, evidências e próximos passos

`selection_frozen.json` foi **regravado após a selagem inicial** para incluir
atestação de código. Não existe snapshot exato da versão anterior de `run.py`.
Os hashes atuais vinculam os registros disponíveis, mas não reconstituem toda a
execução anterior. A auditoria relatada pelo usuário **não encontrou sinal de
consulta antecipada ao teste**, vazamento confirmado ou erro nas métricas.

Há 53 testes e 77 verificações históricos, além de cinco refits técnicos registrados
com previsões reproduzidas. São evidências numéricas, não prova de custódia perfeita;
nenhum desses processos foi reexecutado na TASK 04. Permanecem também seleção de
entregues/censura, maturação dos rótulos, uma janela temporal, clientes recorrentes
entre cortes e cadastros sem versões.

O conteúdo está pronto para orientar slides com essas ressalvas. A rubrica oficial
não foi localizada: a conferência usa os requisitos informados pelo usuário.
A entrega final informada exige **vídeo de 10 a 20 minutos**, por slides, com
**todos os integrantes**, e envio do link do **YouTube pelo AVA**. Slides, vídeo e
envio continuam pendentes; não foram criados nesta tarefa.

Detalhes: [relatório](treinamento_avaliacao.md), [limitações e decisões](limitacoes_decisoes.md)
e [rubrica acadêmica](rubrica_academica.md).
