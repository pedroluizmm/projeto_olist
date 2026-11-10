# Conferência da rubrica acadêmica — TASK 04

Não foi localizado documento oficial do professor no projeto durante a inspeção
dos arquivos disponíveis. A tabela usa **os requisitos fornecidos pelo usuário**;
não constitui transcrição de um documento oficial nem garantia de nota. Confirmar
a versão oficial do enunciado e eventuais exigências adicionais antes do envio.

| Critério informado | Peso | Evidência existente | Situação documental |
|---|---:|---|---|
| Definição do problema | 10% | README; `engenharia_features.md`; `treinamento_avaliacao.md`: regressão, pedido como unidade, alvo compra–recebimento e momento da previsão | Coberto, com alcance restrito aos entregues |
| Pré-processamento: pelo menos quatro procedimentos | 30% | Agregação por pedido e junções validadas; invalidades e indicadores de ausência; imputação por mediana; categoria ausente explícita/one-hot; escala robusta da Ridge; remoção de constantes por `VarianceThreshold(0)` | Pelo menos quatro procedimentos distintos documentados; ajustes somente no treino |
| EDA: pelo menos quatro análises | 10% | Distribuições/estatísticas e ausências; outliers por IQR; correlações; relações com o alvo; geografia por UF/CEP; evolução mensal. Figuras 01–06 e tabelas `eda_train_*` | Pelo menos quatro análises, restritas ao treino e sem causalidade |
| Treinamento: três modelos e otimização | 30% | Ridge, DecisionTreeRegressor e RandomForestRegressor; `GridSearchCV` cronológico; 4 + 6 + 4 configurações; `search_candidates.csv` | Coberto por busca válida, porém limitada; não otimização exaustiva |
| Avaliação e conclusão | 20% | MAE, RMSE, R² e baseline mediana; validação comparativa; teste final histórico; resíduos, grandes erros e importância; interpretação corrigida do R² e seleção | Coberto com ressalvas de calibração temporal, seleção e rastreabilidade |
| **Total dos pesos técnicos** | **100%** | Pesos informados pelo usuário | Não equivale a atribuição de nota |

## Entrega final informada

- Vídeo de **10 a 20 minutos**, apresentado por slides.
- **Participação de todos os integrantes**; definir nomes e distribuição da fala.
- Publicação no **YouTube** e envio do **link pelo AVA**.
- Conferir acesso ao vídeo e as instruções oficiais de envio/prazo.

Não há evidência nesta revisão de slides finais, vídeo, participação gravada ou
submissão pelo AVA. Esses itens **permanecem pendentes** e não foram produzidos na
TASK 04. O documento oficial também permanece pendente de confirmação.

## Prontidão para produzir slides

**Sim, quanto ao conteúdo técnico documentado**, preservando o veredito relatado
de aprovado com ressalvas. Usar o [resumo corrigido](resumo_task02.md), os gráficos
existentes e o [relatório detalhado](treinamento_avaliacao.md). Explicar:

1. Problema, informação disponível na compra e viés de selecionar entregues.
2. Procedimentos de pré-processamento e EDA que atendem aos mínimos informados.
3. Reconciliação dos 96.470 pedidos, exclusões por maturação e 11 casos residuais.
4. Três modelos e busca limitada; Ridge/RF próximos, sem superioridade estatística.
5. MAE/RMSE/R² e baseline; mudança temporal, superestimação e redução da variância.
6. Ressalva de rastreabilidade, evidências disponíveis e limites das conclusões.

Essa prontidão não significa que a entrega audiovisual ou a submissão já estejam
concluídas. Não são necessários novos experimentos para apresentar honestamente
os resultados existentes; a confirmação da rubrica oficial continua necessária.
