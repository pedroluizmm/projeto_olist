# TASK 02 — READ → PLAN → APPLY → VERIFY

## READ

Inspecionados o pedido completo, contrato das 30 features, código da TASK 01,
documentação, dataset tratado e auditoria de seleção. São 96.470 pedidos únicos.
Compras entre 15/09/2016 e 29/08/2018; recebimentos até 17/10/2018.
Eventos posteriores não são entradas. Cadastros/geolocalização não têm histórico
de vigência; essa limitação não pode ser resolvida com o dump disponível.

## PLAN (definido antes dos resultados de treinamento)

1. Preservar ZIPs e arquivos da TASK 01 por SHA-256. Saídas novas em subpastas
   `task02/`, código em `src/olist_modeling/` e dependências em arquivos próprios.
2. Recalcular features e alvo em memória, comparando-os aos dados da TASK 01;
   auditar individualmente as 30 features e os IDs/tempos usados nas divisões.
3. Dividir por dias de compra próximos aos percentis 70%/85%: validação a partir
   de **15/04/2018**, teste a partir de **21/06/2018**. O mesmo dia não será partido.
4. Para evitar rótulos do futuro, treinar somente com pedidos entregues antes de
   15/04/2018 e avaliar candidatos somente na validação entregue antes de 21/06/2018.
   Registrar separadamente os pedidos excluídos por maturação do rótulo.
5. EDA apenas no treino elegível: distribuições, estatísticas, ausências, IQR,
   correlações, relações com o alvo, evolução mensal e análise por UF/CEP.
6. Manter as 30 entradas como contrato. Mediana numérica e categoria ausente
   explícita; one-hot para UF, categoria, mês e dia da semana; escala robusta
   (mediana/IQR) na Ridge; sem escala nas árvores. `VarianceThreshold(0)` remove
   constantes após transformação. Todos os ajustes dentro de cada `Pipeline`.
7. Preservar extremos fisicamente válidos, inclusive alvos longos. Ausências e
   invalidades já sinalizadas na TASK 01; não aparar com base em estatísticas globais.
8. Comparar baseline mediana, Ridge, árvore e Random Forest no mesmo treino/validação.
   `GridSearchCV`, um `PredefinedSplit` cronológico, `refit=False`, MAE como critério;
   desempate por RMSE e nome. A validação é a seleção, não uma estimativa independente.
9. Grades limitadas: Ridge alpha {0.1,1,10,100}; árvore profundidade {6,10,14} ×
   folha {20,80}; floresta 80 árvores, profundidade 16, folha {10,40} ×
   max_features {0.5,1.0}, bootstrap de 85%. Busca sequencial, floresta com 2 threads,
   sementes 42. Previsões truncadas em zero pela restrição física do problema.
10. Selar escolha e parâmetros **antes** da avaliação de teste. Reajustar apenas o
    escolhido e a referência em todas as compras anteriores ao teste cujos rótulos
    já eram conhecidos em 21/06/2018. Consultar o teste uma única vez para avaliação,
    sem trocar de modelo ou parâmetros após conhecer seu resultado.
11. Produzir MAE/RMSE/R², previsões, resíduos, erros por grupos e importância
    descritiva (sem causalidade). Salvar pipelines e contrato de inferência.
12. Verificar métricas com fórmulas independentes, transformações, maturação,
    disjunção de IDs, integridade, recarga dos modelos e refits reprodutíveis.
    Executar testes e notebooks; documentar limites e pendências.

## Limites definidos antecipadamente

Uma única janela de validação não estima variabilidade entre períodos. Clientes
recorrentes podem atravessar cortes (IDs não são entradas); medir a sobreposição.
Purgar por recebimento evita usar rótulos ainda desconhecidos, mas pode favorecer
entregas rápidas perto dos cortes; quantificar e discutir. O teste continua
condicionado à entrega observada no dump, sem data oficial de extração conhecida.
Não serão criados slides, vídeo ou arquivos em outros projetos.
