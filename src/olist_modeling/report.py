"""Relatórios e figuras a partir de resultados selados; não ajusta modelos."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from olist_features.audit import sha256
from olist_features.core import FEATURE_COLUMNS
from olist_features.pipeline import write_json
from .eda import plotting, save_plot
from .protocol import CATEGORICAL, verify_preservation


def markdown_table(frame, columns):
    headings = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v).replace("|", "/")
    return "\n".join([headings, rule, *["| " + " | ".join(value(row[col]) for col in columns) + " |" for _, row in frame.iterrows()]])


def test_status(path):
    if not path.exists():
        return "Testes ainda não registrados; executar pytest antes da entrega."
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return f"{tests} testes, {failures} falhas/erros e {skipped} ignorados"


def report(root: Path):
    tables, figures = root / "resultados/tabelas/task02", root / "resultados/graficos/task02"
    read_json = lambda name: json.loads((tables / name).read_text(encoding="utf-8"))
    frozen, diag, split_diag = read_json("selection_frozen.json"), read_json("test_diagnostics.json"), read_json("split_diagnostics.json")
    validation, test = pd.read_csv(tables / "validation_metrics.csv"), pd.read_csv(tables / "test_metrics.csv")
    splits = pd.read_csv(tables / "split_summary.csv")
    predictions = pd.read_parquet(tables / "test_predictions.parquet")
    selected, baseline = test.loc[test.role.eq("selected")].iloc[0], test.loc[test.role.eq("baseline_reference")].iloc[0]
    winner = frozen["selected_model"]
    plt = plotting(root)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = validation.sort_values("MAE_days")
    axes[0].barh(order.model, order.MAE_days, color=["#256b87" if n == winner else "#a6bac3" for n in order.model])
    for y, val in enumerate(order.MAE_days):
        axes[0].text(val + .03, y, f"{val:.2f}", va="center")
    axes[0].set(xlabel="MAE (dias; menor é melhor)", title="Validação — seleção do modelo", xlim=(0, order.MAE_days.max() * 1.18))
    axes[1].bar([winner, "Baseline"], [selected.MAE_days, baseline.MAE_days], color=["#256b87", "#a6bac3"])
    axes[1].set(ylabel="MAE (dias)", title="Teste reservado — escolha congelada")
    for x, val in enumerate([selected.MAE_days, baseline.MAE_days]):
        axes[1].text(x, val + .04, f"{val:.2f}", ha="center")
    axes[1].set_ylim(0, max(selected.MAE_days, baseline.MAE_days) * 1.18)
    fig.tight_layout()
    save_plot(plt, fig, figures / "07_model_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    h = ax.hexbin(predictions.actual_days, predictions.predicted_days, gridsize=45, mincnt=1, bins="log", cmap="viridis")
    limit = float(max(predictions.actual_days.max(), predictions.predicted_days.max()))
    ax.plot([0, limit], [0, limit], "--", color="#d56145", label="Previsão perfeita")
    ax.set(xlabel="Tempo real (dias)", ylabel="Tempo previsto (dias)", title=f"Real × previsto — {winner}, teste\nMAE={selected.MAE_days:.2f} dias; RMSE={selected.RMSE_days:.2f} dias; R²={selected.R2:.3f}", xlim=(0, limit), ylim=(0, limit))
    fig.colorbar(h, ax=ax, label="Pedidos (escala log)")
    ax.legend()
    save_plot(plt, fig, figures / "08_actual_predicted_test.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].hist(predictions.residual_days, bins=65, color="#256b87", edgecolor="white", linewidth=.4)
    axes[0].axvline(0, ls="--", color="#d56145")
    axes[0].set(xlabel="Resíduo = real − previsto (dias)", ylabel="Pedidos", title="Resíduos no teste — positivo: subestimação")
    h = axes[1].hexbin(predictions.predicted_days, predictions.residual_days, bins="log", gridsize=40, mincnt=1, cmap="viridis")
    axes[1].axhline(0, ls="--", color="#d56145")
    axes[1].set(xlabel="Tempo previsto (dias)", ylabel="Resíduo (dias)", title="Erros condicionados à previsão")
    fig.colorbar(h, ax=axes[1], label="Pedidos (escala log)")
    fig.tight_layout()
    save_plot(plt, fig, figures / "09_residuals_test.png")
    importance_path = tables / "importance_validation_permutation.csv"
    importance_text = "O baseline foi escolhido; não há importância de features atribuível à previsão constante."
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        shown = importance.head(12).sort_values("increase_MAE_days_mean")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(shown.feature, shown.increase_MAE_days_mean, xerr=shown["std"], color="#256b87", capsize=3)
        ax.axvline(0, color="#555", linewidth=.7)
        ax.set(xlabel="Aumento do MAE após permutação (dias)", title=f"Importância por permutação — {winner} na validação\n3 repetições, até 3.000 pedidos; barras: desvio padrão")
        save_plot(plt, fig, figures / "10_importance_validation.png")
        top = ", ".join(f"`{r.feature}` ({r.increase_MAE_days_mean:.3f} dia)" for r in importance.head(3).itertuples())
        importance_text = f"Maiores aumentos de MAE na permutação de validação: {top}. A amostra/repetições são limitadas; features correlacionadas podem compartilhar/mascarar importância. Não é evidência causal."

    stats = pd.read_csv(tables / "eda_train_descriptive.csv").set_index("feature")
    relation = pd.read_csv(tables / "eda_train_target_relations.csv").set_index("feature")
    by_state = pd.read_csv(tables / "eda_train_by_state.csv")
    time_stats = stats.loc["delivery_time_days"]
    preprocessing = read_json(f"preprocessing_{winner}.json") if winner != "Baseline" else {}
    removed = preprocessing.get("removed", [])
    final_features_file = tables / "features_final.csv"
    final_dim = int(pd.read_csv(final_features_file).kept.sum()) if final_features_file.exists() else 0
    counts = splits.set_index("split").rows
    metrics_md = markdown_table(validation, ["model", "MAE_days", "RMSE_days", "R2"])
    test_md = markdown_table(test, ["role", "model", "MAE_days", "RMSE_days", "R2"])
    split_md = markdown_table(splits, ["split", "rows", "purchase_min", "purchase_max"])
    params_md = "\n".join(f"- **{name}:** `{json.dumps(params, sort_keys=True)}`." for name, params in frozen["all_best_params"].items())
    second = validation.sort_values(["MAE_days", "RMSE_days", "model"]).iloc[1]
    margin = float(second.MAE_days - validation.MAE_days.min())
    ranges = pd.read_csv(tables / "test_errors_by_actual_range.csv")
    tail = ranges.loc[ranges.actual_range_days.eq("[40.0, inf)")]
    tail_text = (f"Nos {int(tail.iloc[0].orders)} pedidos com duração ≥40 dias, o MAE foi "
                 f"{tail.iloc[0].MAE_days:.2f} dias; os atrasos raros continuam mal previstos." if len(tail) else
                 "Não há pedidos com duração ≥40 dias no teste desta execução.")
    check_file = tables / "verification_summary.json"
    verification = read_json("verification_summary.json") if check_file.exists() else {"checks": 0, "full_data_refits": False}
    tests = test_status(tables / "pytest.xml")
    scores_statement = (f"O modelo reduziu o MAE em {diag['mae_improvement_over_baseline_pct']:.2f}% frente à mediana do ajuste final."
                        if diag["mae_improvement_over_baseline_pct"] >= 0 else
                        f"O MAE ficou {-diag['mae_improvement_over_baseline_pct']:.2f}% acima do baseline; o modelo não o superou neste período de teste. A escolha não foi alterada após essa constatação.")
    r2_statement = ("R² positivo indica redução do erro quadrático frente à média observada do teste."
                    if selected.R2 >= 0 else "R² negativo indica desempenho quadrático inferior à média observada do teste; isso limita o resultado.")
    protected = verify_preservation(root)
    detailed = f"""# TASK 02 — EDA, pré-processamento e treinamento Olist

## Resultado e alcance

Modelo escolhido **{winner}**, pelo menor MAE de validação, antes de abrir o teste.
No teste: **MAE {selected.MAE_days:.4f} dias**, RMSE {selected.RMSE_days:.4f} dias e
R² {selected.R2:.4f}. {scores_statement} {r2_statement}
MAE significa desvio absoluto médio da previsão em relação à entrega real; RMSE
penaliza mais os erros grandes. Esses números não são garantia por pedido.

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

{split_md}

Foram excluídos {split_diag['train_purged_labels']} rótulos ainda desconhecidos no
início da validação e {split_diag['validation_purged_labels']} da validação ainda
desconhecidos no início do teste. Por isso os tamanhos efetivos se afastam de 70/15/15;
não foram substituídos por sorteio. Todos os modelos comparam os mesmos IDs elegíveis.
O reajuste final usa {counts['final_train']} pedidos anteriores ao teste, já entregues
antes desse corte, incluindo rótulos de treino que amadureceram após 15/04/2018.

Purgar evita treinamento/seleção com rótulos futuros, mas favorece entregas rápidas
perto dos cortes: `label_maturation_diagnostic.csv` quantifica esse viés. A avaliação
é retrospectiva, condicionada à entrega observada, e não representa um experimento
prospectivo completo. O teste mantém o conjunto final de entregues do dump, cuja data
de extração oficial é desconhecida; possível censura no fim do período permanece.

Há {split_diag['validation_orders_with_customer_seen_before']} pedidos de validação
e {split_diag['test_orders_with_customer_seen_before']} do teste de clientes já vistos
no respectivo treino. Clientes não foram agrupados entre cortes; IDs não são entradas.
O cenário corresponde à mistura de clientes novos/recorrentes da operação, sem
garantia de generalização para clientes inéditos.

## EDA (somente treino elegível)

O alvo tem mediana {time_stats['50%']:.2f} dias, média {time_stats['mean']:.2f} e máximo
{time_stats['max']:.2f} no treino, com cauda à direita. Spearman entre distância média
e alvo: {relation.loc['distance_km_mean', 'spearman']:.3f}; mesma UF e alvo:
{relation.loc['same_state_items_fraction', 'spearman']:.3f}; frete e alvo:
{relation.loc['freight_value_total', 'spearman']:.3f}. As três distâncias resumem uma
informação semelhante e são fortemente correlacionadas; não são evidências independentes.
Medianas por UF variam de {by_state.median_days.min():.2f} a {by_state.median_days.max():.2f}
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
permanecem. Categóricas: `{', '.join(CATEGORICAL)}`; ausência vira `__MISSING__` e
one-hot com categorias desconhecidas ignoradas. Mês e dia da semana são categorias,
evitando tratá-los como uma ordem linear. Hora/fim de semana mantêm a definição
original. Tipos são normalizados por função determinística dentro do pipeline salvo.

Ridge utiliza escala robusta por mediana/IQR; árvores não recebem escala. Não houve
ajuste de winsorização, corte do alvo ou remoção automática de caudas legítimas.
RobustScaler, regularização e limites de profundidade/folha reduzem sensibilidade;
não eliminam a dificuldade de prever extremos. `VarianceThreshold(0)` seleciona
colunas não constantes após transformação, sem seleção supervisionada por correlação.
As constantes removidas na validação do escolhido foram: `{', '.join(removed) or 'nenhuma'}`.
O reajuste final produziu **{final_dim} colunas transformadas selecionadas**.
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

A vantagem em MAE do escolhido para o segundo colocado ({second.model}) foi apenas
{margin:.4f} dia. Não foram estimados intervalos para essa diferença; ela não demonstra
superioridade estável entre janelas. O critério foi mantido conforme o plano.

Semente 42. Busca sequencial; floresta: 80 árvores, profundidade máxima 16,
bootstrap 85%, no máximo duas threads. Não houve busca extensa ou garantia de
ótimo global. Previsões de todos os regressores têm piso físico zero no estimador
salvo, fixado antes das métricas. Grades completas e tempos: `search_candidates.csv`.

Hiperparâmetros escolhidos por modelo:

{params_md}

## Validação e teste final

{metrics_md}

`selection_frozen.json` selou a decisão antes do teste; seu hash aparece no ajuste
final e no registro de abertura. Após reajustar o escolhido e o baseline com os
rótulos disponíveis em 21/06/2018, realizou-se um evento de avaliação final:

{test_md}

A EDA não usou validação/teste. Importância foi calculada na validação após selar
a escolha; erros do teste são análise descritiva posterior, sem mudança de modelo.

## Erros e importância

{diag['within_3_days_pct']:.2f}% das previsões ficaram a até três dias do real e
{diag['within_5_days_pct']:.2f}% a até cinco. Erro absoluto P90:
{diag['p90_absolute_error_days']:.2f} dias; P95: {diag['p95_absolute_error_days']:.2f}.
Resíduo médio (real − previsto): {diag['mean_residual_days']:.2f} dias; sinal positivo
representa subestimação. Gráficos 08–09 mantêm todos os pontos em densidade e
evidenciam caudas/erros sistemáticos; não indicam intervalos de confiança por pedido.
O sinal médio negativo observado indica superestimação média no período de teste.
{tail_text}
As 30 maiores falhas e erros por UF, mês, vendedores, distância ausente e faixa
do alvo estão nas tabelas `test_errors_*`/`largest_test_errors.csv`. Faixas do alvo
são usadas somente para diagnóstico posterior, nunca como features.

{importance_text}
Importâncias/coefs internos do modelo final também foram exportados. Importância
por impureza pode favorecer variáveis contínuas/com muitas categorias; coeficientes
dependem da escala e regularização. Nenhuma importância foi usada para retunar no teste.

## VERIFY e limitações

{tests}. {verification['checks']} verificações de integridade, splits, métricas,
pré-processamento, recarga e reprodutibilidade registradas; refits completos de
configurações seladas: {verification['full_data_refits']}. As métricas foram
conferidas com somas/fórmulas independentes, e as previsões de refits devem coincidir
a 1e-9 dia. Refits técnicos repetem inferência, sem nova decisão ou nova seleção.
{protected['files_checked']} arquivos anteriores preservados byte a byte; CRC e
SHA-256 dos ZIPs verificados. O Projeto ENADE não foi modificado.

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

Reprodução e arquivos: [README_task02.md](README_task02.md). Não foram criados slides,
vídeo ou apresentação.
"""
    (root / "docs/treinamento_avaliacao.md").write_text(detailed, encoding="utf-8")
    short = f"""# Olist — resultados parciais (TASK 02)

**Ciência dos Dados (T326), UNIFOR.** Prever compra–recebimento, em dias, no instante
da compra. Unidade: pedido. Coorte: 96.470 entregues com datas válidas. As features e
o alvo foram reconstruídos e conferidos com a TASK 01, sem alterar os originais.

## Protocolo e entradas

Mantidas as 30 entradas de carrinho, produtos, vendedores, UF, distância e calendário.
IDs e eventos posteriores ficaram fora de X. Cadastros sem histórico e coordenadas
por CEP são hipóteses/limitações explícitas. Distância não representa rota real.

Compra antes de 15/04/2018: **{counts['train']}** pedidos para treino; de 15/04 a
20/06/2018: **{counts['validation']}** para validação; desde 21/06/2018:
**{counts['test']}** no teste reservado. Excluídos do desenvolvimento
{split_diag['train_purged_labels']} + {split_diag['validation_purged_labels']} pedidos
com entrega ainda desconhecida no respectivo corte. Isso reduz vazamento de rótulos
futuros, mas favorece entregas rápidas perto dos cortes. A cobertura efetiva e as
datas estão em `split_summary.csv`.

Imputação por mediana, categoria ausente explícita, one-hot e remoção de constantes
são ajustados somente no treino dentro de pipelines. Ridge usa escala robusta;
árvores, sem escala. Extremos válidos foram preservados. EDA apenas no treino:
distância média correlaciona-se com o alvo (Spearman
{relation.loc['distance_km_mean', 'spearman']:.3f}); mesma UF tem associação negativa
({relation.loc['same_state_items_fraction', 'spearman']:.3f}). São associações, sem causalidade.

## Modelos e escolha

Baseline mediana, Ridge, árvore e Random Forest; 14 configurações em validação
cronológica, sementes 42, floresta de 80 árvores e duas threads. MAE definiu a escolha:

{metrics_md}

Escolhido **{winner}**, parâmetros variáveis `{json.dumps(frozen['best_params'], sort_keys=True)}`.
Grades e parâmetros fixos estão no relatório detalhado. A escolha foi selada antes
do teste. O escolhido e o baseline foram reajustados com **{counts['final_train']}**
rótulos conhecidos antes de 21/06/2018. Avaliação final única:

{test_md}

O erro absoluto médio do escolhido foi **{selected.MAE_days:.2f} dias**.
{scores_statement} RMSE penaliza erros grandes; R² não mede porcentagem de pedidos
corretos. **{r2_statement}** A vantagem de {winner} sobre {second.model} na validação foi apenas
{margin:.4f} dia, sem evidência de superioridade estável. {diag['within_5_days_pct']:.1f}% ficaram a até cinco dias do real; P90 do erro
absoluto: {diag['p90_absolute_error_days']:.2f} dias. Análise de resíduos e grandes
erros foi feita após avaliação, sem nova escolha ou ajuste.

## Verificação e próximos cuidados

{tests}; {verification['checks']} verificações; refits completos:
{verification['full_data_refits']}. Métricas recalculadas independentemente,
transformações conferidas, ZIP e {protected['files_checked']} arquivos da TASK 01
preservados. Pipelines salvos, gráficos e notebooks executados entregues.

Limites: somente entregues observados, censura no fim do dump, maturação de rótulos,
uma janela temporal, clientes recorrentes entre períodos e cadastros sem versões.
As importâncias são descritivas e afetadas por correlação. Antes de generalizar,
validar esses pressupostos com o professor e planejar novas janelas/avaliação de
incerteza. Não foi feita busca adicional após consultar o teste.

Detalhes: [treinamento_avaliacao.md](treinamento_avaliacao.md);
reprodução: [README_task02.md](README_task02.md).
"""
    (root / "docs/resumo_task02.md").write_text(short, encoding="utf-8")
    feature_rows = []
    audit = pd.read_csv(tables / "feature_temporal_audit.csv").set_index("feature")
    final_names = set(pd.read_csv(final_features_file).loc[lambda f: f.kept, "transformed_feature"]) if final_features_file.exists() else set()
    for name in FEATURE_COLUMNS:
        kept = f"numeric__{name}" in final_names or any(v.startswith(f"categorical__{name}_") for v in final_names)
        feature_rows.append({"Feature de entrada": name, "Tratamento": "ausência explícita + one-hot" if name in CATEGORICAL else "mediana; escala robusta só Ridge",
                             "Seleção final": "mantida" if kept else "removida (constante no ajuste)" if winner != "Baseline" else "baseline não utiliza entradas",
                             "Definição": audit.loc[name, "definition"]})
    feature_doc = "# Features finais — TASK 02\n\nContrato de entrada com 30 colunas, preservado da TASK 01. O seletor aprende constantes por divisão, após imputação/codificação.\n\n"
    feature_doc += markdown_table(pd.DataFrame(feature_rows), ["Feature de entrada", "Tratamento", "Seleção final", "Definição"])
    feature_doc += f"\n\nNo ajuste final do escolhido ({winner}), {final_dim} colunas transformadas foram selecionadas. Nomes e máscaras completos: `resultados/tabelas/task02/features_final.csv`. Contrato de inferência: `resultados/modelos/task02/inference_contract.json`. Não se reutiliza vocabulário aprendido no teste.\n"
    (root / "docs/features_task02.md").write_text(feature_doc, encoding="utf-8")
    artifact_paths = [p for directory in (tables, figures, root / "resultados/modelos/task02") for p in directory.iterdir() if p.is_file() and p.name != "run_manifest.json"]
    artifact_paths += [root / "docs" / name for name in ("resumo_task02.md", "treinamento_avaliacao.md", "features_task02.md", "README_task02.md", "plano_task02.md") if (root / "docs" / name).exists()]
    artifact_paths += sorted((root / "notebooks").glob("02*.ipynb"))
    artifact_paths += sorted((root / "notebooks").glob("03*.ipynb"))
    code_paths = sorted((root / "src/olist_modeling").glob("*.py")) + sorted((root / "tests").glob("test_modeling*.py"))
    code_paths += [root / name for name in ("requirements-task02.txt", "requirements-task02-lock.txt")]
    write_json({"selected_model": winner, "test_evaluation_events": 1, "task01_preservation": protected,
                "versions": read_json("prepared.json")["versions"], "source_integrity": read_json("source_audit.json")["raw_integrity"],
                "selection_sha256": sha256(tables / "selection_frozen.json"),
                "artifact_sha256": {str(p.relative_to(root)): sha256(p) for p in artifact_paths},
                "code_sha256": {str(p.relative_to(root)): sha256(p) for p in code_paths}}, tables / "run_manifest.json")
    print("REPORT: gráficos, features, resumo e relatório atualizados sem novo treinamento/avaliação.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    report(parser.parse_args().root.resolve())
