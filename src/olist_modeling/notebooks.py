"""Cria e executa notebooks de consulta; não treina nem seleciona modelos."""

import argparse
import json
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


SETUP = '''from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/olist_modeling").is_dir())
assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "Selecione Python (Olist .venv)"
TABLES = ROOT / "resultados/tabelas/task02"
FIGURES = ROOT / "resultados/graficos/task02"
read_json = lambda name: json.loads((TABLES / name).read_text(encoding="utf-8"))
print("Ambiente:", sys.version.split()[0], "| kernel Olist local")'''


def book(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (Olist .venv)", "language": "python", "name": "olist"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.13.6"}
    nb.cells = [nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source) for kind, source in cells]
    return nb


def generate(root):
    eda = book([
        ("md", "# Olist — EDA e auditoria temporal (TASK 02)\n\nCiência dos Dados (T326), UNIFOR. Os scripts reproduzíveis estão em `src/olist_modeling/`. Este notebook consulta resultados reais já gerados; não ajusta transformadores/modelos. **EDA restrita ao treino com rótulos disponíveis antes de 15/04/2018.**"),
        ("code", SETUP),
        ("md", "## Coorte, disponibilidade e cortes\n\nUnidade: pedido; alvo: dias fracionários compra–recebimento. Coorte somente de entregues. Eventos posteriores e IDs não são features. A vigência histórica dos cadastros não pode ser provada pelo dump. Rótulos ainda desconhecidos nos cortes foram purgados; isso também introduz viés de maturação."),
        ("code", '''scope = read_json("eda_scope.json")
audit = read_json("source_audit.json")
assert audit["passed"] and not scope["test_used"]
print(scope)
display(pd.read_csv(TABLES / "split_summary.csv"))
display(pd.read_csv(TABLES / "label_maturation_diagnostic.csv"))
display(pd.read_csv(TABLES / "feature_temporal_audit.csv"))'''),
        ("md", "## Distribuições, ausências e extremos\n\nHistogramas mantêm os extremos; valor e peso usam eixo logarítmico. IQR é diagnóstico e não remove pedidos. O alcance das previsões permanece condicionado à entrega observada."),
        ("code", '''display(pd.read_csv(TABLES / "eda_train_descriptive.csv"))
missing = pd.read_csv(TABLES / "eda_train_missing.csv")
display(missing.loc[missing.missing.gt(0)])
display(pd.read_csv(TABLES / "eda_train_outliers.csv"))
display(pd.read_csv(TABLES / "eda_train_categorical.csv"))
display(Image(filename=str(FIGURES / "01_distributions_train.png"), width=1050))'''),
        ("md", "## Correlações e relações com o alvo\n\nCorrelação de Spearman usa apenas pares conhecidos no treino e omite constantes no gráfico. Distâncias mínima/média/máxima compartilham informação. Associações e importâncias não estabelecem causalidade."),
        ("code", '''relations = pd.read_csv(TABLES / "eda_train_target_relations.csv")
display(relations)
display(Image(filename=str(FIGURES / "02_correlations_train.png"), width=1100))
display(Image(filename=str(FIGURES / "03_feature_target_train.png"), width=1100))'''),
        ("md", "## Geografia e cobertura temporal\n\nUFs têm amostras diferentes; a dispersão por CEP mostra somente prefixos com pelo menos 20 pedidos. Pontos são aproximações geográficas, sem rotas. Meses sem pedidos aparecem como lacunas. Meses iniciais escassos e o último período afetado pela maturação exigem cautela."),
        ("code", '''display(pd.read_csv(TABLES / "eda_train_by_state.csv"))
display(pd.read_csv(TABLES / "eda_train_by_month.csv"))
for name in ["04_state_train.png", "05_geographic_train.png", "06_monthly_train.png"]:
    display(Image(filename=str(FIGURES / name), width=1000))'''),
        ("md", "## Interpretação\n\nA distância e o frete apresentam associação positiva com tempo, e mesma UF apresenta associação negativa. Valores exatos estão na tabela acima. Caudas, diferenças regionais e mudanças temporais permanecem, sem atribuição causal. A EDA não usou validação ou teste para escolher tratamentos."),
    ])
    training = book([
        ("md", "# Olist — pré-processamento, treinamento e avaliação (TASK 02)\n\nNotebook de consulta executado após as decisões seladas. **Não dispara treinamento nem busca.** Para reproduzir o protocolo, consulte `docs/README_task02.md`."),
        ("code", SETUP),
        ("md", "## Pré-processamento e orçamento\n\nMediana numérica, categoria ausente explícita, one-hot e seleção de constantes dentro de Pipeline/ColumnTransformer. Ridge: escala robusta; árvores: sem escala. Extremos válidos preservados. GridSearchCV em uma divisão cronológica, sem refit automático sobre validação; MAE define escolha, sementes 42."),
        ("code", '''protocol = read_json("protocol.json")
frozen = read_json("selection_frozen.json")
print("Configurações pesquisadas:", protocol["candidate_count"])
print("Escolha congelada:", frozen["selected_model"])
print("Parâmetros por modelo:", frozen["all_best_params"])
display(pd.read_csv(TABLES / "split_summary.csv"))
display(pd.read_csv(TABLES / "search_candidates.csv"))'''),
        ("md", "## Escolha na validação e avaliação final única\n\nA seleção foi selada antes de abrir o teste. O vencedor e a referência foram reajustados somente com rótulos conhecidos antes do teste. Comparação de teste abaixo é consulta dos resultados registrados; não altera a escolha."),
        ("code", '''from olist_features.audit import sha256
opening = read_json("test_evaluation_started.json")
assert frozen["sealed_at_utc"] < opening["started_at_utc"]
assert opening["selection_sha256"] == sha256(TABLES / "selection_frozen.json")
display(pd.read_csv(TABLES / "validation_metrics.csv"))
display(pd.read_csv(TABLES / "test_metrics.csv"))
display(Image(filename=str(FIGURES / "07_model_comparison.png"), width=1000))'''),
        ("md", "## Resíduos e maiores falhas\n\nMAE/RMSE em dias; resíduos positivos significam subestimação. Métricas foram conferidas independentemente. Falhas por faixa do alvo são diagnóstico posterior, não features. Esta inspeção não gerou ajuste adicional no teste."),
        ("code", '''print(read_json("test_diagnostics.json"))
display(pd.read_csv(TABLES / "largest_test_errors.csv").head(10))
display(pd.read_csv(TABLES / "test_errors_by_actual_range.csv"))
display(pd.read_csv(TABLES / "test_errors_by_month.csv"))
display(Image(filename=str(FIGURES / "08_actual_predicted_test.png"), width=850))
display(Image(filename=str(FIGURES / "09_residuals_test.png"), width=1050))'''),
        ("md", "## Importância e verificação\n\nPermutação na validação, após selar a escolha, mede sensibilidade preditiva. Correlação pode mascarar a importância. Não há interpretação causal. Refits técnicos mantêm os parâmetros e verificam previsões, sem retunar a partir do teste."),
        ("code", '''importance = TABLES / "importance_validation_permutation.csv"
if importance.exists():
    display(pd.read_csv(importance))
    display(Image(filename=str(FIGURES / "10_importance_validation.png"), width=1000))
print(read_json("verification_summary.json"))
display(pd.read_csv(TABLES / "reproducibility_refits.csv"))
checks = pd.read_csv(TABLES / "verification_checks.csv")
assert checks.passed.all()
print("Verificações aprovadas:", len(checks))'''),
        ("md", "## Modelo salvo e limites\n\n`resultados/modelos/task02/final_model.joblib` contém o pipeline, e `inference_contract.json` define as 30 entradas. A amostra é condicionada à entrega; maturação, censura, snapshot sem histórico, CEPs aproximados e uma única janela limitam a generalização. Não há intervalos calibrados ou garantia operacional. Próximas decisões estão em `docs/treinamento_avaliacao.md`."),
    ])
    return {"02_eda_olist.ipynb": eda, "03_treinamento_olist.ipynb": training}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().root.resolve()
    for variable, relative in [("JUPYTER_RUNTIME_DIR", ".venv/jupyter/runtime"), ("JUPYTER_CONFIG_DIR", ".venv/jupyter/config"),
                                ("IPYTHONDIR", ".venv/ipython")]:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    summary = []
    for name, nb in generate(root).items():
        nbf.validate(nb)
        NotebookClient(nb, timeout=120, kernel_name="olist", resources={"metadata": {"path": str(root)}}).execute()
        nbf.validate(nb)
        errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
        if errors:
            raise AssertionError(errors)
        nbf.write(nb, root / "notebooks" / name)
        summary.append({"notebook": name, "code_cells_executed": sum(c.cell_type == "code" for c in nb.cells),
                        "errors": 0, "kernel": "olist", "passed": True})
        print(f"NOTEBOOK: {name} executado, sem erros.", flush=True)
    (root / "resultados/tabelas/task02/notebook_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
