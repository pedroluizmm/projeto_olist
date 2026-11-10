# Executar e consultar a TASK 02

> **Guia histórico, contextualizado na TASK 04.** Os experimentos já estão
> concluídos. Os comandos abaixo descrevem sua reprodução, não instruções para
> reexecutar a revisão documental atual. A TASK 04 proíbe retreino, nova consulta
> ao teste e alterações dos registros. Consulte o [README atual](../README.md)
> e a [verificação documental](task04/verificacao.md).

As saídas da TASK 01 e seus arquivos de código, documentação, requisitos e testes
foram preservados durante a TASK 02. A TASK 04 revisou apenas documentos
explicitamente identificados; os snapshots e manifestos antigos não foram
alterados para ocultar essas revisões. A TASK 02 acrescentou seu próprio pacote, dependências e pastas.
Não existe dependência do Projeto ENADE.

**Correção E0 posterior:** a importância por permutação foi recalculada com
referência e permutações nos mesmos 12.675 pedidos da validação, sem retreinamento
ou consulta ao teste. Os resultados históricos, modelos e métricas foram
preservados. Consulte [correcao_e0.md](correcao_e0.md) para o procedimento isolado,
os resultados corrigidos e os testes específicos; não execute o pipeline completo
para reproduzir somente E0.

## Ambiente

CPython 3.13.6/Windows, na mesma `.venv` local. Novas dependências diretas fixadas
em `requirements-task02.txt`; ambiente completo em `requirements-task02-lock.txt`.
Os arquivos `requirements.txt`/`requirements-lock.txt` da TASK 01 não foram alterados.

```powershell
Set-Location C:\trabCDados\projeto-olist
.\.venv\Scripts\python.exe -m pip install -r requirements-task02-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
```

Para novo ambiente: `python -m venv .venv` antes da instalação. Reinstale o kernel
local se necessário, conforme o README original. O código novo em `src/olist_modeling`
fica disponível pela instalação editável do projeto. Não é necessário mudar as
dependências declaradas pelo pacote da TASK 01 para usar o arquivo adicional.

## Protocolo histórico de execução inicial (não executar na TASK 04)

```powershell
.\.venv\Scripts\python.exe -m olist_modeling.run --stage prepare
.\.venv\Scripts\python.exe -m olist_modeling.run --stage select
.\.venv\Scripts\python.exe -m olist_modeling.run --stage evaluate
.\.venv\Scripts\python.exe -m olist_modeling.verify
.\.venv\Scripts\python.exe -m pytest -q --junitxml=resultados/tabelas/task02/pytest.xml
.\.venv\Scripts\python.exe -m olist_modeling.run --stage report
```

Em uma pasta de saída nova, `python -m olist_modeling.run --stage all` executa
preparação, seleção, avaliação e relatório. Em seguida execute verificação, testes
e atualize o relatório. Os notebooks são gerados/executados pelo comando abaixo.
O arquivo de preservação `resultados/tabelas/task02/task01_preservation.json` contém
os hashes dos 47 arquivos existentes ao iniciar a tarefa; deve acompanhar esta entrega.

**A entrega atual já foi executada.** A seleção selada e o registro de abertura do
teste bloqueiam nova seleção/avaliação nos mesmos caminhos. Não apague esses registros
para retunar no teste. Use os comandos de consulta/verificação abaixo. Uma reprodução
integral independente deve ser feita em cópia do projeto, preservando os resultados
originais e definindo um novo registro antes de consultar seus resultados.

## Verificações históricas da TASK 02 (não executar na TASK 04)

```powershell
.\.venv\Scripts\python.exe -m olist_modeling.verify
.\.venv\Scripts\python.exe -m pytest -q --junitxml=resultados/tabelas/task02/pytest.xml
.\.venv\Scripts\python.exe -m olist_modeling.notebooks
.\.venv\Scripts\python.exe -m olist_modeling.run --stage report
```

`verify` recarrega modelos, confere fórmulas das métricas, transformadores, splits,
originais e refaz o ajuste das configurações escolhidas nos mesmos dados completos.
Compara previsões a 1e-9 dia; não pesquisa parâmetros, troca o vencedor ou decide a
partir do teste. `--no-refit` omite os refits para uma checagem rápida. Uma verificação
rápida posterior substitui o resumo; para restaurar a evidência completa, execute
novamente sem essa opção. Consultar previsões já seladas e recomputar fórmulas para
verificação não equivale a uma nova rodada de seleção de modelo.

**Atenção ao escopo atual:** mesmo `--no-refit` consulta previsões/dados de teste.
Os comandos de notebooks/testes também podem ler esses artefatos. `report` regrava
documentos, gráficos e um manifesto com os templates anteriores à correção.
Nenhum desses comandos foi executado na TASK 04. Para verificar somente as
alterações documentais, use `python -B docs/task04/verificar_documentacao.py`.
O verificador original de preservação pode sinalizar os documentos agora
autorizadamente alterados: não se deve editar snapshots históricos para fazê-lo
passar. Uma reprodução futura deve distinguir os snapshots da TASK 02 da revisão
documental atual e manter ambas as versões.

Os testes de integração requerem saídas reais; não fabricam substitutos quando
faltam arquivos. Fixtures artificiais são exclusivas dos testes unitários. O XML
novo é gravado na subpasta `task02/`, preservando o XML da TASK 01.

## Pastas e artefatos

```text
src/olist_modeling/
  protocol.py      # contrato, auditoria e separação cronológica
  estimators.py    # Pipeline/ColumnTransformer e grades
  eda.py           # EDA apenas do treino
  run.py           # preparar, selecionar e avaliar uma vez
  verify.py        # métricas independentes e refits técnicos
  report.py        # gráficos e documentos, sem reavaliar
  notebooks.py     # criar/executar notebooks de consulta
resultados/tabelas/task02/
  source_audit.json, feature_temporal_audit.csv
  split_assignments.csv, split_summary.csv, split_diagnostics.json
  label_maturation_diagnostic.csv, eda_train_*.csv
  protocol.json, prepared.json, task01_preservation.json
  search_candidates.csv, validation_metrics.csv
  preprocessing_*.json, features_*.csv
  selection_frozen.json, validation_predictions.parquet
  final_fit.json, test_evaluation_started.json
  test_metrics.csv, test_predictions.parquet, test_errors_*.csv
  largest_test_errors.csv, test_diagnostics.json
  importance_validation_permutation.csv, importance_final_transformed.csv
  independent_metrics.csv, reproducibility_refits.csv
  verification_checks.csv, verification_summary.json
  pytest.xml, pytest.txt, notebook_verification.json
  run_manifest.json
resultados/graficos/task02/       # 6 figuras de treino + avaliação/importância
resultados/modelos/task02/        # pipelines de validação e ajuste final
notebooks/02_eda_olist.ipynb
notebooks/03_treinamento_olist.ipynb
docs/plano_task02.md
docs/features_task02.md
docs/treinamento_avaliacao.md
docs/resumo_task02.md
```

Tabelas sem observações suficientes mantêm R² ausente; não se preenche com uma
métrica inventada. CSVs usam UTF-8 e ponto decimal. MAE/RMSE estão em dias.
Os arquivos `test_*` e auditorias com datas posteriores não são fontes de features.

## Exemplo histórico de inferência (não executar na TASK 04)

Carregue apenas os arquivos Joblib produzidos neste projeto e preservados pelo
manifesto; Joblib usa serialização Python. O pipeline final inclui conversão de
tipos, imputação, codificação, seleção, regressor e piso em zero.

```python
import json
from pathlib import Path
import joblib
import pandas as pd

root = Path(r"C:\trabCDados\projeto-olist")
folder = root / "resultados/modelos/task02"
contract = json.loads((folder / "inference_contract.json").read_text(encoding="utf-8"))
model = joblib.load(folder / contract["model_file"])
data = pd.read_parquet(root / "dados/tratados/olist_orders_features.parquet")
predicted_days = model.predict(data[contract["feature_columns"]])
```

Essa chamada executa inferência, não deve ser usada para buscar melhorias no teste.
Não há garantia de calibração operacional; as limitações estão no relatório técnico.
O relatório inclui links para a documentação oficial do Scikit-learn usada no desenho
dos pipelines e da busca. Nenhum serviço externo é necessário após a instalação.

## Correções interpretativas e rastreabilidade

Os resultados e sua leitura corrigida estão no [relatório](treinamento_avaliacao.md)
e em [limitações e decisões](limitacoes_decisoes.md). A configuração salva da floresta
usa `max_features=1.0` (float: todas as features), e não `1` (inteiro: uma feature).
Ridge e floresta ficaram muito próximas; a seleção seguiu o MAE pré-definido,
sem vantagem estatisticamente demonstrada. O R² negativo de teste não invalida o
ganho de MAE, mas revela limites temporais de calibração/generalização.

`selection_frozen.json` foi regravado depois da selagem inicial para acrescentar
atestação de código. Não há snapshot exato do `run.py` anterior; hashes atuais e
refits disponíveis não eliminam essa lacuna. A auditoria relatada pelo usuário
não encontrou sinal de consulta antecipada ao teste. Esta revisão não altera
retroativamente o registro, os logs ou os manifestos.
