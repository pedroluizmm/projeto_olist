"""VERIFY documental: stdlib, sem fit/predict, sem leitura tabular de teste.

Compara bytes por SHA-256, confere metadados de configuração/divisão e a redação.
Só grava novos relatórios em docs/task04; nunca regrava registros das TASKS 01/02.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/task04"
EDITABLE = {
    "README.md", "docs/treinamento_avaliacao.md", "docs/resumo_task02.md",
    "docs/README_task02.md", "docs/engenharia_features.md",
    "docs/metodologia_detalhada.md", "docs/features_task02.md",
}
NEW_FILES = {
    "docs/limitacoes_decisoes.md", "docs/rubrica_academica.md",
    "docs/task04/preservation_before.json", "docs/task04/preservation_after.json",
    "docs/task04/verificacao.md", "docs/task04/verificar_documentacao.py",
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def files_in_scope():
    paths = [p for folder in ("dados", "docs", "notebooks", "resultados", "src", "tests")
             for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and not any(part.endswith(".egg-info") for part in p.parts)]
    paths.extend(p for p in ROOT.iterdir() if p.is_file())
    return {p.relative_to(ROOT).as_posix(): p for p in paths}


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def compact(text):
    return re.sub(r"\s+", " ", text).replace("**", "").replace("`", "")


def main():
    before = read_json("docs/task04/preservation_before.json")["files"]
    current = files_in_scope()
    checks = []

    def check(name, condition, evidence):
        if not condition:
            raise AssertionError(f"{name}: {evidence}")
        checks.append({"check": name, "passed": True, "evidence": evidence})

    removed = sorted(set(before) - set(current))
    changed = sorted(name for name in before if name in current and sha(current[name]) != before[name]["sha256"])
    added = sorted(set(current) - set(before))
    check("no_preexisting_files_deleted", not removed, removed)
    check("only_authorized_document_changes", set(changed) == EDITABLE, changed)
    check("only_documentary_files_added", set(added).issubset(NEW_FILES), added)
    unchanged = sorted(set(before) - set(changed))
    groups = {}
    for label, prefix in [("raw_data", "dados/brutos/"), ("treated_data", "dados/tratados/"),
                          ("models", "resultados/modelos/"), ("historical_tables_and_logs", "resultados/tabelas/"),
                          ("figures", "resultados/graficos/"), ("notebooks", "notebooks/"),
                          ("source_code", "src/"), ("tests", "tests/")]:
        names = [name for name in before if name.startswith(prefix)]
        check(f"{label}_byte_identical", bool(names) and all(name in unchanged for name in names), len(names))
        groups[label] = len(names)

    zip_results = []
    for name in before:
        if name.startswith("dados/brutos/") and name.lower().endswith(".zip"):
            with zipfile.ZipFile(ROOT / name) as archive:
                check("original_zip_crc", archive.testzip() is None, name)
            zip_results.append({"file": name, "crc_ok": True, "sha256": sha(ROOT / name)})

    # Ler somente resumos de contagem/configuração, nunca dados/previsões de teste.
    folder = ROOT / "resultados/tabelas/task02"
    with (folder / "split_summary.csv").open(encoding="utf-8", newline="") as f:
        counts = {row["split"]: int(row["rows"]) for row in csv.DictReader(f)}
    check("recorded_split_counts", counts == {"train_candidates": 67379, "train": 64401,
          "validation_candidates": 14606, "validation": 12675, "test": 14485, "final_train": 80043}, counts)
    recovered = counts["final_train"] - counts["train"] - counts["validation"]
    eleven = counts["train_candidates"] - (counts["final_train"] - counts["validation"])
    check("recovered_and_eleven_initial_orders", recovered == 2967 and eleven == 11, {"recovered": recovered, "still_unavailable": eleven})
    identities = ["96.470 = 91.561 + 4.909", "91.561 = 64.401 + 12.675 + 14.485",
                  "4.909 = 2.978 + 1.931", "77.076 = 64.401 + 12.675",
                  "80.043 = 77.076 + 2.967", "2.978 = 2.967 + 11"]
    for equation in identities:
        parts = equation.replace(".", "").split("=")
        check("sample_identity_" + parts[0].strip(), int(parts[0]) == sum(map(int, parts[1].split("+"))), equation)

    frozen = read_json("resultados/tabelas/task02/selection_frozen.json")
    maximum = frozen["all_best_params"]["RandomForest"]["model__estimator__max_features"]
    check("saved_configuration_float_1_0", type(maximum) is float and maximum == 1.0, {"value": maximum, "type": type(maximum).__name__})
    selected_hash = sha(folder / "selection_frozen.json")
    fit = read_json("resultados/tabelas/task02/final_fit.json")
    opening = read_json("resultados/tabelas/task02/test_evaluation_started.json")
    check("selection_records_preserved_and_linked", fit["selection_sha256"] == opening["selection_sha256"] == selected_hash,
          {"sha256": selected_hash, "scope": "hash da versão posterior à regravação; não prova imutabilidade inicial"})
    check("recorded_event_still_one", opening["evaluation_event"] == 1, "Registro histórico, não reexecutado")
    for name, expected in (("features_Ridge.csv", 143), ("features_final.csv", 144)):
        with (folder / name).open(encoding="utf-8", newline="") as f:
            count = sum(row["kept"].lower() == "true" for row in csv.DictReader(f))
        check("transformed_features_" + name, count == expected, count)

    central = ["README.md", "docs/treinamento_avaliacao.md", "docs/resumo_task02.md", "docs/limitacoes_decisoes.md"]
    for relative in central:
        text = compact((ROOT / relative).read_text(encoding="utf-8"))
        for equation in identities:
            check("document_identity:" + relative + ":" + equation.split("=")[0].strip(), equation in text, equation)
        numbers = text.replace(",", ".").replace("−", "-")
        for token in ("0.3221", "-0.0246", "4.1081", "14.87", "13.27", "8.37", "10.51", "2.15"):
            check("document_number:" + relative + ":" + token, token in numbers, "Número do relato/registro existente; não recomputado no teste")
        for token in ("variância", "max_features=1.0", "max_features=1", "bagging", "estatisticamente demonstrada",
                      "selection_frozen.json", "regravado", "snapshot", "run.py", "antecipada"):
            check("document_interpretation:" + relative + ":" + token, token.casefold() in text.casefold(), token)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("readme_current_project_status", "Nenhum modelo é treinado" not in readme and "treinados" in readme, "TASK 02 concluída; TASK 04 documental")
    rubric = compact((ROOT / "docs/rubrica_academica.md").read_text(encoding="utf-8"))
    check("rubric_weights", sum((10, 30, 10, 30, 20)) == 100 and all(f"{n}%" in rubric for n in (10, 30, 20)), "Pesos informados: 10/30/10/30/20")
    for token in ("pelo menos quatro procedimentos", "pelo menos quatro análises", "três modelos", "10 a 20 minutos", "todos os integrantes", "YouTube", "AVA", "Não foi localizado documento oficial"):
        check("academic_requirement:" + token, token in rubric, "Checklist baseado no relato do usuário, sem atribuição de nota")
    # Conferir links locais dos documentos revisados/criados, sem acessar URLs externas.
    documents = sorted(EDITABLE | {"docs/limitacoes_decisoes.md", "docs/rubrica_academica.md"})
    link_count = 0
    for relative in documents:
        source = ROOT / relative
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            destination = (source.parent / target.split("#", 1)[0]).resolve()
            check("local_link:" + relative + ":" + target, destination.exists() or destination == OUT / "verificacao.md", target)
            link_count += 1

    result = {"passed": True, "task": "04", "scope": "Verificação documental; sem treinamento, inferência ou consulta tabular ao teste",
              "preexisting_files": len(before), "changed_documents": changed, "unchanged_preexisting_files": len(unchanged),
              "protected_groups": groups, "raw_zip_integrity": zip_results, "checks_passed": len(checks), "local_links_checked": link_count,
              "selection_frozen_sha256_before": before["resultados/tabelas/task02/selection_frozen.json"]["sha256"],
              "selection_frozen_sha256_after": selected_hash,
              "tests_rerun": False, "historical_test_results": "53 testes e 77 verificações registrados na TASK 02",
              "official_rubric_found": False, "rubric_source": "Requisitos informados pelo usuário na TASK 04",
              "additional_means_source": "Auditoria independente relatada pelo usuário; números não recalculados nesta tarefa",
              "checks": checks,
              "sha256_after_for_preexisting_files": {name: sha(ROOT / name) for name in before}}
    (OUT / "preservation_after.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    modified_list = "\n".join(f"- `{name}`" for name in changed)
    group_rows = "\n".join(f"| {name} | {count} | idênticos por SHA-256 |" for name, count in groups.items())
    report = f"""# Verificação documental — TASK 04

**Resultado: aprovado nas {len(checks)} verificações documentais.** Não foram
executados treinamento, busca, inferência, avaliação ou novo acesso tabular ao teste.
Os 53 testes e 77 checks dos experimentos são históricos; não foram reexecutados.

## Preservação antes/depois

Snapshot anterior: `preservation_before.json`; relatório completo: `preservation_after.json`.
Dos {len(before)} arquivos preexistentes inventariados, {len(unchanged)} permaneceram
idênticos; somente os {len(changed)} documentos autorizados foram alterados. Nenhum
arquivo preexistente foi excluído. Novos arquivos ficaram restritos a documentação
e sua verificação em `docs/`. `.venv`, caches e metadados egg-info não integram esse
inventário. Os planos metodológicos históricos não foram alterados.

| Grupo protegido | Arquivos | Resultado |
|---|---:|---|
{group_rows}

As contagens por grupo são de arquivos, incluindo `.gitkeep` e contratos JSON,
quando existentes. Permanecem 10 figuras PNG e seis arquivos Joblib de modelos;
o grupo de modelos também inclui o contrato de inferência.

ZIP original: CRC válido e SHA-256 preservado:
`{zip_results[0]['sha256']}`.
`selection_frozen.json`: SHA-256 antes e depois iguais:
`{selected_hash}`.
Logs, manifestos, métricas, previsões e divisões temporais não foram regravados.

## Correções e consistência

Conferidos os números fornecidos na solicitação e as identidades de reconciliação,
incluindo os 11 pedidos iniciais ainda sem rótulo no corte final. Valores de teste
e médias foram documentados a partir da auditoria/relatos anteriores, não de nova
leitura ou recálculo. A configuração salva confirma `max_features=1.0` float; a
leitura dos parâmetros do Joblib foi feita sem `fit`/`predict` durante a inspeção.
Metadados existentes confirmam 143 colunas no treino inicial e 144 no ajuste final.
Foram conferidos {link_count} links locais dos documentos revisados.

Documentos preexistentes modificados:

{modified_list}

Novos documentos: `docs/limitacoes_decisoes.md`, `docs/rubrica_academica.md` e esta
pasta de verificação (script, snapshot inicial e relatório antes/depois). O código
experimental não foi modificado. Os geradores antigos podem sobrescrever a redação
corrigida e não foram executados; os manifestos continuam snapshots históricos.

## Ressalvas e entrega acadêmica

Persistem a regravação pós-selagem de `selection_frozen.json`, ausência do snapshot
exato de `run.py` anterior, limitações temporais/calibração e seleção de entregues.
A auditoria relatada pelo usuário não encontrou sinal de consulta antecipada ao
teste; a revisão atual não transforma isso em garantia de custódia perfeita.

Rubrica oficial não localizada. O checklist de pesos 10/30/10/30/20 e mínimos de
quatro procedimentos, quatro análises e três modelos com otimização foi conferido
contra os requisitos fornecidos pelo usuário e as evidências existentes. Não se
atribuiu nota. Conteúdo técnico pronto para orientar slides com ressalvas;
confirmar a versão oficial da rubrica. Vídeo de 10 a 20 minutos, slides,
participação de todos e envio do link do YouTube pelo AVA continuam pendentes.
Nenhum desses materiais foi produzido nesta tarefa.

Reexecutar apenas esta verificação documental:
`python -B docs/task04/verificar_documentacao.py`.
"""
    (OUT / "verificacao.md").write_text(report, encoding="utf-8")
    print(f"TASK 04: {len(checks)} verificações; {len(changed)} documentos alterados; {len(unchanged)} arquivos anteriores preservados.")


if __name__ == "__main__":
    main()
