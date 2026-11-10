"""Repete o pipeline e compara hashes de dados e relatórios determinísticos."""

import argparse
import json
from pathlib import Path

from .pipeline import run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().root.resolve()
    path = root / "resultados/tabelas/run_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Execute o pipeline antes da comparação")
    before = json.loads(path.read_text(encoding="utf-8"))
    after = run(root)
    a, b = before["output_sha256"], after["output_sha256"]
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    result = {"passed": not changed, "artifacts_compared": len(a), "changed_artifacts": changed,
              "source_unchanged": before["input_integrity_after"] == after["input_integrity_after"],
              "scope": "hashes de todos os dados e relatórios determinísticos do manifesto; exclui docs, logs, tempos de teste e o próprio manifesto"}
    write_json(result, root / "resultados/tabelas/reproducibility.json")
    if changed or not result["source_unchanged"]:
        raise AssertionError(result)
    print(f"REPRODUCIBILITY: {len(a)} artefatos idênticos; originais intactos.", flush=True)


if __name__ == "__main__":
    main()
