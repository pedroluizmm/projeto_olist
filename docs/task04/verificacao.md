# Verificação documental — TASK 04

**Resultado: aprovado nas 158 verificações documentais.** Não foram
executados treinamento, busca, inferência, avaliação ou novo acesso tabular ao teste.
Os 53 testes e 77 checks dos experimentos são históricos; não foram reexecutados.

## Preservação antes/depois

Snapshot anterior: `preservation_before.json`; relatório completo: `preservation_after.json`.
Dos 136 arquivos preexistentes inventariados, 129 permaneceram
idênticos; somente os 7 documentos autorizados foram alterados. Nenhum
arquivo preexistente foi excluído. Novos arquivos ficaram restritos a documentação
e sua verificação em `docs/`. `.venv`, caches e metadados egg-info não integram esse
inventário. Os planos metodológicos históricos não foram alterados.

| Grupo protegido | Arquivos | Resultado |
|---|---:|---|
| raw_data | 1 | idênticos por SHA-256 |
| treated_data | 5 | idênticos por SHA-256 |
| models | 7 | idênticos por SHA-256 |
| historical_tables_and_logs | 76 | idênticos por SHA-256 |
| figures | 11 | idênticos por SHA-256 |
| notebooks | 3 | idênticos por SHA-256 |
| source_code | 13 | idênticos por SHA-256 |
| tests | 5 | idênticos por SHA-256 |

As contagens por grupo são de arquivos, incluindo `.gitkeep` e contratos JSON,
quando existentes. Permanecem 10 figuras PNG e seis arquivos Joblib de modelos;
o grupo de modelos também inclui o contrato de inferência.

ZIP original: CRC válido e SHA-256 preservado:
`967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784`.
`selection_frozen.json`: SHA-256 antes e depois iguais:
`bf47e61d414a88a919e75762a38446142784822bfe5c5d4f3de41a9eb2abc942`.
Logs, manifestos, métricas, previsões e divisões temporais não foram regravados.

## Correções e consistência

Conferidos os números fornecidos na solicitação e as identidades de reconciliação,
incluindo os 11 pedidos iniciais ainda sem rótulo no corte final. Valores de teste
e médias foram documentados a partir da auditoria/relatos anteriores, não de nova
leitura ou recálculo. A configuração salva confirma `max_features=1.0` float; a
leitura dos parâmetros do Joblib foi feita sem `fit`/`predict` durante a inspeção.
Metadados existentes confirmam 143 colunas no treino inicial e 144 no ajuste final.
Foram conferidos 27 links locais dos documentos revisados.

Documentos preexistentes modificados:

- `README.md`
- `docs/README_task02.md`
- `docs/engenharia_features.md`
- `docs/features_task02.md`
- `docs/metodologia_detalhada.md`
- `docs/resumo_task02.md`
- `docs/treinamento_avaliacao.md`

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
