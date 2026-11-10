# TASK 01 — plano de execução

## READ (inspeção realizada antes da implementação)

- Fonte encontrada: `dados/brutos/brazilian-ecommerce.zip`, nove CSVs.
- SHA-256 inicial: `967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784`.
- CRC de todos os membros validado; nenhuma extração ou alteração dos originais.
- 99.441 pedidos; 96.478 com status `delivered`; oito destes sem data de entrega.
- Chaves de pedidos, clientes, produtos e vendedores únicas; relações principais sem órfãos.
- Itens e pagamentos são relações 1:N; geolocalização não tem chave única por CEP.
- Geolocalização: 1.000.163 linhas, 261.831 duplicatas integrais e 19.015 prefixos.
- Ausências em medidas/categorias de produtos e em eventos de pedidos exigem tratamento explícito.

## PLAN

1. Criar estrutura, ambiente Python isolado e kernel instalado dentro da própria `.venv`.
2. Ler CSVs diretamente do ZIP, preservando CEPs como texto de cinco caracteres.
3. Registrar inventário, tipos físicos/lógicos, ausências, cardinalidades, chaves e relações.
4. Separar construção de features (compra/carrinho/cadastros) e construção do alvo.
5. Agregar itens por pedido antes das junções. Agregar pagamentos apenas em relatório de auditoria.
6. Deduplicar geolocalização em memória, invalidar coordenadas impossíveis/fora de uma caixa ampla do Brasil e calcular medianas por CEP.
7. Gerar features de valores, quantidade, medidas, categorias, vendedores, UF, distância e calendário.
8. Selecionar entregues com compra/entrega válidas, duração não negativa e pelo menos um item.
9. Preservar ausências; sinalizar cobertura parcial. Detectar outliers sem aparar/remover automaticamente.
10. Verificar CRC e SHA-256 antes/depois, contagens, cálculos independentes, unicidade e ausência de eventos posteriores nas features.
11. Executar testes unitários e de integração, verificar repetibilidade e documentar resultados reais.

## APPLY / VERIFY

Implementação em `src/olist_features/`; evidências em `resultados/tabelas/` e
`docs/engenharia_features.md`. Não haverá EDA completa, imputação aprendida,
normalização aprendida, divisão para modelagem, treinamento ou otimização.
