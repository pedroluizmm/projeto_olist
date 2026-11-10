# Features finais — TASK 02

Contrato de entrada com 30 colunas, preservado da TASK 01. O seletor aprende constantes por divisão, após imputação/codificação.

Revisão TASK 04: foram 143 colunas transformadas selecionadas no treino inicial
de 64.401 pedidos e 144 no ajuste final de 80.043 pedidos. O vocabulário categórico
é aprendido em cada treino; não se confunde o tamanho da matriz inicial com a final.
As duas constantes removidas continuam no contrato de entrada de 30 atributos.
Nenhum transformador, modelo ou dado foi reajustado nesta revisão.

| Feature de entrada | Tratamento | Seleção final | Definição |
| --- | --- | --- | --- |
| products_value_total | mediana; escala robusta só Ridge | mantida | Soma dos preços dos itens; BRL. |
| freight_value_total | mediana; escala robusta só Ridge | mantida | Soma do frete dos itens; BRL. |
| order_value_total | mediana; escala robusta só Ridge | mantida | Produtos + frete; não usa pagamentos. |
| items_count | mediana; escala robusta só Ridge | mantida | Número de linhas de itens, não máximo do sequencial. |
| products_count | mediana; escala robusta só Ridge | mantida | Quantidade de produtos distintos no carrinho. |
| sellers_count | mediana; escala robusta só Ridge | mantida | Quantidade de vendedores distintos. |
| seller_states_count | mediana; escala robusta só Ridge | mantida | Quantidade de UFs dos vendedores. |
| same_state_items_fraction | mediana; escala robusta só Ridge | mantida | Fração de itens com vendedor na UF do comprador. |
| weight_g_total | mediana; escala robusta só Ridge | mantida | Soma por item; ausente se houver peso ausente/inválido. |
| weight_g_mean | mediana; escala robusta só Ridge | mantida | Média dos pesos conhecidos por item. |
| volume_cm3_total | mediana; escala robusta só Ridge | mantida | Soma de comprimento × altura × largura; exige todos os itens. |
| length_cm_max | mediana; escala robusta só Ridge | mantida | Maior comprimento conhecido entre itens. |
| height_cm_max | mediana; escala robusta só Ridge | mantida | Maior altura conhecida entre itens. |
| width_cm_max | mediana; escala robusta só Ridge | mantida | Maior largura conhecida entre itens. |
| items_missing_weight_count | mediana; escala robusta só Ridge | mantida | Itens com peso ausente ou não positivo. |
| items_missing_dimensions_count | mediana; escala robusta só Ridge | mantida | Itens com alguma dimensão ausente ou não positiva. |
| items_missing_price_count | mediana; escala robusta só Ridge | removida (constante no ajuste) | Itens com preço ausente, infinito ou negativo. |
| items_missing_freight_count | mediana; escala robusta só Ridge | removida (constante no ajuste) | Itens com frete ausente, infinito ou negativo. |
| categories_count | mediana; escala robusta só Ridge | mantida | Número de categorias conhecidas distintas. |
| main_category | ausência explícita + one-hot | mantida | Moda por item; empate resolvido em ordem alfabética. |
| items_missing_category_count | mediana; escala robusta só Ridge | mantida | Itens sem categoria. |
| customer_state | ausência explícita + one-hot | mantida | UF do endereço do pedido. |
| distance_km_mean | mediana; escala robusta só Ridge | mantida | Haversine médio por item; apenas pares conhecidos. |
| distance_km_min | mediana; escala robusta só Ridge | mantida | Menor distância conhecida entre itens. |
| distance_km_max | mediana; escala robusta só Ridge | mantida | Maior distância conhecida entre itens. |
| items_missing_distance_count | mediana; escala robusta só Ridge | mantida | Itens sem par de coordenadas válido. |
| purchase_weekday | ausência explícita + one-hot | mantida | Dia da semana: segunda=0, domingo=6. |
| purchase_month | ausência explícita + one-hot | mantida | Mês da compra: 1 a 12. |
| purchase_hour | mediana; escala robusta só Ridge | mantida | Hora da compra: 0 a 23. |
| purchase_is_weekend | mediana; escala robusta só Ridge | mantida | Sábado/domingo=1; demais=0. |

No ajuste final do escolhido (Ridge), 144 colunas transformadas foram selecionadas. Nomes e máscaras completos: `resultados/tabelas/task02/features_final.csv`. Contrato de inferência: `resultados/modelos/task02/inference_contract.json`. Não se reutiliza vocabulário aprendido no teste.
