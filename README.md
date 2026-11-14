# Projeto 2 — Previsão do Tempo de Entrega na Olist

**UNIFOR · T326 — Ciência dos Dados · Prof. Caio Ponte · Turma 16/17**  
**Equipe:** Mergulhadores da Deep Web

Pipeline completo de Aprendizado de Máquina (regressão supervisionada) para estimar, **no momento da compra**, quantos dias um pedido levará entre a compra e a entrega ao cliente utilizando o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## Visão geral e principais resultados

- **Amostra:** `96.164` pedidos válidos após filtragem sequencial a partir dos `99.441` pedidos originais (`status == "delivered"`, data de entrega válida, com itens, tempo `> 0` e ruído extremo `> 60` dias removido).
- **Variável alvo (`delivery_time_days`):** `(order_delivered_customer_date - order_purchase_timestamp) / 86.400` segundos (média `12,32` dias · mediana `10,20` dias · desvio padrão `8,37` dias).
- **Features:** `30` variáveis construídas exclusivamente com dados disponíveis no instante da compra (`26` numéricas e `4` categóricas: `customer_state`, `main_category`, `purchase_weekday`, `purchase_month`).
- **Divisão e validação:** `80%` treino (`76.931` pedidos) / `20%` teste reservado (`19.233` pedidos) com `random_state=42`, e ajuste de hiperparâmetros com `KFold(5, shuffle=True, random_state=42)` dentro do `Pipeline` (imputação, `log1p`, padronização e `OneHotEncoder` sem vazamento → `146` colunas).

### Benchmark no conjunto de teste (`19.233` pedidos)

| Modelo | Otimização (CV 5 folds) | MAE (dias) | RMSE (dias) | R² | MAE CV |
|---|---|---:|---:|---:|---:|
| **Gradient Boosting (`HistGradientBoostingRegressor`)** | `RandomizedSearchCV` (10 comb.) | **4,535** | **6,637** | **0,390** | **4,493** |
| Random Forest (`RandomForestRegressor`) | `RandomizedSearchCV` (10 comb.) | 4,584 | 6,684 | 0,381 | 4,536 |
| Árvore de Decisão (`DecisionTreeRegressor`) | `GridSearchCV` (16 comb.) | 4,790 | 6,929 | 0,335 | 4,718 |
| Ridge (`Ridge`) | `GridSearchCV` (6 comb.) | 4,814 | 6,938 | 0,333 | 4,734 |
| Baseline constante (`DummyRegressor(median)`) | — | 5,970 | 8,768 | −0,065 | — |

- **Redução de erro:** o **Gradient Boosting** reduz o MAE em **24,0%** frente ao baseline (`4,535` vs `5,970` dias) e supera amplamente a precisão pontual do prazo conservador informado pela Olist (`MAE = 12,649` dias).
- **Diagnóstico de erros:** acerta bem na faixa central (`7–14` dias: `MAE = 3,24` d), mas subestima caudas longas (`30–60` dias: `MAE = 19,40` d), pois fatores pós-compra (atraso na postagem do vendedor, transportadora, greves e retenções fiscais) não estão disponíveis no checkout.

---

## Estrutura do repositório

```text
projeto-olist/
├── dados/
│   ├── brutos/brazilian-ecommerce.zip        # 9 CSVs originais do dataset Olist
│   └── tratados/olist_dataset.parquet        # Dataset limpo (96.164 pedidos × 36 colunas)
├── src/
│   └── olist_features.py                     # Leitura do ZIP, engenharia das 30 features e limpeza
├── notebooks/
│   ├── 01_preparacao_dados.ipynb             # Definição do problema, impacto, features e limpeza
│   ├── 02_eda.ipynb                          # Estatísticas, distribuições, IQR, correlações e mapa
│   └── 03_modelagem.ipynb                    # Pipeline, busca de hiperparâmetros, benchmark e diagnóstico
├── resultados/
│   ├── graficos/                             # Gráficos 01_*.png a 10_*.png (EDA e modelagem)
│   ├── modelos/melhor_modelo.joblib          # Pipeline final treinado (Gradient Boosting)
│   └── tabelas/                              # CSVs e JSON das etapas 01, 02 e 03
├── apresentacao/
│   ├── index.html                            # Slides interativos HTML (15 slides 16:9)
│   ├── Projeto_2_Olist.pdf                   # PDF exportado da apresentação (15 páginas)
│   ├── roteiro.md                            # Roteiro cronometrado (15 min · 5 integrantes)
│   ├── checagem_rubrica.md                   # Conferência com os requisitos do enunciado
│   └── assets/                               # slides.css e slides.js
└── requirements.txt                          # Dependências Python 3.13.6 verificadas
```

---

## Como reproduzir e visualizar

1. **Instalar dependências** (Python `3.13.6`):
   ```bash
   pip install -r requirements.txt
   ```
2. **Executar os notebooks em ordem**:
   - `notebooks/01_preparacao_dados.ipynb` → gera `dados/tratados/olist_dataset.parquet` e tabelas `01_*`
   - `notebooks/02_eda.ipynb` → gera tabelas `02_*` e gráficos `01` a `06`
   - `notebooks/03_modelagem.ipynb` → gera `resultados/modelos/melhor_modelo.joblib`, tabelas `03_*` e gráficos `07` a `10`
3. **Apresentação**:
   - Abrir `apresentacao/index.html` localmente no Chrome/Edge (`→`/`←` navegam, `F` tela cheia, `N` notas do apresentador, `P` imprimir/PDF) ou abrir diretamente `apresentacao/Projeto_2_Olist.pdf`.
