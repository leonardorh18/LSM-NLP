# Trabalho T2 — Arquitetura LSTM (GEX1083 — Deep Learning)

Modelo de linguagem treinado sobre o **Penn Treebank** com uma **LSTM
implementada inteiramente à mão** em NumPy — nenhuma camada pronta de
biblioteca é utilizada, conforme exigido no enunciado. O código foi escrito
priorizando **clareza**: cada função e cada linha estão documentadas.

## O que está implementado manualmente

| Componente | Observação |
|---|---|
| Embedding `E (V×D)` | indexação direta (equivale a one-hot·E) |
| LSTM `Wx (D×4H)`, `Wh (H×4H)`, `b (4H)` | 4 portas em um único produto de matrizes; viés da porta forget iniciado em +1 |
| Projeção de saída `Wy (H×V)`, `by (V)` | uma única multiplicação `(B·T,H)@(H,V)` para todos os passos |
| Softmax + cross-entropy | estabilizada subtraindo o máximo |
| **BPTT (backward)** | regra da cadeia escrita à mão, passo a passo no tempo |
| Clipping de gradiente | por norma global |
| **SGD** | atualização de uma linha: `peso -= lr * gradiente` |
| **Decaimento de LR manual** | lr ÷ 4 quando a validação estagna |
| **Early stopping manual** | paciência sobre a PPL de validação + restauração dos melhores pesos |

## Métrica: perplexity

A função de custo é a log-loss média (cross-entropy). A **perplexity é
e^logloss** ("euler elevado à log-loss") e é reportada em todas as fases
(treino/validação/teste). Interpretação: número efetivo de palavras entre
as quais o modelo fica indeciso a cada predição (aleatório = 10.000;
menor = melhor).

## Como executar

**Notebook (entrega principal):** `lstm_ptb.ipynb` — código em células com
explicações em Markdown (equações do forward, da BPTT, etc.). Basta executar
as células em ordem; os hiperparâmetros ficam na célula de configuração
(use `SMOKE = True` para uma execução rápida de ~30 s). O dataset é baixado
automaticamente para `./data` se não existir.

**Script equivalente:** `python lstm_ptb.py` (mesmas células concatenadas,
gera os gráficos em `resultados/` sem abrir janela).

## Validações e análises incluídas

- **Verificação numérica do gradiente:** o backward manual é comparado com
  diferenças finitas em um modelo pequeno antes do treinamento (seção 6).
- **Célula de análise final (seção 11):** tabela de logloss/PPL nas três
  divisões, gap de generalização, comparação com baselines (aleatório e
  unigrama), acurácia top-1/top-5 no teste, predições top-5 em contextos
  reais e texto gerado pelo modelo.

## Saídas

- `resultados/curvas_treinamento.png` — log-loss por minibatch, log-loss por
  época e perplexity (e^logloss) por época, treino × validação, com a PPL de
  teste marcada.
- `resultados/log_treinamento.txt` — log do treinamento completo (quando
  executado via script).
