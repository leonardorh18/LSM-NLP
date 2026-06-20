# Trabalho T2: Arquitetura LSTM (GEX1083 / Deep Learning)

Modelo de linguagem treinado sobre o **Penn Treebank** com uma **LSTM feita
inteiramente à mão** em NumPy, sem nenhuma camada pronta de biblioteca, como o
enunciado exige. Tudo (forward, backprop no tempo, loss, otimizador, etc.) está
escrito no próprio notebook e bem comentado.

Arquivo principal: **`lstm_ptb.ipynb`**. É só abrir e rodar as células de cima
para baixo. O dataset é baixado sozinho para `./data` se não existir.

## O que está implementado manualmente

| Componente | Observação |
|---|---|
| Embedding `E (V×D)` | indexação direta (equivale a one-hot·E) |
| LSTM `Wx (D×4H)`, `Wh (H×4H)`, `b (4H)` | 4 portões em um único produto de matrizes; viés do portão de esquecimento em +1 |
| Projeção de saída `Wy (H×V)`, `by (V)` | uma multiplicação `(B·T,H)@(H,V)` para todos os passos |
| Softmax + cross-entropy | estabilizada subtraindo o máximo |
| **BPTT (backward)** | regra da cadeia na mão, passo a passo no tempo |
| Clipping de gradiente | pela norma global |
| **SGD** | atualização de uma linha: `peso -= lr * gradiente` |
| **Decaimento de LR** | lr ÷ 4 quando a validação trava |
| **Early stopping** | paciência sobre a PPL de validação (+ margem mínima) e restauração dos melhores pesos |
| **Busca de hiperparâmetros** | grid search na mão (só laços `for`) sobre LR e clipping; escolhe o melhor pela PPL de validação |

## Métrica: perplexity

A função de custo é a log-loss média (cross-entropy). A **perplexity é
e^logloss** ("euler elevado à log-loss") e é reportada no treino, na validação e
no teste. Dá para pensar nela como o número efetivo de palavras entre as quais o
modelo fica na dúvida a cada chute (aleatório = 10.000; menor = melhor).

## Como funciona o notebook (ordem das seções)

1. dados (download, vocabulário, lotes)
2. matrizes do modelo + forward
3. backward (BPTT)
4. otimizador, clipping e learning rate
5. avaliação e geração de texto
6. **verificação numérica do gradiente** (confere se o backward está certo)
7. função de treino
8. configuração
9. **busca de hiperparâmetros** (testa a grade, ranqueia pela PPL de validação)
10. **treino final** com a melhor combinação no dataset completo
11. resultado no teste
12. gráficos
13. **análise** do modelo
14. conclusões

Para um teste rápido (~1 min), basta colocar `SMOKE = True` na célula de
configuração.

## Análises incluídas (seção 13)

- tabela de logloss/PPL nas três divisões e a diferença treino vs validação;
- comparação com baselines: chute aleatório e unigrama (só frequência);
- acurácia top-1 e top-5 no teste;
- as 5 palavras mais prováveis depois de alguns trechos reais;
- um texto gerado pelo modelo.

## Saídas (geradas em `resultados/`)

- `curvas_treinamento.png`: log-loss por minibatch, log-loss por época e
  perplexity por época (treino vs validação, com a PPL de teste marcada);
- `busca_por_epoca.csv`: uma linha por época de cada combinação da busca, com o
  tempo de cada época;
- `busca_resumo.csv`: ranking das combinações testadas;
- `treino_final_por_epoca.csv`: histórico do treino final, com o tempo por época.
