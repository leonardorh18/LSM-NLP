# Trabalho T2: LSTM

Esse é o trabalho de uma LSTM treinada como modelo de linguagem em cima do
Penn Treebank. Na prática: você dá um pedaço de frase pro modelo e ele tenta
adivinhar a próxima palavra.

## Como foi feito

Montei a rede usando só NumPy pra fazer as contas de matriz (o forward da LSTM,
o backprop, a loss, o otimizador, etc.). Não usei camada pronta de biblioteca.

O que usei:
- **numpy**: toda a matemática do modelo
- **matplotlib**: pra desenhar os gráficos
- **pandas**: pra salvar os resultados em CSV

A métrica é a **perplexity** (que é o `e^logloss`). Quanto menor, melhor. Ela
aparece no treino, na validação e no teste.

## Como rodar

É só abrir o `lstm_ptb.ipynb` e rodar as células de cima pra baixo. O dataset
baixa sozinho na primeira vez. Se quiser só testar rápido, coloca `SMOKE = True`
na célula de configuração.

O notebook faz, mais ou menos nessa ordem: prepara os dados, monta o modelo,
confere se o gradiente está certo, testa algumas combinações de hiperparâmetros
e escolhe a melhor, treina o modelo final e no fim avalia os resultados.

## O que é gerado (pasta `resultados/`)

Pra avaliação ficam salvos:
- `curvas_treinamento.png`: os gráficos de log-loss e de perplexity por época
- `busca_por_epoca.csv`: cada época de cada combinação testada, com o tempo
- `busca_resumo.csv`: o ranking das combinações
- `treino_final_por_epoca.csv`: o histórico do treino final, com o tempo por época

Além disso, no fim do notebook tem uma análise com a perplexity nas três
divisões, comparação com modelos simples (chute aleatório e unigrama), acurácia,
o que o modelo chuta depois de algumas frases e um texto gerado por ele.
