# -*- coding: utf-8 -*-
"""Versão script do notebook lstm_ptb.ipynb (células concatenadas).
Gerado automaticamente — o notebook é o documento principal."""
import matplotlib
matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela


# %% ======================================================================
import os                    # operações com pastas e caminhos de arquivo
import time                  # para medir a duração de cada época
import urllib.request        # para baixar o dataset da internet

import numpy as np           # TODA a matemática do modelo usa apenas NumPy
import matplotlib.pyplot as plt   # usado somente para desenhar os gráficos

# %% ======================================================================
URL_PTB = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data"  # fonte do dataset
ARQUIVOS = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]              # treino/validação/teste


def baixar_ptb(pasta):
    """Baixa os 3 arquivos do Penn Treebank, caso ainda não existam na pasta."""
    os.makedirs(pasta, exist_ok=True)                  # cria a pasta se não existir
    for nome in ARQUIVOS:                              # para cada um dos 3 arquivos
        caminho = os.path.join(pasta, nome)            # caminho local do arquivo
        if not os.path.exists(caminho):                # só baixa se ainda não foi baixado
            print(f"[dados] baixando {nome} ...")
            urllib.request.urlretrieve(f"{URL_PTB}/{nome}", caminho)   # download


def ler_tokens(caminho):
    """Lê o arquivo de texto e devolve a lista de palavras (tokens)."""
    with open(caminho, "r", encoding="utf-8") as f:    # abre o arquivo
        texto = f.read()                               # lê tudo como uma string
    texto = texto.replace("\n", " <eos> ")             # fim de linha vira o token <eos>
    return texto.split()                               # separa por espaços -> lista de palavras


def construir_vocabulario(tokens_treino):
    """
    Dá um número (id) para cada palavra distinta do TREINO.
    Importante: usamos somente o treino — montar o vocabulário olhando a
    validação ou o teste seria vazamento de dados.
    """
    palavras = sorted(set(tokens_treino))                        # palavras únicas, ordem fixa
    palavra_para_id = {w: i for i, w in enumerate(palavras)}     # dicionário palavra -> id
    id_para_palavra = {i: w for i, w in enumerate(palavras)}     # dicionário id -> palavra
    return palavra_para_id, id_para_palavra


def tokens_para_ids(tokens, palavra_para_id):
    """Troca cada palavra pelo seu id (palavras desconhecidas viram <unk>)."""
    unk = palavra_para_id["<unk>"]                               # id do token "desconhecido"
    ids = [palavra_para_id.get(t, unk) for t in tokens]          # converte palavra por palavra
    return np.array(ids, dtype=np.int32)                         # devolve como vetor NumPy


def montar_lotes_continuos(ids, B):
    """
    Reorganiza o corpus (vetor de ids) em duas matrizes (B, n):
      entradas[b, t] = palavra na posição t da fatia b
      alvos[b, t]    = a palavra SEGUINTE (é o que o modelo deve prever)
    """
    n = (len(ids) - 1) // B                            # colunas que cabem (sobra é descartada)
    entradas = ids[0 : n * B].reshape(B, n)            # corpus fatiado em B linhas contíguas
    alvos = ids[1 : n * B + 1].reshape(B, n)           # o mesmo, deslocado 1 posição à frente
    return entradas, alvos

# %% ======================================================================
def sigmoid(x):
    """Função logística: achata qualquer número para o intervalo (0, 1)."""
    x = np.clip(x, -50, 50)            # limita x para a exponencial não estourar
    return 1.0 / (1.0 + np.exp(-x))    # fórmula clássica da sigmoide


def inicializar_parametros(V, D, H, semente):
    """Cria TODAS as matrizes de pesos do modelo, à mão."""
    rng = np.random.default_rng(semente)   # gerador de aleatórios com semente fixa (reprodutível)

    def aleatoria(linhas, colunas):
        """Matriz (linhas x colunas) com valores uniformes em [-0.05, +0.05]."""
        return rng.uniform(-0.05, 0.05, (linhas, colunas)).astype(np.float32)

    p = {
        "E":  aleatoria(V, D),                      # embedding: 1 linha (vetor) por palavra
        "Wx": aleatoria(D, 4 * H),                  # pesos da entrada para as 4 portas
        "Wh": aleatoria(H, 4 * H),                  # pesos recorrentes (h anterior -> portas)
        "b":  np.zeros(4 * H, dtype=np.float32),    # vieses das 4 portas
        "Wy": aleatoria(H, V),                      # projeção: estado oculto -> vocabulário
        "by": np.zeros(V, dtype=np.float32),        # viés da saída
    }
    p["b"][H : 2 * H] = 1.0    # viés da porta de ESQUECIMENTO começa em +1 ("lembrar por padrão")
    return p

# %% ======================================================================
def forward(p, entradas, alvos, h, c, H):
    """Passada de ida da LSTM sobre uma janela (B, T). Devolve (perda, h, c, cache)."""
    B, T = entradas.shape                  # B = sequências em paralelo, T = passos no tempo
    X = p["E"][entradas]                   # busca o embedding de cada id -> tensor (B, T, D)

    # O cache guarda os valores intermediários de cada passo (o backward precisa deles).
    cache = {"X": X, "entradas": entradas, "h_ant": [], "c_ant": [],
             "i": [], "f": [], "g": [], "o": [], "c": []}
    H_todos = np.empty((B, T, H), dtype=np.float32)    # guardará o h de cada passo

    for t in range(T):                     # anda no tempo, um passo por vez
        cache["h_ant"].append(h)           # guarda o h ANTERIOR a este passo
        cache["c_ant"].append(c)           # guarda o c ANTERIOR a este passo
        a = X[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["b"]   # pré-ativação das 4 portas: (B, 4H)
        i = sigmoid(a[:, 0 * H : 1 * H])   # porta de ENTRADA (0 a 1)
        f = sigmoid(a[:, 1 * H : 2 * H])   # porta de ESQUECIMENTO (0 a 1)
        g = np.tanh(a[:, 2 * H : 3 * H])   # CANDIDATO a nova memória (-1 a 1)
        o = sigmoid(a[:, 3 * H : 4 * H])   # porta de SAÍDA (0 a 1)
        c = f * c + i * g                  # atualiza a memória: esquece um pouco, aprende um pouco
        h = o * np.tanh(c)                 # calcula a saída deste passo
        cache["i"].append(i); cache["f"].append(f)   # guarda as portas para o backward
        cache["g"].append(g); cache["o"].append(o)
        cache["c"].append(c)               # guarda o c novo (o backward usa tanh(c))
        H_todos[:, t, :] = h               # guarda a saída deste passo

    # ---- projeção para o vocabulário + softmax + log-loss (todos os passos de uma vez) ----
    Hf = H_todos.reshape(B * T, H)                     # empilha os h de todos os passos
    logits = Hf @ p["Wy"] + p["by"]                    # pontuação de cada palavra: (B*T, V)
    logits = logits - logits.max(axis=1, keepdims=True)  # subtrai o máximo (estabilidade numérica)
    probs = np.exp(logits)                             # exponencia as pontuações
    probs = probs / probs.sum(axis=1, keepdims=True)   # normaliza: cada linha soma 1 (softmax)
    alvos_1d = alvos.reshape(B * T)                    # achata os alvos para casar com probs
    prob_da_certa = probs[np.arange(B * T), alvos_1d]  # prob. que o modelo deu à palavra correta
    perda = -np.log(prob_da_certa).mean()              # log-loss média (PPL será e^perda)

    cache["Hf"] = Hf; cache["probs"] = probs; cache["alvos_1d"] = alvos_1d
    return perda, h, c, cache

# %% ======================================================================
def backward(p, cache, H):
    """Calcula o gradiente da perda em relação a TODAS as matrizes do modelo."""
    X = cache["X"]                         # embeddings usados no forward: (B, T, D)
    B, T, D = X.shape                      # recupera as dimensões
    N = B * T                              # número total de predições na janela

    grad = {nome: np.zeros_like(m) for nome, m in p.items()}   # gradientes começam em zero

    # ---- 1) gradiente da softmax + log-loss: (probs - onehot) / N ----
    dlogits = cache["probs"].copy()                  # começa igual às probabilidades
    dlogits[np.arange(N), cache["alvos_1d"]] -= 1.0  # subtrai 1 na palavra correta
    dlogits /= N                                     # divide por N porque a perda é uma média

    # ---- 2) camada de saída (projeção para o vocabulário) ----
    grad["Wy"] = cache["Hf"].T @ dlogits             # gradiente da projeção
    grad["by"] = dlogits.sum(axis=0)                 # gradiente do viés da saída
    dH = (dlogits @ p["Wy"].T).reshape(B, T, H)      # gradiente que chega em cada h_t

    # ---- 3) anda no tempo DE TRÁS PARA FRENTE ----
    dX = np.empty_like(X)                            # gradiente dos embeddings
    dh_prox = np.zeros((B, H), dtype=np.float32)     # gradiente vindo do passo seguinte (h)
    dc_prox = np.zeros((B, H), dtype=np.float32)     # gradiente vindo do passo seguinte (c)
    for t in range(T - 1, -1, -1):                   # do último passo para o primeiro
        i = cache["i"][t]; f = cache["f"][t]         # recupera as portas guardadas
        g = cache["g"][t]; o = cache["o"][t]
        tanh_c = np.tanh(cache["c"][t])              # tanh da memória deste passo
        dh = dH[:, t, :] + dh_prox                   # gradiente total que chega em h_t
        do = dh * tanh_c                             # h = o*tanh(c)  ->  derivada em relação a o
        dc = dc_prox + dh * o * (1 - tanh_c ** 2)    # ... e em relação a c
        df = dc * cache["c_ant"][t]                  # c = f*c_ant + i*g  ->  derivada de cada termo
        di = dc * g
        dg = dc * i
        # volta pelas ativações das portas (sigmoide: s(1-s); tanh: 1-g^2),
        # juntando os 4 pedaços na MESMA ordem do forward: [i, f, g, o]
        da = np.hstack([di * i * (1 - i),
                        df * f * (1 - f),
                        dg * (1 - g ** 2),
                        do * o * (1 - o)])           # (B, 4H)
        grad["Wx"] += X[:, t, :].T @ da              # acumula o gradiente de Wx ...
        grad["Wh"] += cache["h_ant"][t].T @ da       # ... de Wh ...
        grad["b"] += da.sum(axis=0)                  # ... e dos vieses
        dX[:, t, :] = da @ p["Wx"].T                 # gradiente que chega no embedding
        dh_prox = da @ p["Wh"].T                     # gradiente enviado ao passo anterior (h)
        dc_prox = dc * f                             # gradiente enviado ao passo anterior (c)

    # ---- 4) embedding: soma o gradiente na linha de cada palavra usada ----
    np.add.at(grad["E"], cache["entradas"].reshape(N), dX.reshape(N, D))
    return grad

# %% ======================================================================
def atualizar_sgd(p, grad, lr):
    """SGD: cada matriz anda um passinho contra o seu gradiente."""
    for nome in p:                     # para cada matriz do modelo (E, Wx, Wh, b, Wy, by)
        p[nome] -= lr * grad[nome]     # regra da descida do gradiente


def clipar_gradientes(grad, limite):
    """Se o gradiente (todas as matrizes juntas) for grande demais, encolhe tudo."""
    soma_quadrados = 0.0
    for m in grad.values():                    # soma dos quadrados de todos os números
        soma_quadrados += float((m * m).sum())
    norma = np.sqrt(soma_quadrados)            # comprimento (norma L2) do gradiente inteiro
    if norma > limite:                         # passou do limite?
        for m in grad.values():
            m *= limite / norma                # encolhe todas as matrizes na mesma proporção
    return norma

# %% ======================================================================
def avaliar(p, entradas, alvos, H, B, T):
    """Mede a log-loss média de um conjunto inteiro e devolve (logloss, PPL = e^logloss)."""
    h = np.zeros((B, H), dtype=np.float32)     # estado inicial zerado
    c = np.zeros((B, H), dtype=np.float32)
    soma_perda = 0.0                           # acumula perda * número de palavras
    total_palavras = 0                         # conta as palavras avaliadas
    n_colunas = entradas.shape[1]              # comprimento total de cada fatia
    for inicio in range(0, n_colunas, T):      # percorre em janelas de T passos
        xb = entradas[:, inicio : inicio + T]  # janela de entradas
        yb = alvos[:, inicio : inicio + T]     # janela de alvos
        perda, h, c, _ = forward(p, xb, yb, h, c, H)   # só forward (sem aprender)
        soma_perda += perda * xb.size          # soma ponderada pela qtde de palavras
        total_palavras += xb.size
    logloss = soma_perda / total_palavras      # log-loss média do conjunto
    ppl = float(np.exp(logloss))               # PERPLEXITY = e^logloss
    return logloss, ppl

# %% ======================================================================
def checar_gradiente():
    """Compara o gradiente do backward com diferenças finitas num modelo pequeno."""
    V, D, H, B, T = 12, 7, 9, 4, 5                         # modelo minúsculo (rápido de testar)
    rng = np.random.default_rng(0)
    p = inicializar_parametros(V, D, H, semente=0)
    p = {k: v.astype(np.float64) for k, v in p.items()}    # float64 = mais precisão no teste
    entradas = rng.integers(0, V, (B, T)).astype(np.int32)  # janela de entrada aleatória
    alvos = rng.integers(0, V, (B, T)).astype(np.int32)     # alvos aleatórios
    h0 = np.zeros((B, H)); c0 = np.zeros((B, H))            # estados iniciais

    _, _, _, cache = forward(p, entradas, alvos, h0, c0, H)  # 1 forward ...
    grad = backward(p, cache, H)                             # ... e 1 backward analítico

    eps = 1e-5                                  # tamanho da perturbação
    tudo_ok = True
    for nome, matriz in p.items():              # para cada matriz do modelo
        plana = matriz.reshape(-1)              # vê a matriz como um vetor
        pior = 0.0                              # pior diferença encontrada nesta matriz
        quantos = min(20, plana.size)           # checa até 20 posições de cada matriz
        for ix in rng.choice(plana.size, size=quantos, replace=False):  # posições aleatórias
            original = plana[ix]
            plana[ix] = original + eps          # perturba para cima ...
            perda_mais, _, _, _ = forward(p, entradas, alvos, h0, c0, H)
            plana[ix] = original - eps          # ... e para baixo
            perda_menos, _, _, _ = forward(p, entradas, alvos, h0, c0, H)
            plana[ix] = original                # restaura o valor
            numerico = (perda_mais - perda_menos) / (2 * eps)    # gradiente "medido"
            analitico = grad[nome].reshape(-1)[ix]               # gradiente do backward
            pior = max(pior, abs(numerico - analitico))          # guarda a pior diferença
        ok = pior < 1e-6                        # tolerância (diferenças finitas têm ruído ~1e-9)
        tudo_ok = tudo_ok and ok
        print(f"  {nome:3s}: {'ok' if ok else 'FALHOU'} (pior diferença: {pior:.1e})")
    print("== Gradiente", "CORRETO ==" if tudo_ok else "ERRADO ==")


checar_gradiente()   # executa a verificação

# %% ======================================================================
EPOCAS_MAX = 20      # número máximo de épocas (o early stopping costuma parar antes)
B = 32               # tamanho do minibatch (sequências processadas em paralelo)
T = 25               # tamanho da janela da BPTT truncada (passos no tempo)
H = 200              # dimensão do estado oculto da LSTM
D = 200              # dimensão dos embeddings
LR_INICIAL = 20.0    # taxa de aprendizado do SGD (alta porque a perda é média por palavra)
FATOR_DECAIMENTO = 4.0   # divide a LR por 4 quando a validação não melhora
CLIP = 0.25          # norma global máxima permitida para o gradiente
PACIENCIA = 3        # épocas seguidas sem melhora para acionar o early stopping
SEMENTE = 42         # semente dos aleatórios (reprodutibilidade)
SMOKE = False        # True = execução rápida de sanidade

# %% ======================================================================
baixar_ptb("data")                                            # garante que o dataset existe

tokens_tr = ler_tokens(os.path.join("data", "ptb.train.txt"))  # lê as palavras do treino
tokens_va = ler_tokens(os.path.join("data", "ptb.valid.txt"))  # ... da validação
tokens_te = ler_tokens(os.path.join("data", "ptb.test.txt"))   # ... e do teste

if SMOKE:                          # modo rápido: usa só um pedacinho de cada conjunto
    tokens_tr = tokens_tr[:40000]
    tokens_va = tokens_va[:8000]
    tokens_te = tokens_te[:8000]
    EPOCAS_MAX = 2

palavra_para_id, id_para_palavra = construir_vocabulario(tokens_tr)   # monta o vocabulário
V = len(palavra_para_id)                                              # tamanho do vocabulário

ids_tr = tokens_para_ids(tokens_tr, palavra_para_id)   # treino como vetor de ids
ids_va = tokens_para_ids(tokens_va, palavra_para_id)   # validação como vetor de ids
ids_te = tokens_para_ids(tokens_te, palavra_para_id)   # teste como vetor de ids

Xtr, Ytr = montar_lotes_continuos(ids_tr, B)   # matrizes (B, n) de entrada/alvo do treino
Xva, Yva = montar_lotes_continuos(ids_va, B)   # ... da validação
Xte, Yte = montar_lotes_continuos(ids_te, B)   # ... e do teste

print(f"vocabulário: {V} palavras")
print(f"tokens: treino={len(ids_tr):,} | validação={len(ids_va):,} | teste={len(ids_te):,}")

# %% ======================================================================
p = inicializar_parametros(V, D, H, SEMENTE)        # cria as matrizes do modelo
lr = LR_INICIAL                                     # LR atual (vai decaindo)

melhor_ppl_va = float("inf")                        # melhor PPL de validação vista
melhores_pesos = None                               # cópia dos pesos do melhor momento
epocas_sem_melhora = 0                              # contador do early stopping

historico = {"perda_tr": [], "ppl_tr": [],          # guarda as métricas de cada época
             "perda_va": [], "ppl_va": [],
             "perda_passos": []}                    # perda de cada minibatch (curva fina)

janelas_por_epoca = Xtr.shape[1] // T               # quantas janelas de T passos cabem
print(f"modelo: V={V} D={D} H={H} | B={B} T={T} | lr={lr} clip={CLIP}")
print(f"{janelas_por_epoca} janelas por época | máx. {EPOCAS_MAX} épocas | paciência {PACIENCIA}\n")

for epoca in range(1, EPOCAS_MAX + 1):
    inicio_epoca = time.time()                      # cronômetro da época
    h = np.zeros((B, H), dtype=np.float32)          # estado zerado no começo da época
    c = np.zeros((B, H), dtype=np.float32)
    soma_perda, total_palavras = 0.0, 0             # acumuladores da perda média

    for j in range(janelas_por_epoca):              # percorre o treino janela a janela
        col = j * T                                 # coluna inicial desta janela
        xb = Xtr[:, col : col + T]                  # entradas da janela
        yb = Ytr[:, col : col + T]                  # alvos da janela
        perda, h, c, cache = forward(p, xb, yb, h, c, H)   # 1) passada de ida
        grad = backward(p, cache, H)                # 2) gradientes via BPTT
        clipar_gradientes(grad, CLIP)               # 3) evita explosão do gradiente
        atualizar_sgd(p, grad, lr)                  # 4) passo do SGD
        soma_perda += perda * xb.size               # acumula para a média da época
        total_palavras += xb.size
        historico["perda_passos"].append(perda)     # guarda a perda deste minibatch
        if (j + 1) % max(1, janelas_por_epoca // 5) == 0:    # imprime 5 parciais por época
            print(f"  época {epoca} janela {j+1}/{janelas_por_epoca} "
                  f"| logloss {soma_perda/total_palavras:.3f} "
                  f"| PPL {np.exp(soma_perda/total_palavras):.1f}")

    perda_tr = soma_perda / total_palavras          # log-loss média da época no treino
    ppl_tr = float(np.exp(perda_tr))                # PPL de treino = e^logloss
    perda_va, ppl_va = avaliar(p, Xva, Yva, H, B, T)  # mede na validação (sem aprender)
    historico["perda_tr"].append(perda_tr); historico["ppl_tr"].append(ppl_tr)
    historico["perda_va"].append(perda_va); historico["ppl_va"].append(ppl_va)
    print(f"[época {epoca:2d}] treino: PPL={ppl_tr:.1f} | validação: PPL={ppl_va:.1f} "
          f"| lr={lr:g} | {time.time()-inicio_epoca:.0f}s")

    # ----------------- EARLY STOPPING manual -----------------
    if ppl_va < melhor_ppl_va:                      # a validação melhorou?
        melhor_ppl_va = ppl_va                      # registra o novo recorde
        melhores_pesos = {k: v.copy() for k, v in p.items()}   # fotografa os pesos
        epocas_sem_melhora = 0                      # zera o contador
        print(f"            -> melhorou; pesos salvos (PPL validação {ppl_va:.1f})")
    else:                                           # não melhorou:
        epocas_sem_melhora += 1                     # conta mais uma época ruim
        lr = lr / FATOR_DECAIMENTO                  # DECAIMENTO manual da LR
        print(f"            -> sem melhora ({epocas_sem_melhora}x); lr reduzida para {lr:g}")
        if epocas_sem_melhora >= PACIENCIA:         # paciência esgotou?
            print("            -> EARLY STOP")
            break                                   # para o treinamento

if melhores_pesos is not None:
    p = melhores_pesos                              # restaura o MELHOR modelo (não o último)

# %% ======================================================================
perda_te, ppl_te = avaliar(p, Xte, Yte, H, B, T)    # avalia o melhor modelo no teste
print("================= RESULTADO FINAL =================")
print(f" logloss (teste)    : {perda_te:.4f}")
print(f" PERPLEXITY (teste) : e^{perda_te:.4f} = {ppl_te:.2f}")
print(f" melhor PPL validação: {melhor_ppl_va:.2f}")
print("===================================================")

# %% ======================================================================
eixo_epocas = np.arange(1, len(historico["perda_tr"]) + 1)   # 1, 2, 3, ... épocas treinadas

fig, eixos = plt.subplots(1, 3, figsize=(16, 4.5))           # três gráficos lado a lado

# (a) perda de cada minibatch: mostra a dinâmica fina do treinamento
eixos[0].plot(historico["perda_passos"], lw=0.5, color="tab:gray")
eixos[0].set_title("Log-loss por minibatch (treino)")
eixos[0].set_xlabel("minibatch"); eixos[0].set_ylabel("logloss")

# (b) log-loss média por época, treino x validação
eixos[1].plot(eixo_epocas, historico["perda_tr"], "o-", label="treino")
eixos[1].plot(eixo_epocas, historico["perda_va"], "s-", label="validação")
eixos[1].set_title("Log-loss por época")
eixos[1].set_xlabel("época"); eixos[1].set_ylabel("logloss")
eixos[1].legend(); eixos[1].grid(alpha=0.3)

# (c) perplexity (e^logloss) por época, com a PPL final de teste marcada
eixos[2].plot(eixo_epocas, historico["ppl_tr"], "o-", label="treino")
eixos[2].plot(eixo_epocas, historico["ppl_va"], "s-", label="validação")
eixos[2].axhline(ppl_te, color="tab:red", ls="--", label=f"teste = {ppl_te:.1f}")
eixos[2].set_title("Perplexity (e^logloss) por época")
eixos[2].set_xlabel("época"); eixos[2].set_ylabel("PPL")
eixos[2].legend(); eixos[2].grid(alpha=0.3)

fig.tight_layout()
os.makedirs("resultados", exist_ok=True)                      # pasta para salvar a figura
fig.savefig(os.path.join("resultados", "curvas_treinamento.png"), dpi=130)
plt.show()

# %% ======================================================================
# ============ 1) tabela-resumo: logloss e PPL nas três divisões ============
perda_tr_f, ppl_tr_f = avaliar(p, Xtr, Ytr, H, B, T)   # treino (agora SEM aprender)
perda_va_f, ppl_va_f = avaliar(p, Xva, Yva, H, B, T)   # validação
perda_te_f, ppl_te_f = avaliar(p, Xte, Yte, H, B, T)   # teste

print("conjunto     logloss   PPL = e^logloss")
print(f"treino       {perda_tr_f:7.4f}   {ppl_tr_f:8.2f}")
print(f"validação    {perda_va_f:7.4f}   {ppl_va_f:8.2f}")
print(f"teste        {perda_te_f:7.4f}   {ppl_te_f:8.2f}")
print(f"\ngap de generalização (PPL validação - PPL treino): {ppl_va_f - ppl_tr_f:.1f}")
print("(gap pequeno = pouco overfitting; o early stopping parou antes de decorar o treino)\n")

# ============ 2) comparação com modelos de referência ============
# unigrama: P(palavra) = frequência dela no treino (nenhum contexto é usado)
contagens = np.bincount(ids_tr, minlength=V)           # quantas vezes cada palavra aparece
prob_unigrama = contagens / contagens.sum()            # vira probabilidade
logloss_uni = -np.log(prob_unigrama[ids_te]).mean()    # log-loss do unigrama no teste
ppl_uni = float(np.exp(logloss_uni))                   # PPL do unigrama = e^logloss

print("modelo                        PPL no teste")
print(f"chute aleatório               {V:10.1f}   (1/V para cada palavra)")
print(f"unigrama (só frequência)      {ppl_uni:10.1f}")
print(f"nossa LSTM                    {ppl_te_f:10.1f}   "
      f"({ppl_uni/ppl_te_f:.1f}x melhor que o unigrama)\n")

# ============ 3) acurácia top-1 e top-5 no teste ============
acertos_top1, acertos_top5, total = 0, 0, 0
h = np.zeros((B, H), dtype=np.float32)                 # estado inicial zerado
c = np.zeros((B, H), dtype=np.float32)
for inicio in range(0, Xte.shape[1], T):               # percorre o teste em janelas
    xb = Xte[:, inicio : inicio + T]
    yb = Yte[:, inicio : inicio + T]
    _, h, c, cache = forward(p, xb, yb, h, c, H)       # só forward
    probs = cache["probs"]                             # (n, V): distribuição prevista
    alvos_1d = cache["alvos_1d"]                       # palavra correta de cada posição
    top5 = np.argsort(probs, axis=1)[:, -5:]           # as 5 palavras mais prováveis
    acertos_top1 += (top5[:, -1] == alvos_1d).sum()    # a última coluna é a nº 1
    acertos_top5 += (top5 == alvos_1d[:, None]).any(axis=1).sum()   # correta entre as 5?
    total += len(alvos_1d)

print(f"acurácia top-1 no teste: {100 * acertos_top1 / total:.1f}% "
      f"(palavra correta é a 1ª aposta)")
print(f"acurácia top-5 no teste: {100 * acertos_top5 / total:.1f}% "
      f"(palavra correta está entre as 5 primeiras)\n")

# ============ 4) o que o modelo prevê depois de trechos reais ============
trechos = ["the federal government", "he said the", "in new york",
           "the company said it"]
for trecho in trechos:
    h1 = np.zeros((1, H), dtype=np.float32)            # estado zerado, 1 sequência só
    c1 = np.zeros((1, H), dtype=np.float32)
    for palavra in trecho.split():                     # alimenta o trecho palavra a palavra
        entrada = np.array([[palavra_para_id[palavra]]], dtype=np.int32)
        alvo_falso = np.array([[0]], dtype=np.int32)   # alvo irrelevante: só queremos as probs
        _, h1, c1, cache = forward(p, entrada, alvo_falso, h1, c1, H)
    probs = cache["probs"][0]                          # distribuição da PRÓXIMA palavra
    melhores = probs.argsort()[-5:][::-1]              # 5 mais prováveis, da maior p/ menor
    sugestoes = ", ".join(f"{id_para_palavra[i]} ({100*probs[i]:.0f}%)" for i in melhores)
    print(f'"{trecho} ..." -> {sugestoes}')

# ============ 5) texto gerado pelo modelo (qualitativo) ============
rng_amostra = np.random.default_rng(123)               # semente própria da amostragem
h1 = np.zeros((1, H), dtype=np.float32)
c1 = np.zeros((1, H), dtype=np.float32)
atual = palavra_para_id["<eos>"]                       # começa em "início de frase"
geradas = []
for _ in range(50):                                    # gera 50 palavras
    entrada = np.array([[atual]], dtype=np.int32)
    alvo_falso = np.array([[0]], dtype=np.int32)
    _, h1, c1, cache = forward(p, entrada, alvo_falso, h1, c1, H)
    probs = cache["probs"][0]                          # distribuição da próxima palavra
    atual = int(rng_amostra.choice(V, p=probs))        # sorteia a próxima palavra
    geradas.append(id_para_palavra[atual])             # guarda a palavra sorteada
print("\ntexto gerado pelo modelo:")
print(" ".join(geradas))
