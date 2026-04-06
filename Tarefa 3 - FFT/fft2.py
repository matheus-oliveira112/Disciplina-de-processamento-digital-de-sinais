import csv
import numpy as np
import matplotlib.pyplot as plt
import time 

sinal_original = []

with open("sinal01_C.csv", "r") as arquivo:
    conteudo = arquivo.readline().strip()                       # lê só a primeira linha
    sinal_original = [float(v) for v in conteudo.split(",")]               # Pega os valores do sinal e acrescenta no vetor chamado sinal

# Converte para numpy array (facilita a soma com o ruído)
sinal_original = np.array(sinal_original)

# Gera ruído branco 
ruido = np.random.uniform(-1, 1, size=sinal_original.shape)

sinal = sinal_original + ruido  #Soma o sinal existente com o ruido


def reconstruir_por_energia(X, frac):
    """
    Reconstrói o sinal no tempo a partir do espectro X (FFT completa),
    mantendo o menor conjunto de coeficientes que satisfaça a porcentagem requerida. 
    Também imprime informações sobre
    quantos coeficientes foram mantidos e quais são.
    """

    X = np.asarray(X, dtype=complex)
    N = X.size

    # Energia por coeficiente
    eng = (np.abs(X)**2) / (N**2)       #Calcula a energia media para cada coeficiente
    total = eng.sum()                   #Somatorio total das energias e feito

    # Ordena coeficientes por energia decrescente
    idx_ord = np.argsort(eng)[::-1]     #Ordena os coeficientes a partir da maior para menor energia
    cum = np.cumsum(eng[idx_ord])       #Somatorio das energias e feito a partir da maior para menor energia

    # Determina quantos coeficientes são necessários
    alvo = frac * total                         #Energia necessaria para satisfazer a porcentagem desejada
    k_idx = int(np.searchsorted(cum, alvo))     #Procura o coeficiente que satifaça a quantidade de energia minima para satisfazer a porcentagem desejada, o numero de harmonicas utilizadas
    idx_keep = idx_ord[:k_idx + 1]              #Procura os coeficientes das harmonicas que sao utilizadas para satisfazer... e armazena em um vetor

    # Cria espectro filtrado
    Xf = np.zeros_like(X, dtype=complex)        #Cria um vetor complexo cheio de zeros
    Xf[idx_keep] = X[idx_keep]                  #Armazena em Xf apenas as harmonicas utilizadas na reconstrução

    # Reconstrói no tempo
    x_rec = np.fft.ifft(Xf).real

    # Prints informativos
    print(f"\nReconstrução com {frac*100:.1f}% da energia:")
    print(f"  → Coeficientes mantidos: {k_idx+1} de {len(X)}")
    print(f"  → Índices das harmônicas mantidas: {idx_keep.tolist()}")

    return x_rec

def mse_tempo(y: np.ndarray, y_hat: np.ndarray):
    """
    Calcula e imprime o Erro Quadrático Médio (MSE) entre dois sinais no tempo.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"\nFormas incompatíveis: {y.shape} vs {y_hat.shape}\n")

    mse = np.mean((y - y_hat)**2)
    print(f"\nErro Quadrático Médio (MSE): {mse:.6e}\n")


inicio = time.perf_counter()
Xk = np.fft.fft(sinal)     #Faz a fft do sinal
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")

sinal_rec = reconstruir_por_energia(Xk, 0.96)         #faz a ifft do sinal que satisfaça a porcentagem desejada
mse_tempo(sinal_original, sinal_rec)

L = len(sinal)
k = np.arange(L)        #Cria um vetor de 0 ate L-1
n = np.arange(L)

amp = np.abs(Xk)        #Pega a amplitude de xk e armazena no vetor
fase = np.angle(Xk)            #Pega a fase de xk

# --------- PLOTS ---------

# 1) Sinal no tempo
plt.subplot(2,1,1)
plt.plot(n, sinal, marker='.')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Sinal no tempo')
plt.grid(True)

# 2) Sinal reconstruido no tempo
plt.subplot(2,1,2)
plt.plot(n, sinal_original, marker='.', color='b', label='Sinal original')
plt.plot(n, sinal_rec, marker='.', color='r', label='Sinal reconstruído')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Sinal original vs reconstruído')
plt.legend()  # legenda explicando as cores
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Espectro de amplitude da fft
plt.subplot(2,1,1)
plt.plot(k, amp, marker='.', linestyle='None')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DFT - Espectro de Amplitude')
plt.grid(True)

# 4) Espectro de fase
plt.subplot(2,1,2)
plt.plot(k, fase, marker='.', linestyle='None')
plt.xlabel('k')
plt.ylabel('Fase (rad)')
plt.title('DFT - Fase')
plt.grid(True)
plt.tight_layout()
plt.show()

