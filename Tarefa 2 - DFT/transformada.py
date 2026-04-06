import csv
import numpy as np
import matplotlib.pyplot as plt

sinal = []

with open("Sinal02.csv", "r") as arquivo:
    leitor = csv.reader(arquivo)
    for linha in leitor:
        sinal.append(float(linha[0]))               # Pega os valores do sinal e acrescenta no vetor chamado sinal


def transformada (xn, L):
    xn = np.asarray(xn, dtype=complex)
    Xk = np.zeros(L, dtype=complex)


    for k in range(L):
        soma = 0.0 + 0.0j 
        for n in range(L):
            soma += xn[n] * np.exp(-1j * 2 * np.pi * k * n / L)
        Xk[k] = soma                # Vetor no qual fica armazenada a DFT

    return Xk

def idft(Xk):
    Xk = np.asarray(Xk, dtype=complex)   # garante vetor complexo
    L = len(Xk)                          # número de pontos
    x = np.zeros(L, dtype=complex)       # aloca saída

    for n in range(L):                   # varre cada amostra no tempo
        soma = 0.0 + 0.0j
        for k in range(L):               # soma todas as frequências
            soma += Xk[k] * np.exp(1j * 2 * np.pi * k * n / L)
        x[n] = soma / L                  # divide por L (definição da IDFT)

    return x.real                        # se o sinal deveria ser real

def reconstruir(x, N_comp):
    L = len(x)
    X = transformada(x, L)                                     # DFT do sinal

    # Ordena índices das maiores magnitudes
    idx_sorted = np.argsort(np.abs(X))[::-1]
    idx_keep = idx_sorted[:N_comp]

    # Mantém só as N componentes
    X_filtrado = np.zeros_like(X, dtype=complex)
    X_filtrado[idx_keep] = X[idx_keep]

    # Reconstrução via IDFT
    x_rec = idft(X_filtrado)
    return x_rec.real

L = len(sinal)
Xk = transformada(sinal, L)
k = np.arange(L)        #Cria um vetor de 0 ate L-1
n = np.arange(L)

amp = np.abs(Xk)        #Pega a amplitude de xk e armazena no vetor

fase = np.angle(Xk)            #Pega a fase de xk

# --------- PLOTS ---------

# 1) Sinal no tempo
plt.subplot(3,1,1)
plt.plot(n, sinal, marker='.')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Sinal no tempo')
plt.grid(True)

# 2) Espectro de amplitude
plt.subplot(3,1,2)
plt.plot(k, amp, marker='.', linestyle='None')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DFT - Espectro de Amplitude')
plt.grid(True)

# 3) Espectro de fase
plt.subplot(3,1,3)
plt.plot(k, fase, marker='.', linestyle='None')
plt.xlabel('k')
plt.ylabel('Fase (rad)')
plt.title('DFT - Fase')
plt.grid(True)

plt.tight_layout()
plt.show()

x_rec = reconstruir(sinal, 8)

# 4) Sinal reconstruido no tempo
plt.figure(figsize=(8,4))
plt.plot(n, x_rec, marker='.', label='Reconstruído (8 comps)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Sinal reconstruído')
plt.grid(True)
plt.legend()
plt.show()
