import csv
import numpy as np
import matplotlib.pyplot as plt

sinal_original = []
sinal_ruido = []

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

matriz = np.loadtxt("Sinal1 Original.csv", delimiter=",")
sinal_original = matriz.flatten()               #Pega o sinal original
matriz1 = np.loadtxt("Sinal1 Ruido.csv", delimiter=",")
sinal_ruido = matriz1.flatten()                 #Pega o sinal com ruido

sinal_original = sinal_original[::1200]           #Faz a subamostragem do sinal
sinal_ruido = sinal_ruido[::1200]

xk = np.fft.fft(sinal_ruido)

L = len(sinal_original)
k = np.arange(L)        #Cria um vetor de 0 ate L-1
n = np.arange(L)

amp = np.abs(xk)

sinal_ruido = np.fft.ifft(xk).real

mse_tempo(sinal_original, sinal_ruido)

# --------- PLOTS ---------

# 1) Sinal no tempo
plt.subplot(1,1,1)
plt.plot(n, sinal_original, marker='.', color='b', label='Sinal original')
plt.plot(n, sinal_ruido, marker='.', color='r', label='Sinal com ruido')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Sinal original vs com ruido')
plt.legend()  # legenda explicando as cores
plt.grid(True)
plt.show()

# 2) Espectro de amplitude da fft
plt.subplot(1,1,1)
plt.stem(k, amp, linefmt='b-', markerfmt='b.', basefmt='k-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('FFT - Espectro de Amplitude')
plt.grid(True)
plt.show()
