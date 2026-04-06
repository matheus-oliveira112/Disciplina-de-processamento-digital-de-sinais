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

def media_movel(x, M):
    N = len(x)
    y = []
    
    for i in range(N - M + 1):
        soma = 0
        for j in range(M):
            soma += x[i + j]
        y.append(soma / M)
    
    return y

def plota(sinal_freq, k, sinal_tempo):
    amp = np.abs(sinal_freq)

    # 1) Sinal no tempo
    plt.subplot(1,1,1)
    plt.plot(k, sinal_tempo, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
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


matriz = np.loadtxt("Sinal_1 Original.csv", delimiter=",")
sinal_original = matriz.flatten()               #Pega o sinal original
matriz1 = np.loadtxt("Sinal_1 Ruido.csv", delimiter=",")
sinal_ruido = matriz1.flatten()                 #Pega o sinal com ruido


L = len(sinal_original)
k = np.arange(L)        #Cria um vetor de 0 ate L-1
sinal_original_freq = np.fft.fft(sinal_original)
plota(sinal_original_freq, k, sinal_original)
sinal_ruido_freq = np.fft.fft(sinal_ruido)
plota(sinal_ruido_freq, k, sinal_ruido)


#ORDEM 4
y1_comruido = media_movel(sinal_ruido, 4)
N = min(len(y1_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y1_comruido = y1_comruido[:N]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N]
k1 = np.arange(len(y1_comruido))        #Cria um vetor de 0 ate L-1
y1_freq = np.fft.fft(y1_comruido)
mse_tempo(sinal_original, y1_comruido)
plota(y1_freq, k1, y1_comruido)

#ORDEM 8
y2_comruido = media_movel(sinal_ruido, 8)
N2 = min(len(y2_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y2_comruido = y2_comruido[:N2]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N2]
k2 = np.arange(len(y2_comruido))        #Cria um vetor de 0 ate L-1
y2_freq = np.fft.fft(y2_comruido)
mse_tempo(sinal_original, y2_comruido)
plota(y2_freq, k2, y2_comruido)

#ORDEM 16
y3_comruido = media_movel(sinal_ruido, 16)
N3 = min(len(y3_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y3_comruido = y3_comruido[:N3]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N3]
k3 = np.arange(len(y3_comruido))        #Cria um vetor de 0 ate L-1
y3_freq = np.fft.fft(y3_comruido)
mse_tempo(sinal_original, y3_comruido)
plota(y3_freq, k3, y3_comruido)



