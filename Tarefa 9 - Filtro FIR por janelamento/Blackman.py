import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time

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

def filtro_blackman(signal_in, fs, fc, M):

    # --- filtro FIR com frequência de corte absoluta e janela Hanning ---
    h = signal.firwin(M + 1, cutoff=fc, fs=fs, window='blackman')
    
    # --- aplica o filtro ---
    signal_out = signal.lfilter(h, 1.0, signal_in)
    
    return signal_out

def plota(sinal_freq, k, sinal_tempo):
    amp = np.abs(sinal_freq)

    # 1) Sinal no tempo
    plt.subplot(1,1,1)
    plt.plot(k, sinal_tempo, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
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


matriz = np.loadtxt("Sinal_Original.csv", delimiter=",")
sinal_original = matriz.flatten()               #Pega o sinal original
matriz1 = np.loadtxt("Sinal_Ruido.csv", delimiter=",")
sinal_ruido = matriz1.flatten()                 #Pega o sinal com ruido


#ORDEM 32
inicio = time.perf_counter()
y1_comruido = filtro_blackman(sinal_ruido, 512, 120, 32)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")
N = min(len(y1_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y1_comruido = y1_comruido[:N]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N]
k1 = np.arange(len(y1_comruido))        #Cria um vetor de 0 ate L-1
y1_freq = np.fft.fft(y1_comruido)
mse_tempo(sinal_original, y1_comruido)
plota(y1_freq, k1, y1_comruido)

#ORDEM 48
inicio = time.perf_counter()
y2_comruido = filtro_blackman(sinal_ruido, 512, 120, 48)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")
N2 = min(len(y2_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y2_comruido = y2_comruido[:N2]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N2]
k2 = np.arange(len(y2_comruido))        #Cria um vetor de 0 ate L-1
y2_freq = np.fft.fft(y2_comruido)
mse_tempo(sinal_original, y2_comruido)
plota(y2_freq, k2, y2_comruido)

#ORDEM 64
inicio = time.perf_counter()
y3_comruido = filtro_blackman(sinal_ruido, 512, 120, 64)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")
N3 = min(len(y3_comruido), len(sinal_original))       # Descobre o tamanho mínimo
y3_comruido = y3_comruido[:N3]       # Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N3]
k3 = np.arange(len(y3_comruido))        #Cria um vetor de 0 ate L-1
y3_freq = np.fft.fft(y3_comruido)
mse_tempo(sinal_original, y3_comruido)
plota(y3_freq, k3, y3_comruido)



