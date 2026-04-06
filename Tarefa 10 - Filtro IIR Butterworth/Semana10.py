import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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

def filtro_hamming(signal_in, fs, fc, M):

    # --- filtro FIR com frequência de corte absoluta e janela Hanning ---
    h = signal.firwin(M + 1, cutoff=fc, fs=fs, window='hamming')
    
    # --- aplica o filtro ---
    signal_out = signal.lfilter(h, 1.0, signal_in)
    
    return signal_out

def plota(sinal_freq, k, sinal_tempo, fs):
    amp = np.abs(sinal_freq)
    N = len(sinal_freq)

    # 1) Sinal no tempo
    plt.subplot(1,1,1)
    plt.plot(k, sinal_tempo, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
    plt.grid(True)
    plt.show()

    x = k*fs/N
    # 2) Espectro de amplitude da fft
    plt.subplot(1,1,1)
    plt.stem(x, amp, linefmt='b-', markerfmt='b.', basefmt='k-')
    plt.xlabel('Hz')
    plt.ylabel('|X[k]|')
    plt.title('FFT - Espectro de Amplitude')
    plt.grid(True)
    plt.show()

def erro(k, original, reconstruido):

    erro = original - reconstruido

    plt.subplot(1,1,1)
    plt.plot(k, erro, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
    plt.grid(True)
    plt.show()

Fs = 512.0    # Hz - frequência de amostragem
fp = 120.0    # Hz - frequência de passagem
fs_ = 200.0   # Hz - frequência de rejeição
Rp = 1.0      # dB - ripple na banda passante
Rs = 43.0     # dB - atenuação mínima na rejeição


# Freq. digitais 
omega_p = 2 * np.pi * fp / Fs   # rad
omega_s = 2 * np.pi * fs_ / Fs  # rad
print(f"ωp (digital) = {omega_p:.4f} rad")
print(f"ωs (digital) = {omega_s:.4f} rad")

# Freq. analógicas
Omega_p = 2 * Fs * np.tan(omega_p / 2)   # rad/s
Omega_s = 2 * Fs * np.tan(omega_s / 2)   # rad/s
print(f"Ωp (analógica) = {Omega_p:.2f} rad/s")
print(f"Ωs (analógica) = {Omega_s:.2f} rad/s")

# Acha o ordem e frequencia de corte do filtro
N, Omega_c = signal.buttord(Omega_p, Omega_s, Rp, Rs, analog=True)
print(f"Ordem do Butterworth N = {N}")
print(f"Ωc (freq. de corte analógica) = {Omega_c:.2f} rad/s")

omega_c = 2 * np.arctan(Omega_c / (2 * Fs))   # rad (digital)
fc = omega_c * Fs / (2 * np.pi)              # Hz
print(f"ωc (digital) = {omega_c:.4f} rad")
print(f"fc (corte em Hz) ≈ {fc:.2f} Hz")

Wn = omega_c / np.pi

# Coeficientes do filtro IIR passa-baixas
b, a = signal.butter(N, Wn, btype='low', analog=False)
print("\nCoeficientes do numerador (b):")
print(b)
print("Coeficientes do denominador (a):")
print(a)

#Pega os sinais
matriz = np.loadtxt("Sinal1_Original.csv", delimiter=",")
sinal_original = matriz.flatten()               #Pega o sinal original
matriz1 = np.loadtxt("Sinal1_Ruido.csv", delimiter=",")
sinal_ruido = matriz1.flatten()                 #Pega o sinal com ruido

# Filtragem 
sinal_filtrado_but = signal.filtfilt(b, a, sinal_ruido)

# Descobre o tamanho mínimo
N = min(len(sinal_original), len(sinal_filtrado_but))

# Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N]
sinal_filtrado_but = sinal_filtrado_but[:N]
n = np.arange(len(sinal_original))

#pega a fft dos sinais
sinal_original_freq = np.fft.fft(sinal_original)
sinal_ruido_freq = np.fft.fft(sinal_ruido)
sinal_filtrado_but_freq = np.fft.fft(sinal_filtrado_but)

mse_tempo(sinal_original, sinal_filtrado_but)
plota(sinal_original_freq, n, sinal_original, Fs)
plota(sinal_ruido_freq, n, sinal_ruido, Fs)
plota(sinal_filtrado_but_freq, n, sinal_filtrado_but, Fs)
erro(n, sinal_original, sinal_filtrado_but)

sinal_hamming = filtro_hamming(sinal_ruido, 512, fc, 64)
mse_tempo(sinal_original, sinal_hamming)
sinal_hamming_freq = np.fft.fft(sinal_hamming)
plota(sinal_hamming_freq, n, sinal_hamming, Fs)
erro(n, sinal_original, sinal_hamming)
