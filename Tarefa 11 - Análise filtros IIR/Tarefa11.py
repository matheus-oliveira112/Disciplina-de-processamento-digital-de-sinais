import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time 

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

def plota_funcao_tranferencia(w, h, fs):
    f = w * fs / (2 * np.pi)

    # 1) Magnitude em dB
    plt.figure(figsize=(10,5))
    plt.plot(f, 20*np.log10(abs(h)))
    plt.title("Resposta em Frequência")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()

    # 2) Fase
    plt.figure(figsize=(10,5))
    plt.plot(f, np.angle(h))
    plt.title("Fase do Filtro")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Fase (rad)")
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

Fs = 125.0    # Hz - frequência de amostragem
fp = 1.0    # Hz - frequência de passagem
fs_ = 2.0   # Hz - frequência de rejeição
Rp = 1.0      # dB - ripple na banda passante
Rs = 30.0     # dB - atenuação mínima na rejeição

#=======FILTRO BUTTERWORTH========
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
sos = signal.butter(N, Wn, btype='low', output='sos')

# Coeficientes do filtro IIR passa-baixas
b, a = signal.butter(N, Wn, btype='low', analog=False)
print("\nCoeficientes do numerador (b):")
print(b)
print("Coeficientes do denominador (a):")
print(a)

#Pega os sinais
matriz = np.loadtxt("resp.csv", delimiter=",")
sinal_original = matriz.flatten()               #Pega o sinal original
matriz1 = np.loadtxt("resp_ruido.csv", delimiter=",")
sinal_ruido = matriz1.flatten()                 #Pega o sinal com ruido

# Descobre o tamanho mínimo
N = min(len(sinal_original), len(sinal_ruido))

# Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N]
sinal_ruido = sinal_ruido[:N]
n = np.arange(len(sinal_original))

# Filtragem 
inicio = time.perf_counter()
sinal_filtrado_but = signal.sosfiltfilt(sos, sinal_original)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")

#pega a fft dos sinais
sinal_original_freq = np.fft.fft(sinal_original)
sinal_ruido_freq = np.fft.fft(sinal_ruido)
sinal_filtrado_but_freq = np.fft.fft(sinal_filtrado_but)

#PLOTS
mse_tempo(sinal_original, sinal_filtrado_but)
plota(sinal_original_freq, n, sinal_original, Fs)
plota(sinal_ruido_freq, n, sinal_ruido, Fs)
plota(sinal_filtrado_but_freq, n, sinal_filtrado_but, Fs)
erro(n, sinal_original, sinal_filtrado_but)

#Função de transferencia do filtro
w, h = signal.freqz(b, a)
plota_funcao_tranferencia(w, h, Fs)

#=======FILTRO CHEBYSHEV========
Wp = fp / (Fs/2)    # borda de passante normalizada
Ws = fs_ / (Fs/2)   # borda de rejeição normalizada
N1, Wn1 = signal.cheb1ord(Wp, Ws, Rp, Rs)
print("Ordem Chebyshev I:", N1)
print("Frequência crítica normalizada:", Wn1)

#Coeficientes do filtro
b_cheb1, a_cheb1 = signal.cheby1(N1, Rp, Wn1, btype='low')
print("b_cheb1:", b_cheb1)
print("a_cheb1:", a_cheb1)

#Filtragem
inicio = time.perf_counter()
sinal_filt_cheb1 = signal.filtfilt(b_cheb1, a_cheb1, sinal_ruido)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")

#PLOTS
sinal_filt_cheb1_freq = np.fft.fft(sinal_filt_cheb1)
mse_tempo(sinal_original, sinal_filt_cheb1)
plota(sinal_filt_cheb1_freq, n, sinal_filt_cheb1, Fs)
erro(n, sinal_original, sinal_filt_cheb1)

#Função de transferencia 
w1, h1 = signal.freqz(b_cheb1, a_cheb1)
plota_funcao_tranferencia(w1, h1, Fs)

#=======FILTRO CHEBYSHEV INVERSO========
N2, Wn2 = signal.cheb2ord(Wp, Ws, Rp, Rs)
print("Ordem do Chebyshev II:", N2)
print("Frequência crítica normalizada:", Wn2)

#Coeficientes do filtro
b_cheb2, a_cheb2 = signal.cheby2(N2, Rs, Wn2, btype='low')
print("b_cheb2:", b_cheb2)
print("a_cheb2:", a_cheb2)

#Filtragem
inicio = time.perf_counter()
sinal_filt_cheb2 = signal.filtfilt(b_cheb2, a_cheb2, sinal_ruido)
fim = time.perf_counter()
print(f"Tempo de execução: {fim - inicio:.6f} segundos")

#PLOTS
sinal_filt_cheb2_freq = np.fft.fft(sinal_filt_cheb2)
mse_tempo(sinal_original, sinal_filt_cheb2)
plota(sinal_filt_cheb2_freq, n, sinal_filt_cheb2, Fs)
erro(n, sinal_original, sinal_filt_cheb2)

#Função de transferencia 
w2, h2 = signal.freqz(b_cheb2, a_cheb2)
plota_funcao_tranferencia(w2, h2, Fs)
