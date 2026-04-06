import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

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

def add_tonal_noise(x, fs, f0, snr_db):
    N = len(x)
    t = np.arange(N) / fs

    # gera tom de interferência
    tone = np.sin(2 * np.pi * f0 * t)

    # potência do sinal
    Ps = np.mean(x**2)

    # SNR desejado
    snr_lin = 10**(snr_db/10)

    # potência do ruído
    Pn = Ps / snr_lin

    # normaliza o tom para ter a potência desejada
    tone = tone / np.sqrt(np.mean(tone**2)) * np.sqrt(Pn)

    # retorna o sinal ruidoso e o ruído
    return x + tone

def add_white_noise(x, snr_db):
    # potência do sinal
    Ps = np.mean(x**2)

    # SNR desejado (linear)
    snr_lin = 10**(snr_db/10)

    # potência do ruído necessária
    Pn = Ps / snr_lin

    # gera ruído branco gaussiano
    noise = np.random.randn(len(x))

    # normaliza o ruído para ter a potência correta
    noise = noise / np.sqrt(np.mean(noise**2)) * np.sqrt(Pn)

    # adiciona ao sinal
    y = x + noise

    return y

#Pega as amostras do audio
sinal_original, fs = sf.read("violino_flauta.wav")

#Adiciona ruido branco
sinal_ruido1 = add_white_noise(sinal_original, 30)

#Adiona ruido em alguma frequencia fixa
sinal_ruido2 = add_tonal_noise(sinal_original, fs, 10000, 30)

#Etapa de filtragem
sinal_filtrado1 = wiener(sinal_ruido1, mysize=31)
sinal_filtrado2 = wiener(sinal_ruido2, mysize=31)

# Descobre o tamanho mínimo
N = min(len(sinal_original), len(sinal_filtrado1))

# Corta os dois para o mesmo tamanho
sinal_original = sinal_original[:N]
sinal_ruido1 = sinal_ruido1[:N]
sinal_ruido2 = sinal_ruido2[:N]
sinal_filtrado1 = sinal_filtrado1[:N]
sinal_filtrado2 = sinal_filtrado2[:N]
n = np.arange(len(sinal_original))

#Pega o sinais no dominio da frequencia
sinal_original_freq = np.fft.fft(sinal_original)
sinal_ruido1_freq = np.fft.fft(sinal_ruido1)
sinal_ruido2_freq = np.fft.fft(sinal_ruido2)
sinal_filtrado1_freq = np.fft.fft(sinal_filtrado1)
sinal_filtrado2_freq = np.fft.fft(sinal_filtrado2)

#plota o sinal original, com ruido e os filtrados
plota(sinal_original_freq, n, sinal_original, fs)
plota(sinal_ruido1_freq, n, sinal_ruido1, fs)
plota(sinal_filtrado1_freq, n, sinal_filtrado1, fs)
plota(sinal_ruido2_freq, n, sinal_ruido2, fs)
plota(sinal_filtrado2_freq, n, sinal_filtrado2, fs)

#calcula o erro quadratico medio para as filtragens
print("Erro para a filtragem com ruido branco: ")
mse_tempo(sinal_original, sinal_filtrado1)
print("\nErro para a filtragem com ruido fixo em 10khz: ")
mse_tempo(sinal_original, sinal_filtrado2)

#salva os audios filtrados em .wav
sf.write("ruido1.wav", sinal_ruido1, fs)
sf.write("violino_flauta_filtrada1.wav", sinal_filtrado1, fs)
sf.write("ruido2.wav", sinal_ruido2, fs)
sf.write("violino_flauta_filtrada2.wav", sinal_filtrado2, fs)