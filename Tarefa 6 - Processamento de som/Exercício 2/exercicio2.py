import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

letraA, fs = sf.read("Matheus_letraA_0.wav")
letraE, fs = sf.read("Matheus_letraE_0.wav")
letraI, fs = sf.read("Matheus_letraI_0.wav")
letraO, fs = sf.read("Matheus_letraO_0.wav")
letraU, fs = sf.read("Matheus_letraU_0.wav")

#letraA = letraA - np.mean(letraA)

N = 2048

letraA = letraA[:N]
letraE = letraE[:N]
letraI = letraI[:N]
letraO = letraO[:N]
letraU = letraU[:N]

k = np.arange(len(letraO))

A = np.fft.fft(letraA)
E = np.fft.fft(letraE)
I = np.fft.fft(letraI)
O = np.fft.fft(letraO)
U = np.fft.fft(letraU)

def plota(sinal, k, letra, char_letra):
    amp = np.abs(sinal)

    # 1) Sinal no tempo
    plt.subplot(1,1,1)
    plt.plot(k, letra, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title(f'Letra {char_letra}')
    plt.legend()  # legenda explicando as cores
    plt.grid(True)
    plt.show()

    k = k*(44100/2048)
    # 2) Espectro de amplitude da fft
    plt.subplot(1,1,1)
    plt.stem(k, amp, linefmt='b-', markerfmt='b.', basefmt='k-')
    plt.xlabel('Hz')
    plt.ylabel('|X[k]|')
    plt.title('FFT - Espectro de Amplitude')
    plt.grid(True)
    plt.show()


plota(A, k, letraA, 'A')
plota(E, k, letraE, 'E')
plota(I, k, letraI, 'I')
plota(O, k, letraO, 'O')
plota(U, k, letraU, 'U')
