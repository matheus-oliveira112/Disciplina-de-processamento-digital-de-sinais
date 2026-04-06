import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

dados, fs = sf.read("Sinal2.wav")       #Pega os dados normalizados em float
dados_ruido, fs= sf.read("Sinal2_ruido.wav")

# Descobre o tamanho mínimo
N = min(len(dados_ruido), len(dados))

# Corta os dois para o mesmo tamanho
dados_ruido = dados_ruido[:N]
dados = dados[:N]

def estereo_para_mono(dados):
    if len(dados.shape) == 2:
        sinal = np.mean(dados, axis=1)        #Transforma um sinal levemente estereo em mono
    else:
        sinal = dados  # já é mono

    return sinal

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
    # print(f"  → Índices das harmônicas mantidas: {idx_keep.tolist()}")

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

def reconstrucao_e_erro(Xk, porcentagem, sinal_original):
    sinal_rec = reconstruir_por_energia(Xk, porcentagem)
    mse_tempo(sinal_original, sinal_rec)

    return sinal_rec

def plota_graficos_sobrepostos(sinal_rec, sinal_original, n, porcentagem): 
    plt.subplot(1,1,1)
    plt.plot(n, sinal_original, marker='.', color='b', label='Sinal original')
    plt.plot(n, sinal_rec, marker='.', color='r', label='Sinal com ruido')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title(f'Sinal original vs reconstruido com {porcentagem:.1f}%')
    plt.legend()  # legenda explicando as cores
    plt.grid(True)
    plt.show()

def plota_sinal_original(sinal_original, n):
    plt.subplot(1,1,1)
    plt.plot(n, sinal_original, marker='.', linewidth=0.5, markersize=1)
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
    plt.grid(True)
    plt.show()


sinal_original = estereo_para_mono(dados)
sinal_ruido = estereo_para_mono(dados_ruido)

"""
print("Taxa de amostragem:", fs)
print("Shape:", sinal_original.shape)
print("Tipo:", sinal_original.dtype)
print("Primeiras 10 amostras:", sinal_original[:10])
"""

Xk = np.fft.fft(sinal_ruido)
sinal_rec_97 = reconstrucao_e_erro(Xk, 0.97, sinal_original)
sinal_rec_95 = reconstrucao_e_erro(Xk, 0.95, sinal_original)
sinal_rec_93 = reconstrucao_e_erro(Xk, 0.93, sinal_original)
sinal_rec_90 = reconstrucao_e_erro(Xk, 0.90, sinal_original)
sinal_rec_85 = reconstrucao_e_erro(Xk, 0.85, sinal_original)

n = np.arange(len(sinal_original))

# --------- PLOTS ---------
plota_sinal_original(sinal_original, n)
plota_graficos_sobrepostos(sinal_rec_97, sinal_original, n, 97)
plota_graficos_sobrepostos(sinal_rec_95, sinal_original, n, 95)
plota_graficos_sobrepostos(sinal_rec_93, sinal_original, n, 93)
plota_graficos_sobrepostos(sinal_rec_90, sinal_original, n, 90)
plota_graficos_sobrepostos(sinal_rec_85, sinal_original, n, 85)

sf.write('sinal_rec97.wav', sinal_rec_97, fs, subtype='PCM_16')
