import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

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

def estereo_para_mono(dados):
    if len(dados.shape) == 2:
        sinal = np.mean(dados, axis=1)        #Transforma um sinal levemente estereo em mono
    else:
        sinal = dados  # já é mono

    return sinal

def ZRL(x):
    out = []

    # Se o primeiro valor for diferente de zero → grava literal e começa do segundo
    if x[0] != 0:
        out.append(int(x[0]))
        i = 1
    else:
        # Caso comece com zeros → começa do início
        i = 0

    # Percorre o vetor, contando zeros e emitindo pares (valor, zeros_anteriores)
    zeros = 0
    while i < len(x):
        if x[i] == 0:
            zeros += 1                # acumula zeros
        else:
            out.extend([int(x[i]), zeros])  # grava o valor e quantos zeros vieram antes
            zeros = 0
        i += 1

    # Zeros finais não são codificados neste formato
    return out

def calcular_compressao_zrl(x, enc):
    # Cada valor (int8) ocupa 8 bits
    bits_orig = len(x) * 8
    bits_codificado = len(enc) * 8

    # Calcula razão e ganho
    razao = bits_codificado / bits_orig
    ganho_percent = (1 - razao) * 100

    # 3️⃣ Exibe os resultados formatados
    print("📊 RESULTADO DA COMPRESSÃO ZRL")
    print("------------------------------------")
    print(f"Tamanho original:     {len(x)} elementos")
    print(f"Tamanho codificado:   {len(enc)} elementos")
    print(f"Bits originais:       {bits_orig} bits")
    print(f"Bits codificados:     {bits_codificado} bits")
    print(f"Razão de compressão:  {razao:.3f}")
    print(f"Ganho de compressão:  {ganho_percent:.2f}%")
    print("------------------------------------\n")

def ZRL_decode(enc):
    out = []
    if len(enc) == 0:
        return out

    # Se o tamanho é ímpar → há literal no início
    idx = 0
    if len(enc) % 2 == 1:
        out.append(int(enc[0]))
        idx = 1

    # Decodifica pares (valor, zeros_anteriores)
    while idx + 1 < len(enc):
        val = int(enc[idx])
        z = int(enc[idx + 1])
        out.extend([0] * z)
        out.append(val)
        idx += 2

    return out

def plota(k, sinal):
    plt.subplot(1,1,1)
    plt.plot(k, sinal, marker='.', color='r')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title('Sinal no tempo')
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

audio, fs = sf.read("bom dia.wav")
audio = estereo_para_mono(audio)

#Faz a dct sobre o sinal do audio
X = dct(audio, norm='ortho')

#Converte para interiros de 8bits
escala = 127 / np.max(np.abs(X))
X_int8 = np.round(X * escala).astype(np.int8)

limiar = 2      #Define um limiar: valores menores em módulo que ele serão zerados
X_int8[np.abs(X_int8) < limiar] = 0     #Zera valores próximos de zero

#Faz o ZRL
X_comprimido = ZRL(X_int8)    

#Calcula a compressao
calcular_compressao_zrl(X_int8, X_comprimido)   

#Reconstroi o sinal que estava comprimido aplicando o inverso da ZRL
X_int_reconstruido = ZRL_decode(X_comprimido)

#Transforma o sinal que estava em int8 para float novamente
X_rec = X_int8.astype(float) / escala

#Aplica a idct
SINAL_REC = idct(X_rec, norm='ortho')

# Descobre o tamanho mínimo
N = min(len(audio), len(SINAL_REC))

# Corta os dois para o mesmo tamanho
audio = audio[:N]
SINAL_REC = SINAL_REC[:N]
n = np.arange(len(SINAL_REC))

mse_tempo(audio, SINAL_REC)
plota(n, audio)
plota(n, SINAL_REC)
erro(n, audio, SINAL_REC)
