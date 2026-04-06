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

def calcular_compressao_zrl(bits_orig, bits_codificado):
    # Calcula razão e ganho
    razao = bits_codificado / bits_orig
    ganho_percent = (1 - razao) * 100

    # 3️⃣ Exibe os resultados formatados
    print("📊 RESULTADO DA COMPRESSÃO ZRL")
    print("------------------------------------")
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

def soma_overlap(frames, hop):
    N = len(frames[0])
    num_frames = len(frames)
    out_len = hop * (num_frames - 1) + N

    y = np.zeros(out_len)

    for k, fr in enumerate(frames):
        start = k * hop
        y[start:start+N] += fr  #Soma as amostras do sinal reconstruido com 20% de overlay

    return y

#Pega as amostras do audio
audio, fs = sf.read("bom dia.wav")
audio = estereo_para_mono(audio)

N = 2048           # tamanho dos blocos
overlap = 0.60     # 60% de sobreposição
hop = int(N * (1 - overlap))  # passo entre blocos

frames = []
for i in range(0, len(audio) - N, hop):     #Divide o audio em blocos
    frame = audio[i:i+N]                    #Cada bloco fica guardado em frame
    frames.append(frame)                    #Todos os blocos ficam guardados em frames
frames = np.array(frames)

#Faz o janelamento em todos os blocos
w = np.hanning(N)
frames_janelados = frames * w[None, :]

#Faz a dct em todos os blocos
dcts = dct(frames_janelados, type=2, norm='ortho', axis=1)

# saídas
X_int8_frames = []   # lista de vetores int8 na qual os blocos sao guardados
escalas = []         # escala por bloco (float), necessária na desquantização

limiar = 2      #Define um limiar: valores menores em módulo que ele serão zerados

#Converte para 8bits
for k in range(dcts.shape[0]):
    X = dcts[k]  # DCT do bloco k 

    # Escala por bloco para caber em int8 (-128..127)
    maxabs = np.max(np.abs(X))
    escala = 127.0 / (maxabs)

    # Quantização para int8
    X_int8 = np.round(X * escala).astype(np.int8)

    # Zera coeficientes pequenos em int8
    X_int8[np.abs(X_int8) < limiar] = 0

    # guarda as escalas utilizadas na quantização para int8 e os blocos
    X_int8_frames.append(X_int8)
    escalas.append(escala)

codificados = []           # lista com o resultado do ZRL de cada bloco
bits_orig_total = 0        # soma de bits dos blocos antes do ZRL
bits_cod_total = 0         # soma de bits dos blocos depois do ZRL

for k in range(len(X_int8_frames)):       # percorre cada bloco
    X_int8 = X_int8_frames[k]

    # Aplica o ZRL
    enc = ZRL(X_int8.tolist())

    # Guarda o bloco codificado
    codificados.append(enc)

    # Soma os tamanhos em bits
    bits_orig_total += len(X_int8) * 8    # antes do ZRL
    bits_cod_total  += len(enc) * 8       # depois do ZRL

#Calcula a compressao
calcular_compressao_zrl(bits_orig_total, bits_cod_total)

#Reconstroi o sinal que estava comprimido aplicando o inverso da ZRL
X_rec_frames = []
for enc in codificados:
    X_rec = np.array(ZRL_decode(enc), dtype=np.int8)

    # garante tamanho N para todos os blocos apois aplicar o inverso da ZRL
    if len(X_rec) < N:
        X_rec = np.pad(X_rec, (0, N - len(X_rec)))

    X_rec_frames.append(X_rec)

frames_rec = []   # lista para armazenar os blocos reconstruídos no tempo

for k, X_rec in enumerate(X_rec_frames):  
    # Desquantiza: volta ao formato float
    X_hat = X_rec.astype(float) / escalas[k]
    # IDCT: volta ao domínio do tempo
    xw_rec = idct(X_hat, norm='ortho')
    
    frames_rec.append(xw_rec)       

sinal_reconstruido = soma_overlap(frames_rec, hop)

# Descobre o tamanho mínimo
M = min(len(audio), len(sinal_reconstruido))

# Corta os dois para o mesmo tamanho
audio = audio[:M]
sinal_reconstruido = sinal_reconstruido[:M]
n = np.arange(len(sinal_reconstruido))

mse_tempo(audio, sinal_reconstruido)
plota(n, audio)
plota(n, sinal_reconstruido)
erro(n, audio, sinal_reconstruido)