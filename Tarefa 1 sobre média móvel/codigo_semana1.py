import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 1. Carregar dados
# ==========================
arquivo = "Download Data - STOCK_BR_BVMF_PETR3.csv"
df = pd.read_csv(arquivo)

# Ajustar nomes de colunas
df.columns = [c.lower() for c in df.columns]

# Procurar coluna de fechamento
if "close" in df.columns:
    preco_col = "close"
else:
    raise ValueError("Não encontrei coluna de fechamento no CSV.")

# Converter datas para datetime
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    raise ValueError("Não encontrei coluna de datas no CSV.")

# Ordenar por data
df = df.sort_values("date").reset_index(drop=True)

# ==========================
# 2. Calcular média móvel de 4 dias
# ==========================
df["mm4"] = df[preco_col].rolling(window=4).mean()

# ==========================
# 3. Estratégia 1 - 3 quedas / 3 altas
# ==========================
def estrategia1(df, preco_col):
    saldo = 1000.0
    acoes = 0.0
    portfolio = []  # evolução do portfólio

    for i in range(3, len(df)):
        p0, p1, p2, p3 = df[preco_col][i], df[preco_col][i-1], df[preco_col][i-2], df[preco_col][i-3]

        # 3 quedas seguidas -> compra
        if p0 < p1 and p1 < p2 and p2 < p3:
            if saldo > 0:
                acoes = saldo / p0
                saldo = 0
        # 3 altas seguidas -> venda
        elif p0 > p1 and p1 > p2 and p2 > p3:
            if acoes > 0:
                saldo = acoes * p0
                acoes = 0

        # Registrar valor do portfólio
        portfolio.append(saldo + acoes * p0)

    return saldo + acoes * df[preco_col].iloc[-1], portfolio

# ==========================
# 4. Estratégia 2 - Cruzamento preço x MM4
# ==========================
def estrategia2(df, preco_col):
    saldo = 1000.0
    acoes = 0.0
    portfolio = []

    for i in range(1, len(df)):
        preco_hoje = df[preco_col][i]
        preco_ontem = df[preco_col][i-1]
        mm4_hoje = df["mm4"][i]
        mm4_ontem = df["mm4"][i-1]

        if pd.isna(mm4_hoje) or pd.isna(mm4_ontem):
            continue

        # Cruzou para cima -> compra
        if preco_hoje > mm4_hoje and preco_ontem <= mm4_ontem:
            if saldo > 0:
                acoes = saldo / preco_hoje
                saldo = 0
        # Cruzou para baixo -> venda
        elif preco_hoje < mm4_hoje and preco_ontem >= mm4_ontem:
            if acoes > 0:
                saldo = acoes * preco_hoje
                acoes = 0

        portfolio.append(saldo + acoes * preco_hoje)

    return saldo + acoes * df[preco_col].iloc[-1], portfolio

# ==========================
# 5. Benchmark (compra única)
# ==========================
def compra_unica(df, preco_col):
    preco_inicio = df[preco_col].iloc[0]
    preco_fim = df[preco_col].iloc[-1]
    return 1000 / preco_inicio * preco_fim

# ==========================
# 6. Rodar simulações
# ==========================
res1, port1 = estrategia1(df, preco_col)
res2, port2 = estrategia2(df, preco_col)
resB = compra_unica(df, preco_col)

print("Resultado Estratégia 1:", round(res1, 2))
print("Resultado Estratégia 2:", round(res2, 2))
print("Resultado Compra Única:", round(resB, 2))

# ==========================
# 7. Gráfico de preço + MM4
# ==========================
plt.figure(figsize=(12,6))
plt.plot(df["date"], df[preco_col], label="Preço")
plt.plot(df["date"], df["mm4"], label="Média Móvel 4d")
plt.title("Preço da PETR3 e MM4")
plt.xlabel("Data")
plt.ylabel("Preço (R$)")
plt.legend()
plt.grid()
plt.show()

# ==========================
# 8. Gráficos da evolução do portfólio
# ==========================
plt.figure(figsize=(12,6))
plt.plot(df["date"].iloc[-len(port1):], port1, label="Estratégia 1")
plt.plot(df["date"].iloc[-len(port2):], port2, label="Estratégia 2")
plt.axhline(resB, color="gray", linestyle="--", label="Compra Única (final)")
plt.title("Evolução do Portfólio - Estratégias")
plt.xlabel("Data")
plt.ylabel("Valor da carteira (R$)")
plt.legend()
plt.grid()
plt.show()
