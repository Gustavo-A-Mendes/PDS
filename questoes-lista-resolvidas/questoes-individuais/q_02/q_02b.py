import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==================================================
# Definindo função:
def function(x):
    N = len(x)

    # => Gerando derivadas retificadas y0 e y2:
    # Condições iniciais:
    x_minus_2 = 0
    x_minus_1 = 0
    y0n = 0
    y2n = 0

    y = np.zeros((4, N))
    for n in np.arange(0, N):

        # Pegando valores futuros:
        x_plus_1  = x[n+1] if (n+1 < N) else 0
        x_plus_2  = x[n+2] if (n+2 < N) else 0

        # Primeira derivada retificada y0[n]:
        # y0[n] = | x[n+1] - x[n-1] |
        y0n = np.abs(x_plus_1 - x_minus_1)
        y[0, n] = y0n

        # Segunda derivada retificada y2[n]:
        # y2[n] = | x[n+2] − 2x[n] + x[n−2] |:
        y2n = np.abs(x_plus_2 - 2*x[n] + x_minus_2)
        y[2, n] = y2n

        # Atualizando valores:
        x_minus_2 = x_minus_1
        x_minus_1 = x[n]

    # => Gerando derivada suavizada y1:
    # Condições iniciais:
    y_minus_1 = 0
    y1n = 0

    for n in np.arange(0, N):

        # Pegando valores futuros:
        y_plus_1  = y[0, n+1] if (n+1 < N) else 0

        # Primeira derivada suavizada y1[n]:
        # y1[n] = ( y0[n-1] + 2y0[n] + y0[n+1] ) / 4:
        y1n = (y_minus_1 + 2*y[0, n] + y_plus_1) / 4
        y[1, n] = y1n

        # Atualizando valores:
        y_minus_1 = y[0, n]

    # => Gerando sinal y3:
    # y3[n] = 2( y1 + y2 ):
    y[3, :] = 2*(y[1, :] + y[2, :])

    # => Lambda1 e lambda2:
    lambda1 = 0.5*max(y[3, :])
    lambda2 = 0.1*max(y[3, :])

    return y, lambda1, lambda2


# Função que detecta os complexos QRS:
def qrs_detect(signal, lambda1, lambda2):
    N = len(signal)
    qrs_positions = []

    # Filtrando os picos que ultrapassam o primeiro limiar:
    for id, amostra in enumerate(signal):
        if amostra < lambda1:
            continue

        # Refinando a detecção com o segundo limiar:
        """
            Regra: As próximas 6 amostras devem ser maiores
            ou iguais ao segundo limiar (lambda2).
        """
        intervalo_teste = signal[id+1:id+7]
        if np.all(intervalo_teste >= lambda2):
            qrs_positions.append((id, amostra))


    return qrs_positions


# Lê o arquivo 100_norm.csv:
df = pd.read_csv('files/100_norm.csv')

# Imprime os dados:
# print(df)

# Separa cada coluna em um array diferente:
t = df.iloc[:, 0].values[3000:5000]
x0 = df.iloc[:, 1].values[3000:5000]
x1 = df.iloc[:, 2].values[3000:5000]

y0, lambda1_0, lambda2_0 = function(x0)
y1, lambda1_1, lambda2_1 = function(x1)

# print(len(t))
qrs0 = qrs_detect(y0[3, :], lambda1_0, lambda2_0)
qrs1 = qrs_detect(y1[3, :], lambda1_1, lambda2_1)

signal_qrs_0 = np.zeros(len(t))
for pos, value in qrs0:
    signal_qrs_0[pos] = value

signal_qrs_1 = np.zeros(len(t))
for pos, value in qrs1:
    signal_qrs_1[pos] = value

# Plotagem dos resultados:
fig = plt.figure(figsize=(30, 10))
gs = fig.add_gridspec(3, 2)

# Sinal ECG 0:
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, x0)
ax1.set_title('Sinal ECG 0')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# Resultado intermediário:
print('Limiarização do Sinal ECG 0:')
print(f'Lambda 1: {lambda1_0:.4f}')
print(f'Lambda 2: {lambda2_0:.4f}')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, y0[3, :])
ax2.axhline(lambda1_0, color='r', linestyle='--', label='λ₁')
ax2.axhline(lambda2_0, color='g', linestyle='--', label='λ₂')
ax2.plot(t, signal_qrs_0, 'k')

ax2.set_title('Sinal $y_3[n]$ com Limiarização')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

# Detecção de complexos QRS no sinal ECG 0:
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(t, signal_qrs_0)
ax3.set_title('Detecção de Complexos QRS - Sinal ECG 0')
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Amplitude')
ax3.grid(True)

# Sinal ECG 1:
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t, x1)
ax1.set_title('Sinal ECG 1')
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# Resultado intermediário:
print('Limiarização do Sinal ECG 1:')
print(f'Lambda 1: {lambda1_1:.4f}')
print(f'Lambda 2: {lambda2_1:.4f}')

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t, y1[3, :])
ax2.axhline(lambda1_1, color='r', linestyle='--', label='λ₁')
ax2.axhline(lambda2_1, color='g', linestyle='--', label='λ₂')
ax2.plot(t, signal_qrs_1, 'k')

ax2.set_title('Sinal $y_3[n]$ com Limiarização')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

# Detecção de complexos QRS no sinal ECG 1:
ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(t, signal_qrs_1)
ax3.set_title('Detecção de Complexos QRS - Sinal ECG 1')
ax3.set_xlabel('Tempo (s)')
ax3.set_ylabel('Amplitude')
ax3.grid(True)

plt.tight_layout()
plt.show()