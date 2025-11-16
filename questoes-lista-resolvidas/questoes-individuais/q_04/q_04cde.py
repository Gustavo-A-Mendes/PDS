import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# QUESTAO 04 - (c)

# Parâmetros do sistema:
A = 1
phi = 0
Fs = 100000 # Fs uma Ordem an
Ts = 1/Fs
f0 = 1000

# Parâmetros da simulação:
T = 0.01
t = np.arange(0, T, Ts)

N = len(t)


w0 = 2*np.pi*(f0)*Ts

# ==================================================
xn = 1*(t == 0)  # impulso unitário

yn2 = yn1 = 0
xn1 = 0

yn = np.zeros(N)
for n in np.arange(0, N):
    yn[n] = A*np.sin(phi) * xn[n] + A*np.sin(w0 - phi) * xn1 + 2*np.cos(w0)*yn1 - yn2

    # Atualizando parâmetros:
    yn2 = yn1
    yn1 = yn[n]
    xn1 = xn[n]

# Definindo h[n]:
h = A*np.sin(w0*t + phi)

# Plotagem:
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3, 1)

ax1 = fig.add_subplot(gs[0])
ax1.plot(t, yn)
# ax1.plot(t, h)
ax1.set_title('Resposta ao Impulso $h[n]$')
ax1.set_ylabel('$h[n]$')
ax1.grid(True)

# ==================================================
# QUESTAO 04 - (d)

def comparador(x, valor, N):
    quadrada = np.zeros(N)
    for i in np.arange(0, N):
        if x[i] >= valor:
            quadrada[i] = 1
        else:
            quadrada[i] = -1

    return quadrada


# Definindo entrada ( sinal gerado na (c) )
signal_in = yn

# Gerando onda quadrada com comparador:
sq_wave = comparador(signal_in, 0, len(signal_in))

# Plotagem:
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, sq_wave)
ax2.set_title('Sinal Onda Quadrada')
ax2.set_ylabel('$sq_{wave}[n]$')
ax2.grid(True)

# ==================================================
# QUESTAO 04 - (e)

def integrador(x, N):
    # Parâmetros iniciais:
    yn = 0
    yn1 = 0
    xn1 = 0

    # vetor da onda dente de serra:
    trig = np.zeros(N)
    for i in np.arange(0, N):
        yn = yn1  + (x[i])

        # Atualizando parâmetros:
        yn1 = yn
        xn1 = x[i]

        trig[i] = yn

    return trig


# Definindo entrada ( onda quadrada gerada na (d) )
x_sq = sq_wave

# Gerando onda dente de serra com integrador:
y_trig = integrador(x_sq, len(x_sq))

# Plotagem:
ax3 = fig.add_subplot(gs[2])
ax3.plot(t, y_trig)
ax3.set_title('Onda Dente de Serra $y_{trig}[n]$')
ax3.set_ylabel('$y_{trig}[n]$')
ax3.grid(True)

plt.tight_layout()
plt.show()