import numpy as np
import matplotlib.pyplot as plt

# Parâmetros de simulação:
Fs = 200
Ts = 1/Fs
T = 5                # Tempo de simulação
t = np.arange(0, T, Ts) # Vetor espaço amostral

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

# ==================================================
# entrada qualquer x[n]:
# f = 20
# x = np.sin(f*np.pi*t)
x = 0.6*np.sin(2*np.pi*1.2*t) + 0.4*np.sin(2*np.pi*20*t)*(np.sin(2*np.pi*1.2*t) > 0)

y, lambda1, lambda2 = function(x)
# ==================================================
# Plotagem:
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(5, 1)

# entrada x[n]:
ax1 = fig.add_subplot(gs[0])
ax1.plot(t, x)
# ax1.set_xlim([1,2])
ax1.set_title('Sinal $x[n] = sin(0.1 \\pi t)$')
ax1.set_ylabel('$x[n]$')
ax1.grid(True)

# primeira derivada retificada y0[n]:
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, y[0, :])
ax2.set_title('Sinal $y_0[n] = |x[n+1] - x[n-1]|$')
ax2.set_ylabel('$y_0[n]$')
ax2.grid(True)

# primeira derivada suavizada y1[n]:
ax3 = fig.add_subplot(gs[2])
ax3.plot(t, y[1, :])
ax3.set_title('Sinal $y_1[n] = (y_0[n-1] + 2y_0[n] + y_0[n+1]) / 4$')
ax3.set_ylabel('$y_1[n]$')
ax3.grid(True)

# segunda derivada retificada y2[n]:
ax4 = fig.add_subplot(gs[3])
ax4.plot(t, y[2, :])
ax4.set_title('Sinal $y_2[n] = |x[n+2] - 2x[n] + x[n-2]|$')
ax4.set_ylabel('$y_2[n]$')
ax4.grid(True)

# primeira suavizada + segunda retificada y3[n]:
ax5 = fig.add_subplot(gs[4])
ax5.plot(t, y[3, :], label='$y_3[n]$')
ax5.axhline(lambda1, color='r', linestyle='--', label='λ₁')
ax5.axhline(lambda2, color='g', linestyle='--', label='λ₂')
ax5.set_title('Sinal $y_3[n] = 2(y_1 + y_2)$')
ax5.set_ylabel('$y_3[n]$')
ax5.legend()
ax5.grid(True)

plt.tight_layout()
plt.show()