import numpy as np
import matplotlib.pyplot as plt

# Parâmetros de simulação:
N = 200                 # Tempo amostrado
n = np.arange(0, N) # Vetor espaço amostral


# Declarando função da saída:
def gera_saida(x, variancia, N):
    # Parâmetros do sistema:
    alpha = 0.9
    D = 20

    # => Sinal x[n-D] (com tratamento dos índices):
    x_desl_D = np.zeros(N)
    if D+len(x) < N:
        x_desl_D[D:(D+len(x))] = x
    else:
        x_desl_D[D:] = x[:N-D]

    # => Erro Gaussiano:
    w = np.random.normal(0, np.sqrt(variancia), N)

    # => saída:
    saida = alpha*x_desl_D + w

    return x_desl_D, saida

# ==================================================

x = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]

# => Sinal x[n] com espaço amostral de 200:
xN = np.zeros(N)
xN[:len(x)] = x

xD, y = gera_saida(x, 0.01, N)

# ==================================================
# Plotagem:
fig = plt.figure(figsize=(10,6))
gs = fig.add_gridspec(3, 1)

# plt.suptitle(f'Filtro de Média Móvel (M={M})', fontsize=16, fontweight='bold')

# sinal x[n]:
ax1 = fig.add_subplot(gs[0])
ax1.stem(n, xN, markerfmt='')
ax1.set_title('Sinal $x[n]$')
ax1.set_ylabel('$x[n]$')
ax1.grid(True)

# sinal x[n-D]:
ax2 = fig.add_subplot(gs[1])
ax2.stem(n, xD, markerfmt='')
ax2.set_title('Sinal $x[n-D]$')
ax2.set_ylabel('$x[n-D]$')
ax2.grid(True)

# sinal y[n]:
ax3 = fig.add_subplot(gs[2])
ax3.stem(n, y, markerfmt='')
ax3.set_title('Sinal $y[n] = α x[n-D] + [n]$')
ax3.set_ylabel('$y[n]$')
ax3.grid(True)

plt.tight_layout()
plt.show()