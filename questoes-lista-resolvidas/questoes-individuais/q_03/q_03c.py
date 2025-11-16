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

# r_xy:
r_xy = np.correlate(y, xN, 'full')

# espaço amostral resultante varia de [-(N-1), (N-1)]
n_r = np.arange(-N+1, N)

D_estimado = n_r[np.argmax(r_xy)]
print(f'D estimado: {D_estimado}')

# ==================================================
# Plotagem:
fig = plt.figure(figsize=(15,6))
gs = fig.add_gridspec(1, 1)

# plt.suptitle(f'Filtro de Média Móvel (M={M})', fontsize=16, fontweight='bold')

ax1 = fig.add_subplot(gs[0])
ax1.stem(n_r, r_xy)
ax1.set_title('$r_{xy}[l]$')
ax1.set_ylabel('$r_{xy}[l]$')
ax1.grid(True)

plt.tight_layout()
plt.show()
