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

    # => Sinal x[n-D]:
    x_desl_D = np.zeros(N)
    x_desl_D[D:(D+len(x))] = x

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

# ==================================================
# Plotagem:
fig = plt.figure(figsize=(15,6))
gs = fig.add_gridspec(3, 2)

variancias = [0.1, 1.0]
plt.suptitle(f'Saída $y[n] = αx[n-D] + w[n]$ $(σ^2={variancias[0]}$ e ${variancias[1]})$', fontsize=16, fontweight='bold')

for i in range(len(variancias)):
    xD, y = gera_saida(x, variancias[i], N)

    # sinal x[n]:
    ax1 = fig.add_subplot(gs[0, i])
    ax1.stem(n, xN, markerfmt='')
    ax1.set_title('Sinal $x[n]$')
    ax1.set_ylabel('$x[n]$')
    ax1.grid(True)

    # sinal y[n]:
    ax2 = fig.add_subplot(gs[1, i])
    ax2.stem(n, y, markerfmt='')
    ax2.set_title('Sinal $y[n] = α x[n-D] + [n]$')
    ax2.set_ylabel('$y[n]$')
    ax2.grid(True)

    # ==================================================
    # Cálculo da correlação cruzada r_xy:

    # r_xy:
    r_xy = np.correlate(y, xN, 'full')
    n_r = np.arange(-N+1, N)

    # D estimado:
    D_estimado = n_r[np.argmax(r_xy)]
    print(f'D estimado [σ^2 = {variancias[i]}]: {D_estimado}')

    ax3 = fig.add_subplot(gs[2, i])
    ax3.stem(n_r, r_xy, markerfmt='')
    ax3.set_title('$r_{xy}[l]$')
    ax3.set_ylabel('$r_{xy}[l]$')
    ax3.grid(True)


plt.tight_layout()
plt.show()