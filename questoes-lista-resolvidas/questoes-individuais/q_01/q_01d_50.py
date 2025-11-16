import numpy as np
import matplotlib.pyplot as plt

# Parâmetros de simulação:
M = 50                  # Ordem
N = 200                 # Tempo amostrado
n = np.arange(0, N) # Vetor espaço amostral

# definindo sinal x[n] = r[n] + eta[n]
r = np.sin(0.1*np.pi*n)

# Parâmetros do erro Gaussiano:
variancia = [0.01, 0.05, 0.1]
mu = 0

# vetor resultados:
eta = np.zeros((len(variancia), N))
x = np.zeros((len(variancia), N))
y = np.zeros((len(variancia), N))
e = np.zeros((len(variancia), N))
mse = np.zeros(len(variancia))

for i in range(len(variancia)):

    '''
        O ruído gaussiano é gerado a partir da função:

            np.random.normal(mu, sigma, N)

        em que mu é a média, sigma é o desvio padrão e N é o tamanho do vetor.
    '''
    eta[i] = np.random.normal(mu, np.sqrt(variancia[i]), N)

    # Definindo entrada x[n]
    x[i] = r + eta[i]

    # definindo sinal de saída:
    # defindindo y = (1/M)*sum_{k=0}^{M-1}{x[n-k]}
    y_temp = np.zeros(N+M)
    for j in range(0, M):
        y_temp[j:N+j] += (1/M)*x[i]

    # sinal de saída com espaço reduzido:
    y[i] = y_temp[:N]

    # erro absoluto:
    e[i] = np.abs(r - y[i])

    # MSE
    mse[i] = np.mean(e[i]**2)

    print(f'MSE [$σ^2 = 0.01$]: {mse[i]:.4f}')


# ==================================================
# Plotagem:
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(5, 3)

plt.suptitle(f'Filtro de Média Móvel (M={M})', fontsize=16, fontweight='bold')

# i. sinal r[n]:
ax1 = fig.add_subplot(gs[0, 1])
ax1.stem(n, r, markerfmt='')
ax1.set_title('Sinal $r[n]$', fontsize=10)
ax1.set_xlabel('n (amostras)')
ax1.set_ylabel('$r[n]$')
ax1.grid(True)

for i in range(len(variancia)):

    # ii. sinal eta[n]:
    ax2 = fig.add_subplot(gs[1, i])
    ax2.stem(n, eta[i], markerfmt='')
    ax2.set_title(f'Sinal $η[n]$ $(σ^2 = {variancia[i]})$')
    ax2.set_ylabel('$η[n]$')
    ax2.grid(True)

    # iii. sinal x[n]:
    ax3 = fig.add_subplot(gs[2, i])
    ax3.stem(n, x[i], markerfmt='')
    ax3.set_title('Sinal $x[n] = r[n] + η[n]$')
    ax3.set_ylabel('$x[n]$')
    ax3.grid(True)

    # iv. sinal y[n]:
    ax4 = fig.add_subplot(gs[3, i])
    ax4.stem(n, y[i], markerfmt='')
    ax4.set_title('Sinal $y[n] = \\frac{1}{M} \\sum^{M-1}_{k=0}{x[n-k]}$')
    ax4.set_ylabel('$y[n]$')
    ax4.grid(True)

    # v. sinal e[n]:
    ax5 = fig.add_subplot(gs[4, i])
    ax5.stem(n, e[i], markerfmt='')
    ax5.set_title(f'Sinal $e[n] = |r[n] - y[n]$ (MSE = {mse[i]:.4f})')
    ax5.set_ylabel('$e[n]$')
    ax5.grid(True)

plt.tight_layout()
plt.show()