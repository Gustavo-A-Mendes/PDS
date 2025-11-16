import numpy as np
import matplotlib.pyplot as plt

# Parâmetros de Simulação:
N = 50
n = np.arange(0, N)

# Parâmetros de Cálculo:
a = 1/2
b = 1/4

# ==================================================
# sinal aleatório x[n]:
x = np.random.rand(N)

# ==================================================
# sinal r[n] = x[n] + ax[n-1] + bx[n-2]

# Condições iniciais:
xn1 = 0
xn2 = 0
rn = 0

r = np.zeros(N)
for i in np.arange(0, N):
    rn = x[i] + a*xn1 + b*xn2
    r[i] = rn

    # Atualizando Parâmetros:
    xn2 = xn1
    xn1 = x[i]

# ==================================================
# Cálculo do Erro e[n] = |x[n] - r[n]|:
e = np.abs(x - r)

# ==================================================
# Cálculo do MSE:
mse = np.mean(e**2)


# ==================================================
# Plotagem:
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3,1)

ax1 = fig.add_subplot(gs[0])
ax1.plot(n, x)
ax1.set_title('Sinal Aleatório $x[n]$')
ax1.set_ylabel('$x[n]$')

ax2 = fig.add_subplot(gs[1])
ax2.plot(n, r)
ax2.set_title('Sinal $r[n]$')
ax2.set_ylabel('$r[n]$')

ax3 = fig.add_subplot(gs[2])
ax3.plot(n, e)
ax3.set_title(f'Sinal de Erro Absoluto $e[n]$ $(MSE = {mse:.4f}$)')
ax3.set_xlabel('n')
ax3.set_ylabel('$e[n]$')

plt.tight_layout()
plt.show()