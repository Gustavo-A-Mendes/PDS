import numpy as np
import matplotlib.pyplot as plt

# => Sinal r[n] e valor de "a", "b" e "N" declarados no código anterior:
# Parâmetros de Simulação:
N = 50
n = np.arange(0, N)

# Parâmetros de Cálculo:
a = 1/2
b = 1/4

# sinal aleatório x[n]:
x = np.random.rand(N)

# ==================================================
# GERANDO SAÍDA DO CANAL DE COMUNICAÇÃO

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
# GERANDO SAÍDA DO EQUALIZADOR

# Sinal da equalização x_tilde[n] = r[n] - ax[n-1] - bx[n-2];

# Condições iniciais:
xn1 = 0
xn2 = 0
xn = 0

x_tilde = np.zeros(N)
for i in np.arange(0, N):
    xn = r[i] - a*xn1 - b*xn2
    x_tilde[i] = xn

    # Atualizando Parâmetros:
    xn2 = xn1
    xn1 = xn

# ==================================================
# Cálculo do erro absoluto da equalização:
e_eq = np.abs(x - x_tilde)

# ==================================================
# Cálculo do MSE após a equalização:
mse2 = np.mean(e_eq**2)

print(f"MSE após a equalização: {mse2:.6e}")
# ==================================================
# Plotagem:
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3,1)

ax1 = fig.add_subplot(gs[0])
ax1.plot(n, x)
ax1.set_title('Sinal Original $x[n]$')
ax1.set_ylabel('$x[n]$')

ax2 = fig.add_subplot(gs[1])
ax2.plot(n, x_tilde)
ax2.set_title('Sequência Equalizada $\\~{x}[n]$')
ax2.set_ylabel('$\\~{x}[n]$')

ax3 = fig.add_subplot(gs[2])
ax3.plot(n, e_eq)
ax3.set_title(f'Sinal de Erro Absoluto da Equalização $e_eq[n]$ $(MSE = {mse2:.4e}$)')
ax3.set_ylabel('$e_{eq}[n]$')

plt.tight_layout()
plt.show()
