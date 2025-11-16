import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

# ============================================================
# FILTRO PROTÓTIPO: Chebyshev

K = 3

# Freq. do final da Banda de Passagem:
Omega_p = 10     # 1 rad/s

# Banda de Passagem:
alpha_p = 2

# 
epsilon = np.sqrt(10**(alpha_p/10) - 1)

# Polos do Filtro:
i = np.arange(1, K+1)       # [1, 2, ..., K]

polos = -Omega_p * np.sin(np.pi * (2*i-1) / (2*K)) * np.sinh(np.arcsinh(1/epsilon)/K) + 1j*Omega_p * np.cos(np.pi*(2*i-1)/(2*K)) * np.cosh(np.arcsinh(1/epsilon)/K)

# Gerando o polinômio do denominador:
den = np.poly(polos)

# Mantendo só a parte real: dos coeficientes:
denominator = np.real(den)

Kn = 1 / np.sqrt(1 + epsilon**2) if K%2 == 0 else 1
numerador = Kn * np.real(np.prod(-polos))

# ============================================================
# Resposta em Frequência do Filtro:
Ts = 1e-3    # Período de amostragem
w = np.arange(0, 30, Ts)  # Eixo de frequências (rad/s)

# s = jw:
H = denominator[-1] / np.polyval(denominator, 1j * w)

plt.figure()
plt.plot(w, 20*np.log10(np.abs(H)))
plt.grid()
plt.tight_layout()
plt.show()
