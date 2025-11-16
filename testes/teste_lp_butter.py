import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

# ============================================================
# FILTRO PROTÓTIPO: Butterworth
def prop_lp(K):
    # Freq. Corte:
    Omega_c = 1     # 1 rad/s

    # Polos do Filtro:
    i = np.arange(1, K+1)       # [1, 2, ..., K]
    polos = 1j * Omega_c * np.exp(1j * np.pi * (2*i - 1) / (2*K))

    # Gerando o polinômio do denominador:
    den = np.poly(polos)

    # Mantendo só a parte real: dos coeficientes:
    denominador = np.real(den)

    # Gerando numerador:
    numerador = np.atleast_1d(denominador[-1])

    return numerador, denominador

def lp2lp(b, a, omega_c):
    # novo numerador:
    M = len(b)
    new_b = np.array([b[i] * (1/omega_c)**(M-i-1) for i in range(M)])
    # novo denominador:
    N = len(a)
    new_a = np.array([a[i] * (1/omega_c)**(N-i-1) for i in range(N)])

    # ajusta o ganhos dos coeficientes:
    new_b = new_b / new_a[0]
    new_a = new_a / new_a[0]

    return new_b, new_a

# ============================================================
# Ordem do Filtro:
K = 4

num_filter, den_filter = prop_lp(K)
print(num_filter, den_filter)
num_filter2, den_filter2 = lp2lp(num_filter, den_filter, 20)
print(num_filter2, den_filter2)

# ============================================================
# Resposta em Frequência do Filtro:
Ts = 1e-3    # Período de amostragem
w = np.arange(0, 3000, Ts)  # Eixo de frequências (rad/s)

# s = jw:
# denominator2 = np.polyval(denominator, 1j * w)
H = num_filter2 / np.polyval(den_filter2, 1j * w)

plt.figure()
plt.plot(w, 20*np.log10(np.abs(H)))
plt.grid()
plt.tight_layout()
plt.show()
