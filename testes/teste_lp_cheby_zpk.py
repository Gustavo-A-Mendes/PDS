import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FILTRO PROTÓTIPO: Chebyshev
def ceil_decimal(valor, casas_decimais):
    fator = 10**casas_decimais
    return np.ceil(valor*fator) / fator


def calcula_valores_filtro(Omega_p, Omega_s, alpha_p, alpha_s, type="lowpass"):
    
    Omega_p = np.atleast_1d(Omega_p)
    Omega_s = np.atleast_1d(Omega_s)
    
    # Verifica se as bandas de rejeição tem possui a mesma quantidade de valor:
    if Omega_p.shape != Omega_s.shape:
        raise ValueError("As bandas de passagem e rejeição precisam ter a mesma quantidade de valores, 1 ou 2.")

    if Omega_p.shape[0] == 1:
        if type == "lowpass":
            Omega_prot_p = Omega_p
            Omega_prot_s = Omega_s
        
        elif type == "highpass":
            Omega_prot_p = 1
            Omega_prot_s = Omega_p / Omega_s

    elif Omega_p.shape[0] == 2:
        prod_wp = Omega_p[0]*Omega_p[1]
        sub_wp = Omega_p[1] - Omega_p[0]

        if type == "bandpass":
            Omega_prot_p = 1
            Omega_prot_s = min(((prod_wp - Omega_s[0]**2)/(Omega_s[0]*sub_wp)),((Omega_s[1]**2 - prod_wp)/(Omega_s[1]*sub_wp)))

        elif type == "bandstop":
            Omega_prot_p = 1
            Omega_prot_s = min(((Omega_s[0]*sub_wp)/(prod_wp - Omega_s[0]**2)),((Omega_s[1]*sub_wp)/(Omega_s[1]**2 - prod_wp)))
        
        else:
            raise ValueError("Não foi possível detectar o tipo de filtro")

    else:
        raise ValueError("É preciso que as bandas de passagem e rejeição possuam 1 ou 2 valores")


    # Cálculo da Ordem Mínima do Filtro:
    K = np.arccosh(np.sqrt((10**(alpha_s/10) - 1) / (10**(alpha_p/10) - 1))) / (np.arccosh(Omega_prot_s / Omega_prot_p))
    K = np.ceil(K)

    # Cálculo do parâmetro de controle de ondulação da banda:
    epsilon = np.sqrt(10**(alpha_p/10)-1)

    return K, epsilon


def prop_lp(K, epsilon):
    # Freq. do final da Banda de Passagem:
    Omega_p = 1     # 1 rad/s

    # Polos do Filtro:
    i = np.arange(1, K+1)       # [1, 2, ..., K]

    polos = -Omega_p * np.sin(np.pi * (2*i-1) / (2*K)) * np.sinh(np.arcsinh(1/epsilon)/K) + 1j*Omega_p * np.cos(np.pi*(2*i-1)/(2*K)) * np.cosh(np.arcsinh(1/epsilon)/K)

    # Zeros do Filtro:
    zeros = np.array([])

    # Ganho do Filtro:
    K_0 = np.prod(-polos) / np.sqrt(1 + epsilon**2) if K%2 == 0 else np.real(np.prod(-polos))

    return zeros, polos, K_0


def lp2lp_zpk(z, p, k, Omega_p):
    
    Omega_p = np.atleast_1d(Omega_p)

    # Ordem do polos e zeros:
    N = len(p)
    M = len(z)

    # novos polos:
    polos_lp = Omega_p * p
    
    # novos zeros:
    zeros_lp = Omega_p * z

    # novo ganho:
    ganho_lp = k * Omega_p**(N)

    return zeros_lp, polos_lp, ganho_lp


def lp2hp_zpk(z, p, k, Omega_p):
    
    # Ordem do polos e zeros:
    N = len(p)
    M = len(z)

    # novos polos:
    polos_hp = Omega_p / p
    
    # novos zeros:
    zeros_hp = Omega_p / z
    zeros_hp = np.concatenate((zeros_hp, np.zeros(N)))

    # novo ganho:
    ganho_hp = k

    return zeros_hp, polos_hp, ganho_hp


def lp2bp_zpk(z, p, k, Omega_p):

    # Ordem do denominador e numerador:
    N = len(p)
    M = len(z)

    # novos polos:
    P1 = p*(Omega_p[1]- Omega_p[0])/2 + np.sqrt((p*(Omega_p[1]- Omega_p[0])/2)**2- Omega_p[0]*Omega_p[1])
    P2 = p*(Omega_p[1]- Omega_p[0])/2- np.sqrt((p*(Omega_p[1]- Omega_p[0])/2)**2- Omega_p[0]*Omega_p[1])
    
    polos_bp = np.concatenate((P1, P2))

    # novos zeros:
    zeros_bp = np.zeros(int(N))

    # novo ganho:
    ganho_bp = k * ((Omega_p[1] - Omega_p[0])**N)

    return zeros_bp, polos_bp, ganho_bp


def lp2bs_zpk(z, p, k, Omega_p):
    
    # Ordem do denominador e numerador:
    N = len(p)
    M = len(z)

    # novos polos:
    P1 = (Omega_p[1]- Omega_p[0])/(2*p) + np.sqrt(((Omega_p[1]- Omega_p[0])/(2*p))**2- Omega_p[0]*Omega_p[1])
    P2 = (Omega_p[1]- Omega_p[0])/(2*p) - np.sqrt(((Omega_p[1]- Omega_p[0])/(2*p))**2- Omega_p[0]*Omega_p[1])
    
    polos_bs = np.concatenate((P1, P2))

    # novos zeros (Gerando os N zeros em +- sqrt(Omega_p1*Omega_p2) ):
    Z1 = + 1j * np.sqrt(Omega_p[0]*Omega_p[1])*np.ones(N)
    Z2 = - 1j * np.sqrt(Omega_p[0]*Omega_p[1])*np.ones(N)

    zeros_bs = np.concatenate((Z1, Z2))

    # novo ganho:
    ganho_bs = k / np.prod(-p)

    return zeros_bs, polos_bs, ganho_bs


# ============================================================
# Especificações do filtro:
Omega_p = [10]
Omega_s = [16.5]
alpha_p = 2
alpha_s = 20

#  Calcula a Ordem mínima do Filtro e a frequência de :
K, epsilon = calcula_valores_filtro(Omega_p, Omega_s, alpha_p, alpha_s, type="lowpass")
print(K, epsilon)

# Calcula filtro Low-Pass protótipo:
z_prot, p_prot, k_prot = prop_lp(K, epsilon)

den_filter = np.real(np.poly(p_prot))
num_filter = k_prot * np.real(np.poly(z_prot))
print("num e den do protótipo: ", num_filter, den_filter, k_prot)

# Calcula filtro Low-Pass, na freq. de corte:
z, p, k = lp2lp_zpk(z_prot, p_prot, k_prot, Omega_p)
# z, p, k = lp2hp_zpk(z_prot, p_prot, k_prot, Omega_p)
# z, p, k = lp2bp_zpk(z_prot, p_prot, k_prot, Omega_p)
# z, p, k = lp2bs_zpk(z_prot, p_prot, k_prot, Omega_p)

den_filter2 = np.real(np.poly(p))
num_filter2 = k * np.real(np.poly(z))
print(num_filter2, den_filter2)

# ============================================================
# Resposta em Frequência do Filtro:
Ts = 1e-3    # Período de amostragem
w = np.arange(0, 30, Ts)  # Eixo de frequências (rad/s)

# s = jw:
H = np.polyval(num_filter2, 1j * w) / np.polyval(den_filter2, 1j * w)

plt.figure()
plt.plot(w, 20*np.log10(np.abs(H)))
# plt.ylim([-10, 1])
plt.grid()
plt.tight_layout()
plt.show()
