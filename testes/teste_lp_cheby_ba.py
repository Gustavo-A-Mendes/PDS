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
            Omega_prot_p = Omega_p[0]
            Omega_prot_s = Omega_s[0]
        
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

    # Gerando o polinômio do denominador:
    den = np.poly(polos)

    # Mantendo só a parte real: dos coeficientes:
    denominador = np.real(den)

    # Gerando o polinômio do numerador:
    Kn = 1 / np.sqrt(1 + epsilon**2) if K%2 == 0 else 1
    """ 
        OBS: O numpy, quando detecta que o vetor só possui 1 valor, 
        ele converte o vetor para um valor numérico.
        O np.atleast_1d() serve para forçar o numpy a criar um vetor com apenas 1 valor 
    """
    numerador = np.atleast_1d(Kn * np.real(np.prod(-polos)))

    return numerador, denominador, Kn


def lp2lp(b, a, Omega_p):

    Omega_p = np.atleast_1d(Omega_p)

    # novo numerador:
    M = len(b)
    new_b = np.array([b[i] * (1/Omega_p)**(M-i-1) for i in range(M)])
    # novo denominador:
    N = len(a)
    new_a = np.array([a[i] * (1/Omega_p)**(N-i-1) for i in range(N)])

    # ajusta o ganhos dos coeficientes:
    new_b = new_b / new_a[0]
    new_a = new_a / new_a[0]

    return new_b, new_a


def lp2hp(b, a, ganho_0, omega_p):
    
    # Extrai os polos e zeros:
    polos = np.roots(a)
    zeros = np.roots(b)
    
    # Extranho o ganho:
    ganho = ganho_0

    # Ordem do denominador e numerador:
    N = len(polos)
    M = len(zeros)

    # novos polos:
    new_polos = omega_p / polos

    # novos zeros:
    new_zeros = np.zeros(int(N))

    # novo ganho:
    new_ganho = ganho

    # novo numerador:
    new_b = new_ganho * np.poly(new_zeros)
    # novo denominador:
    new_a = np.poly(new_polos)

    # ajusta o ganhos dos coeficientes:
    new_a = new_a / new_a[0]
    new_b = new_b / new_a[0]

    return new_b, new_a


def lp2bp(b, a, ganho_0, Omega_p):

    # Extrai os polos e zeros:
    polos = np.roots(a)
    zeros = np.roots(b)
    
    # Extranho o ganho:
    ganho = ganho_0

    # Ordem do denominador e numerador:
    N = len(polos)
    M = len(zeros)

    # novos polos:
    P1 = polos*(Omega_p[1]- Omega_p[0])/2 + np.sqrt((polos*(Omega_p[1]- Omega_p[0])/2)**2- Omega_p[0]*Omega_p[1])
    P2 = polos*(Omega_p[1]- Omega_p[0])/2- np.sqrt((polos*(Omega_p[1]- Omega_p[0])/2)**2- Omega_p[0]*Omega_p[1])
    
    new_polos = np.concatenate((P1, P2))

    # novos zeros:
    new_zeros = np.zeros(int(N))

    # novo ganho:
    new_ganho = ganho

    # novo denominador:
    new_a = np.real(np.poly(new_polos))
    
    # novo numerador:
    new_b = new_ganho * np.real(np.prod(-polos)) * ((Omega_p[1] - Omega_p[0])**N) * np.real(np.poly(new_zeros))
    
    # ajusta o ganhos dos coeficientes:
    new_a = new_a / new_a[0]
    new_b = new_b / new_a[0]

    return new_b, new_a

def lp2bs(b, a, ganho_0, Omega_p):

    # Extrai os polos e zeros:
    polos = np.roots(a)
    zeros = np.roots(b)
    
    # Extranho o ganho:
    ganho = ganho_0

    # Ordem do denominador e numerador:
    N = len(polos)
    M = len(zeros)

    # novos polos:
    P1 = (Omega_p[1]- Omega_p[0])/(2*polos) + np.sqrt(((Omega_p[1]- Omega_p[0])/(2*polos))**2- Omega_p[0]*Omega_p[1])
    P2 = (Omega_p[1]- Omega_p[0])/(2*polos) - np.sqrt(((Omega_p[1]- Omega_p[0])/(2*polos))**2- Omega_p[0]*Omega_p[1])
    
    new_polos = np.concatenate((P1, P2))

    # novos zeros (Gerando os N zeros em +- sqrt(Omega_p1*Omega_p2) ):
    Z1 = + 1j * np.sqrt(Omega_p[0]*Omega_p[1])*np.ones(N)
    Z2 = - 1j * np.sqrt(Omega_p[0]*Omega_p[1])*np.ones(N)

    new_zeros = np.concatenate((Z1, Z2))

    # novo ganho:
    new_ganho = ganho

    # novo denominador:
    new_a = np.real(np.poly(new_polos))
    
    # novo numerador:
    new_b = new_ganho * np.real(np.poly(new_zeros))
    
    # ajusta o ganhos dos coeficientes:
    new_a = new_a / new_a[0]
    new_b = new_b / new_a[0]

    return new_b, new_a


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
num_filter, den_filter, ganho_n = prop_lp(K, epsilon)
print(num_filter, den_filter, ganho_n)

# Calcula filtro Low-Pass, na freq. de corte:
num_filter2, den_filter2 = lp2lp(num_filter, den_filter, Omega_p)
# num_filter2, den_filter2 = lp2hp(num_filter, den_filter, ganho_n, Omega_p)
# num_filter2, den_filter2 = lp2bp(num_filter, den_filter, ganho_n, Omega_p)
# num_filter2, den_filter2 = lp2bs(num_filter, den_filter, ganho_n, Omega_p)
print(num_filter2, den_filter2)

# ============================================================
# Resposta em Frequência do Filtro:
Ts = 1e-3    # Período de amostragem
w = np.arange(0, 30, Ts)  # Eixo de frequências (rad/s)

# s = jw:
H = np.polyval(num_filter2, 1j * w) / np.polyval(den_filter2, 1j * w)

plt.figure()
plt.plot(w, 20*np.log10(np.abs(H)))
plt.grid()
plt.tight_layout()
plt.show()
