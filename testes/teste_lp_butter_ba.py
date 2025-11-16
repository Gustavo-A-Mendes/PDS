import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FILTRO PROTÓTIPO: Butterworth
def ceil_decimal(valor, casas_decimais):
    fator = 10**casas_decimais
    return np.ceil(valor*fator) / fator


def calcula_valores_filtro(Omega_p, Omega_s, alpha_p, alpha_s, Oc_type="mean", type="lowpass"):
    
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
            Omega_prot_s = Omega_prot_s = min(((Omega_s[0]*sub_wp)/(prod_wp - Omega_s[0]**2)),((Omega_s[1]*sub_wp)/(Omega_s[1]**2 - prod_wp)))

    else:
        raise ValueError("É preciso que as bandas de passagem e rejeição possuam 1 ou 2 valores")

    # Cálculo da Ordem Mínima do Filtro:
    K = np.log10((10**(alpha_s/10) - 1) / (10**(alpha_p/10) - 1)) / (2 * np.log10(Omega_prot_s / Omega_prot_p))
    K = np.ceil(K)

    # Cálculo da Frequência de corte (média dos intervamos máximo e mínimo):
    Omega_c_min = Omega_prot_p / (10**(alpha_p/10) - 1)**(1/(2*K))
    Omega_c_max = Omega_prot_s / (10**(alpha_s/10) - 1)**(1/(2*K))
    
    if Oc_type == "min":
        Omega_c = Omega_c_min
    elif Oc_type == "max":
        Omega_c = Omega_c_max
    elif Oc_type == "mean":
        Omega_c = (Omega_c_max + Omega_c_min) / 2
        # Omega_c = ceil_decimal((Omega_c_max + Omega_c_min) / 2, 1)

    return K, Omega_c


def prop_lp(K):
    # Freq. do final da Banda de Passagem:
    Omega_c = 1     # 1 rad/s

    # Polos do Filtro:
    i = np.arange(1, K+1)       # [1, 2, ..., K]
    polos = 1j * Omega_c * np.exp(1j * np.pi * (2*i - 1) / (2*K))

    # Gerando o polinômio do denominador:
    den = np.poly(polos)

    # Mantendo só a parte real: dos coeficientes:
    denominador = np.real(den)

    # Gerando o polinômio do numerador:
    """ 
        OBS: O numpy, quando detecta que o vetor só possui 1 valor, 
        ele converte o vetor para um valor numérico.
        O np.atleast_1d() serve para forçar o numpy a criar um vetor com apenas 1 valor 
    """
    numerador = np.atleast_1d(denominador[-1])

    K0 = 1

    return numerador, denominador, K0


def lp2lp(b, a, Omega_c):
    # novo numerador:
    M = len(b)-1
    new_b = np.array([b[i] * (1/Omega_c)**(M-i) for i in range(M+1)])
    # novo denominador:
    N = len(a)-1
    new_a = np.array([a[i] * (1/Omega_c)**(N-i) for i in range(N+1)])

    # ajusta o ganhos dos coeficientes:
    new_b = new_b / new_a[0]
    new_a = new_a / new_a[0]

    return new_b, new_a


def lp2hp(b, a, ganho_0, Omega_c):

    # Extrai os polos e zeros:
    polos = np.roots(a)
    zeros = np.roots(b)
    
    # Extranho o ganho:
    ganho = ganho_0

    # Ordem do denominador e numerador:
    N = len(polos)
    M = len(zeros)

    # novos polos:
    new_polos = Omega_c / polos

    # novos zeros:
    new_zeros = np.zeros(int(N))

    # novo ganho:
    new_ganho = ganho

    # novo numerador:
    new_b = new_ganho * np.poly(new_zeros)
    # novo denominador:
    new_a = np.poly(new_polos)

    # ajusta o ganhos dos coeficientes:
    new_b = new_b / new_a[0]
    new_a = new_a / new_a[0]

    return new_b, new_a


def lp2bp(b, a, ganho_0, Omega_c, Omega_p):
    
    # Atualiza para a frequência do passa-baixas correta:
    att_b, att_a = lp2lp(b, a, Omega_c)

    # Extrai os polos e zeros:
    polos = np.roots(att_a)
    zeros = np.roots(att_b)
    
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


def lp2bs(b, a, ganho_0, Omega_c, Omega_p):
    
    # Atualiza para a frequência do passa-baixas correta:
    att_b, att_a = lp2lp(b, a, Omega_c)
    print("Ganho pos lp2lp: ", att_b[0]/att_a[0])

    # Extrai os polos e zeros:
    polos = np.roots(att_a)
    zeros = np.roots(att_b)
    
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
Omega_p = [60, 260]
Omega_s = [100, 150]
alpha_p = 2.2
alpha_s = 20

#  Calcula a Ordem mínima do Filtro e a frequência de :
K, Omega_c = calcula_valores_filtro(Omega_p, Omega_s, alpha_p, alpha_s, Oc_type="min", type="bandstop")
print("Ordem e Freq.: ", K, Omega_c)

# Calcula filtro Low-Pass protótipo:
num_filter, den_filter, ganho_0 = prop_lp(K)
print("Ganho: ", ganho_0)

print("num e den do protótipo: ", num_filter, den_filter)

# Calcula filtro Low-Pass, na freq. de corte:
# num_filter2, den_filter2 = lp2lp(num_filter, den_filter, Omega_c)
# num_filter2, den_filter2 = lp2hp(num_filter, den_filter, ganho_0, Omega_p)
# num_filter2, den_filter2 = lp2bp(num_filter, den_filter, ganho_0, Omega_c, Omega_p)
num_filter2, den_filter2 = lp2bs(num_filter, den_filter, ganho_0, Omega_c, Omega_p)
print("Ganho_bs: ", num_filter2[0]/den_filter2[0])

print(num_filter2, den_filter2)

# ============================================================
# Resposta em Frequência do Filtro:
Ts = 1e-3    # Período de amostragem
w = np.arange(0, 200, Ts)  # Eixo de frequências (rad/s)

# s = jw:
# denominator2 = np.polyval(denominator, 1j * w)
H = np.polyval(num_filter2, 1j * w) / np.polyval(den_filter2, 1j * w)

plt.figure()
plt.plot(w, 20*np.log10(np.abs(H)))
# plt.ylim([-50, 1])
plt.grid()
plt.tight_layout()
plt.show()
