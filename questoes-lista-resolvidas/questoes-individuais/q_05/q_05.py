import numpy as np

from scipy.signal import bilinear

# Parâmetros de simulação:
Fs = 10
Ts = 1/Fs

'''
    Função para converter FT de tempo contínuo para discreto,
    com transformação bilinear sem warping
'''
def transformacao_bilinear(num, den, fs):

    Ts = 1/fs

    # Extraindo ganho K:
    K = num[0]

    # Extraindo os polos e zeros:
    polos = np.roots(den)
    zeros = np.roots(num)

    # Calculando a Ordem dos polinômios:
    N = len(polos)
    M = len(zeros)

    # ==================================================
    # Calculando novo ganho:
    Kn = K * (np.prod((2/Ts) - zeros) / np.prod((2/Ts) - polos))

    # remapeando polos e zeros, no tempo discreto:
    polos_z = (1 + (polos * (Ts/2))) / (1 - (polos * (Ts/2)))
    zeros_z = (1 + (zeros * (Ts/2))) / (1 - (zeros * (Ts/2)))

    # Adicionando zeros:
    for _ in range (N-M):
        zeros_z = np.append(zeros_z, -1)

    # Gerando numerador e denominador do filtro digital:
    den_z = np.real(np.poly(polos_z))
    num_z = np.real(Kn) * np.real(np.poly(zeros_z))

    return num_z, den_z


# ==================================================
# Coeficientes da FT H(s):
num = [1]                   # 1
den = [1, np.sqrt(2), 1]    # s² + sqrt(2)s + 1

# Resultado da função implementada:
num_z, den_z = transformacao_bilinear(num, den, Fs)
# num_z, den_z
# Resultado da função nativa:
bz, az = bilinear(num, den, fs=Fs)

# Resultado da função implementada:
print("Resultado implementado:")
print("\tNumerador (num_z):\t", np.round(num_z, 6))
print("\tDenominador (den_z):\t", np.round(den_z, 6))
print()
print("Resultado de função nativa:")
print("\tNumerador (bz): \t", np.round(bz, 6))
print("\tDenominador (az): \t", np.round(az, 6))