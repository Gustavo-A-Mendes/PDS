import numpy as np
import DigitalFilter as df

# ============================================================
# QUESTÃO 06e - [REJEITA-FAIXAS BUTTERWORTH]

# Parâmetros do filtro:
filter_type = "bandstop"

Ts = 0.01       # Período de amostragem (s)
Omega_p = [20, 100]   # banda de passagem (rad/s)
Omega_s = [40, 80]  # banda de rejeição (rad/s)
alpha_p = 1     # atenuação máxima (dB)
alpha_s = 20    # atenuação mínima (dB)

# Inicializando filtro
digital_filter = df.DigitalFilter()

num, den = digital_filter.butterworth_filter(
    omega_p=Omega_p,
    omega_s=Omega_s,
    alpha_p=alpha_p,
    alpha_s=alpha_s,
    type_response=filter_type,
    oc_type="max",      # requisito da banda de rejeição
    Ts=Ts,
    warping=True
)

parametros = [Omega_p, Omega_s, alpha_p, alpha_s]

# Exibe valores filtrados pelo filtro digital, em rad/amostra:
print(f"""Frequências filtro analógico
\tOmega_p: {Omega_p[0]}
\tOmega_s: {Omega_s[0]}
Frequências filtro digital:
\tOmega_p: {Omega_p[0]*Ts}
\tOmega_s: {Omega_s[0]*Ts}

Numerador: {num}
Denominador: {den}
""")

digital_filter.plot_response(
    num=num,
    den=den,
    x_min=Omega_p[0]*0.8,
    x_max=Omega_p[1]*1.5,
    Ts=Ts,
    params=parametros
)
