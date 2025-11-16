import numpy as np
import DigitalFilter as df

# ============================================================
# QUESTÃO 06b - [CHEBYSHEV]

# Parâmetros do filtro:
filter_type = "lowpass"

Ts = 1e-3       # Período de amostragem (s)
Omega_p = [100]   # banda de passagem (rad/s)
Omega_s = [200]  # banda de rejeição (rad/s)
alpha_p = 1     # atenuação máxima (dB)
alpha_s = 22    # atenuação mínima (dB)

# Inicializando filtro
digital_filter = df.DigitalFilter()

num, den = digital_filter.chebyshev_filter(
    omega_p=Omega_p,
    omega_s=Omega_s,
    alpha_p=alpha_p,
    alpha_s=alpha_s,
    type_response=filter_type,
    # oc_type="max",      # requisito da banda de rejeição
    Ts=Ts,
    warping=True
)

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

digital_filter.plot_response(num, den, Ts)