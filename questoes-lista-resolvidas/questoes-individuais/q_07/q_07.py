import numpy as np
import soundfile as sf
import IPython

# ============================================================
# CALCULANDO AS FREQUÊNCIAS W0 DE CADA FILTRO:

Fs = 44100        # Hz
f0 = 235          # Hz
BW = 80           # Hz

harmonicas = [1, 2, 3, 4]  # f1, f2, f3, f4

# Cálculo de ω0 e r:
w0_arr = [(2 * np.pi * h * (f0 / Fs)) for h in harmonicas]
r = 1 - (BW / Fs) * np.pi

print(f"ω0 (rad/amostra): {np.round(w0_arr, 6)}")
print(f"r: {np.round(r, 6)}")

# ============================================================
# lEITURA DO AUDIO:

# Ler o arquivo de áudio:
x, Fs = sf.read("files/vuvuzela.mp3", dtype='float32')

# Converte para mono, caso o audio seja estéreo:
if x.ndim > 1:
    x = x.mean(axis=1)

# Normaliza o sinal:
x = x / np.max(np.abs(x))


# ============================================================
# IMPLEMENTAÇÃO DO FILTRO:

# Aplica cada os filtro em cascata (implementação recursiva):

# Cópia do sinal de entrada:
xn = x.copy()

# Dimensões:
N = len(x)
M = len(harmonicas)

# Aplica cada filtro em cascata:
y = np.zeros((M, N))
for i in np.arange(0, M):

    # Condições Iniciais do filtro i:
    xn2 = 0
    xn1 = 0
    yn2 = 0
    yn1 = 0
    yn = 0

    # y[n] = x[n] - 2cos(ω0)x[n-1] + x[n-2] + 2rcos(ω0)y[n-1] - r²y[n-2]
    for j in np.arange(0, N):
        yn = xn[j] - 2*np.cos(w0_arr[i])*xn1 + xn2 + 2*r*np.cos(w0_arr[i])*yn1 - r**2*yn2
        y[i, j] = yn

        # Atualizando parâmetros:
        xn2 = xn1
        xn1 = xn[j]
        yn2 = yn1
        yn1 = yn

    # Nova entrada para o próximo filtro => Saída do filtro atual:
    xn = y[i, :]

# Sinal de saída final:
y_final = y[M-1, :]

# Normaliza o sinal filtrado:
y_final = y_final / np.max(np.abs(y_final))

# ============================================================
# EXPORTANDO AUDIO FILTRADO:

sf.write('./vuvuzela_filtrado.mp3', y_final, Fs)

