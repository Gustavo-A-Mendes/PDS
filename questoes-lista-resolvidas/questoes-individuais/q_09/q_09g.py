import numpy as np
import matplotlib.pyplot as plt
import IPython
from scipy.io import wavfile

# ============================================================
# LENDO O ARQUIVO DE AUDIO ORIGINAL:

Fs, x = wavfile.read('files/guitar.wav')
x = x/np.abs(max(x))
Nmax = len(x)

time = np.arange(0,Nmax)*(1/Fs)  #[0 1/Fs 2/Fs 3/Fs .... (N-1)/Fs]

# ============================================================
# APLICANDO O EFEITO DE FUZZ:

y_fuzz = np.zeros(Nmax)

a_gain = 5   # ganho
d_deph = 0.9    # Fuzz Depth (0 < deph < 1)


for n in np.arange(0,Nmax):
  if x[n] >= 0:
    y_e = (1 - np.exp(-a_gain*(x[n])))
  else:
    y_e = -(1 - np.exp(-a_gain*abs(x[n])))

  y_fuzz[n] = d_deph*y_e + (1 - d_deph)*x[n]


# Plotagem:
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2, 1)

ax1 =fig.add_subplot(gs[0])
ax1.plot(time,x)
ax1.set_xlabel('Tempo')
ax1.set_ylabel('$Amplitude$')
ax1.set_title('Sinal Original $x[n]$')

ax2 = fig.add_subplot(gs[1])
ax2.plot(time,y_fuzz)
ax2.set_xlabel('Tempo')
ax2.set_ylabel('Amplitude')
ax2.set_title('$x[n]$ com Reversão')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Exportando audio:
y_fuzz_int = np.int16(y_fuzz * 32767)
wavfile.write("./guitar_fuzzed.wav", Fs, y_fuzz_int)

# IPython.display.Audio(data=y_reverse, rate=Fs)