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
# APLICANDO O EFEITO DE SUBAMOSTRAGEM POR 2:

y_sub2 = np.zeros(int(Nmax/2))
time_sub2 = np.arange(0,int(Nmax/2))*(1/Fs)

for n in np.arange(0,int(Nmax/2)):
    y_sub2[n] = x[2*n]

# Plotagem:
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2, 1)

ax1 =fig.add_subplot(gs[0])
ax1.plot(time,x)
ax1.set_xlabel('Tempo')
ax1.set_ylabel('$Amplitude$')
ax1.set_title('Sinal Original $x[n]$')

ax2 = fig.add_subplot(gs[1])
ax2.plot(time_sub2,y_sub2)
ax2.set_xlabel('Tempo')
ax2.set_ylabel('Amplitude')
ax2.set_title('$x[n]$ com Reversão')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Exportando audio:
y_sub2_int = np.int16(y_sub2 * 32767)
wavfile.write("./guitar_faster.wav", Fs, y_sub2_int)

# IPython.display.Audio(data=y_reverse, rate=Fs)