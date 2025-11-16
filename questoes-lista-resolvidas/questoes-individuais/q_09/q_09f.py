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
# APLICANDO O EFEITO DE TREMOLO:

y_tremolo = np.zeros(Nmax)

Fx = 5    # Frequencia do tremolo
eta = 0.9 # 0 < eta < 1

for n in np.arange(0, Nmax):
    y_tremolo[n] = x[n] * (1 + eta * np.cos(2*np.pi*(Fx/Fs)*n))


# Plotagem:
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2, 1)

ax1 =fig.add_subplot(gs[0])
ax1.plot(time,x)
ax1.set_xlabel('Tempo')
ax1.set_ylabel('$Amplitude$')
ax1.set_title('Sinal Original $x[n]$')

ax2 = fig.add_subplot(gs[1])
ax2.plot(time,y_tremolo)
ax2.set_xlabel('Tempo')
ax2.set_ylabel('Amplitude')
ax2.set_title('$x[n]$ com Reversão')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Exportando audio:
y_tremolo_int = np.int16(y_tremolo * 32767)
wavfile.write("./guitar_tremolo.wav", Fs, y_tremolo_int)

# IPython.display.Audio(data=y_reverse, rate=Fs)