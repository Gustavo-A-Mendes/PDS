import numpy as np
import matplotlib.pyplot as plt

i = 1
# funcao para plotar grafico
def plotaRaiz(n, y, er, A, N):
  global i
  plt.subplot(2,2,i)
  plt.stem(n, y, label= f'erro= {er:.2f}')
  plt.xlabel('n')
  plt.ylabel('$y[n]$')
  plt.title(f'$A = {A}$\t\t$N = {N}$')
  plt.legend()
  i +=1

# funcao da raiz
def raizQuadrada (valor, iteracoes):

  # espaço amostral
  n = np.arange(0, iteracoes)
  u = 1*(n>=0)
  # condicoes iniciais
  xn = 0
  yn = 0
  yn1 = valor / 2

  y_vetor = np.zeros(iteracoes)
  for index in np.arange(0, iteracoes):
    xn = valor*u[index]
    yn = 1/2*(yn1 + xn/yn1)
    yn1 = yn
    y_vetor[index] = yn

  # calculo do erro
  raiz_real = np.sqrt(valor)
  erro = y_vetor[-1] - raiz_real

  plotaRaiz(n, y_vetor, erro, valor, iteracoes)
  return y_vetor

# customização da plotagem
plt.figure(figsize=(10,10))

plt.subplots_adjust(hspace=0.5)
plt.suptitle('$ y[n] = \\frac{1}{2} \\left( y[n − 1] + \\frac{x[n]}{y[n − 1]} \\right) $', size=16)

# cálculo dos valores sugeridos
y1 = raizQuadrada(5, 10)
y2 = raizQuadrada(21, 25)
y3 = raizQuadrada(21, 4)
y4 = raizQuadrada(121, 30)
print(f'{y1[-1]:.2f}\t{y2[-1]:.2f}\t{y3[-1]:.2f}\t{y4[-1]:.2f}')

plt.show()