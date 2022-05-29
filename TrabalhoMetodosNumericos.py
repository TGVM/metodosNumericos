# Trabalho de métodos numéricos
#Lucas Voelcker, Matheus Homrich, Thiago Mello 

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

#x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
#y = [16, 15, 15, 15, 15, 14, 14, 14, 16, 18, 20, 22, 23, 24, 25, 25, 25, 24, 22, 21, 19, 18, 17, 16]

#Sugestões de ações:
# 1. Realizar interpolação para avaliar a função em algum ponto, comparando com dados reais.
# 2. escolher curva a ser ajustada e realizar a avaliação do erro.
# 3. extrapolar valores através do ajuste de curvas.
# 4. utilizar um método iterativo para resolver o sistema linear e obter os coeficientes do polinômio interpolador.


# Polinomio interpolador de Lagrange
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html
def lagrange():

    # Reading number of unknowns
    n = int(input('Enter number of data points: '))

    # Making numpy array of n & n x n size and initializing 
    # to zero for storing x and y value along with differences of y
    x = np.zeros((n))
    y = np.zeros((n))


    # Reading data points
    print('Enter data for x and y: ')
    for i in range(n):
        x[i] = float(input( 'x['+str(i)+']='))
        y[i] = float(input( 'y['+str(i)+']='))


    # Reading interpolation point
    xp = float(input('Enter interpolation point: '))

    # Set interpolated value initially to zero
    yp = 0

    # Implementing Lagrange Interpolation
    for i in range(n):

        p = 1

        for j in range(n):
            if i != j:
                p = p * (xp - x[j])/(x[i] - x[j])

        yp = yp + p * y[i]

    # mostrar resultado para 1 valor
    print('Interpolated value at %.3f is %.3f.' % (xp, yp))
    
    # plotagem de gráfico abaixo não funciona

    # x_new = np.arange(0, 2.1, 0.1)
    # plt.scatter(x, y)
    # #plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Polynomial')
    # #plt.plot(x_new, 3x_new**2 - 2x_new + 0*x_new, label=r"$3 x^2 - 2 x$", linestyle='-.')
    # plt.legend()
    # plt.show()


# Polinomio interpolador de Newton
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.05-Newtons-Polynomial-Interpolation.html

# Splines
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
def spline():
    rng = np.random.default_rng()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    y = [16, 15, 15, 15, 15, 14, 14, 14, 16, 18, 20, 22, 23, 24, 25, 25, 25, 24, 22, 21, 19, 18, 17, 16]

    plt.plot(x, y, 'ro', ms=5)

    spl = UnivariateSpline(x, y)
    xs = np.linspace(0, 23)
    plt.plot(xs, spl(xs), 'g', lw=3)

    spl.set_smoothing_factor(0.5)
    plt.plot(xs, spl(xs), 'b', lw=3)
    plt.show()

# Ajuste de curvas
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


print("Menu:")
print("1 - Lagrange")
print("2 - Newton")
print("3 - Spline")
print("4 - Ajuste")
op = int(input('Escolha a operação: '))
if(op==1): lagrange()
if(op==2): print()
if(op==3): spline()
if(op==4): print()
