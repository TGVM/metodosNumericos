# Trabalho de métodos numéricos
#Lucas Voelcker, Matheus Homrich, Thiago Mello 

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

# Polinomio interpolador de Lagrange
def lagrange():

    x = np.array([0, 6, 15, 21])
    y = np.array([16, 14, 25, 18])


    # Reading interpolation point
    xp = float(input('Escolha o valor de x para a interpolação: '))

    # Set interpolated value initially to zero
    yp = 0

    # Implementing Lagrange Interpolation
    for i in range(len(x)):

        p = 1

        for j in range(len(x)):
            if i != j:
                p = p * (xp - x[j])/(x[i] - x[j]) #funcao do polinomio de lagrange

        yp = yp + p * y[i] #execucao do polinomio

    # mostrar resultado para 1 valor
    print('Valor interpolado para %.3f é %.3f.' % (xp, yp))
    
    # plotagem de gráfico abaixo não funciona

    # x_new = np.arange(0, 2.1, 0.1)
    # plt.scatter(x, y)
    # #plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Polynomial')
    # #plt.plot(x_new, 3x_new**2 - 2x_new + 0*x_new, label=r"$3 x^2 - 2 x$", linestyle='-.')
    # plt.legend()
    # plt.show()


# Polinomio interpolador de Newton
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.05-Newtons-Polynomial-Interpolation.html
def newton():
    def divided_diff(x, y):
    
    #function to calculate the divided differences table
    
        n = len(y)
        coef = np.zeros([n, n])
        # the first column is y
        coef[:,0] = y
        
        for j in range(1,n):
            for i in range(n-j):
                coef[i][j] = \
            (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
                
        return coef

    def newton_poly(coef, x_data, x):
        '''
        evaluate the newton polynomial 
        at x
        '''
        n = len(x_data) - 1 
        p = coef[n]
        for k in range(1,n+1):
            p = coef[n-k] + (x -x_data[n-k])*p
        return p

    x = np.array([0, 6, 15, 21])
    y = np.array([16, 14, 25, 18])
    # get the divided difference coef
    a_s = divided_diff(x, y)[0, :]
    
    # evaluate on new data points
    x_new = np.arange(0, 24)
    y_new = newton_poly(a_s, x, x_new)

    plt.figure(figsize = (12, 8))
    plt.plot(x, y, 'bo')
    plt.plot(x_new, y_new)
    plt.show()

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
# Não tem material de ajuste de curvas em python sem ser do scipy


def ajuste():
    def func(x, a, b, c):
        return a * np.exp(x * 2) + b * x + c
        #y = ax^2 + bx + c

    xdata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    ydata = np.array([16, 15, 15, 15, 15, 14, 14, 14, 16, 18, 20, 22, 23, 24, 25, 25, 25, 24, 22, 21, 19, 18, 17, 16])
    
    plt.plot(xdata, ydata, 'b-', label='data')

    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

print("Menu:")
print("1 - Lagrange")
print("2 - Newton")
print("3 - Spline")
print("4 - Ajuste")
op = int(input('Escolha a operação: '))
if(op==1): lagrange()
if(op==2): newton()
if(op==3): spline()
if(op==4): ajuste()
