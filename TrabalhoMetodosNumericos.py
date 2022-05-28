# Trabalho de métodos numéricos
#Lucas Voelcker, Matheus Homrich, Thiago Mello 

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


#Sugestões de ações:
# 1. Realizar interpolação para avaliar a função em algum ponto, comparando com dados reais.
# 2. escolher curva a ser ajustada e realizar a avaliação do erro.
# 3. extrapolar valores através do ajuste de curvas.
# 4. utilizar um método iterativo para resolver o sistema linear e obter os coeficientes do polinômio interpolador.


# Polinomio interpolador de Lagrange
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html

# Polinomio interpolador de Newton
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.05-Newtons-Polynomial-Interpolation.html

# Splines
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
rng = np.random.default_rng()
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
plt.plot(x, y, 'ro', ms=5)

spl = UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3)

spl.set_smoothing_factor(0.5)
plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()

# Ajuste de curvas
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html