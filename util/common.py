import numpy as np
import matplotlib.pyplot as plt

plt.style.use('util/deeplearning.mplstyle')

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0';
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')



def compute_cost(X: np.array, y: np.array, w:np.array, b:float) -> float:
    """ Loop version of multi-variable compute_cost """
    m = X.shape[0]

    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b           #(n,)(n,)=scalar
        cost = cost + (f_wb_i - y[i])**2

    cost = cost / (2 * m)
    return cost 