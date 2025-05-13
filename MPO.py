import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag

data = pd.read_csv("3DailyReturns.csv", index_col=0, usecols=[1,2,3,4,5,6,7,8])

length = data.shape[0] // 2
print("length:", length)

h1 = data.iloc[:length].cumprod()
h2 = data.iloc[length:].cumprod()

a1 = h1.iloc[-1]
a2 = h2.iloc[-1]
array = pd.DataFrame([a1,a2])
array = np.array(array)

cov1 = (h1-1).cov() * length
cov2 = (h2-1).cov() * length

phi_1_2 = a1 * a2
phi_2_2 = a2

a = np.concatenate([phi_1_2.values, phi_2_2.values])

Sigma_w = block_diag(cov1.values, cov2.values)
Q = Sigma_w

n_assets = data.shape[1]

T = 2
w = cp.Variable(n_assets * T)

objective = cp.Minimize(cp.quad_form(w, Q))

constraints = []

for k in range(T):
    start = k * n_assets
    end = (k + 1) * n_assets

    if k == 0:
        constraints.append(cp.sum(w[start:end]) == 1)
    else:
        constraints.append(cp.sum(w[start:end]) == 0)

# Solve
prob = cp.Problem(objective, constraints)
prob.solve()

print("Optimal weights:", w.value)
x = w.value[:n_assets]
print("Returns:\n", array)
print("annualReturn:", np.sum((array[0]*array[1]) * x))
print("Min Std:", prob.value**(1/2))