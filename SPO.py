import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

data = pd.read_csv("3DailyReturns.csv", index_col=0, usecols=[1,2,3,4,5,6,7,8])
simpleReturn = pd.read_csv("2SimpleReturn.csv", usecols=[1])
simpleReturn = simpleReturn.iloc[:,0].to_numpy()/100

data = data - 1
mean = data.mean() * 252
cov = data.cov() * 252

#print(cov)

mean = np.array(mean)
cov = np.array(cov)

w = cp.Variable(7)
objective = cp.Minimize(cp.quad_form(w, cov))
constraints = [cp.sum(w)==1]

problem = cp.Problem(objective, constraints)
problem.solve()

#print(data)
print(simpleReturn)
print("Optimal weights:", w.value)
print("Return:", simpleReturn * w.value)

print("Return:", np.sum(simpleReturn * w.value))
print("Min Std:", problem.value**(1/2))