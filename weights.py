import pandas as pd
import numpy as np

data = pd.read_csv("3DailyReturns.csv", index_col=0, usecols=[1,2,3,4,5,6,7,8])

data = data - 1
mean = data.mean() * 252
annualReturn = (data+1).cumprod().iloc[-1,:] - 1
cov = data.cov() * 252

# Tangent Portfolio Weights
meanMinusRf = mean - 0.0472 # Step 1
meanMinusRf = np.array(meanMinusRf)
print("annualReturn:", np.array(annualReturn))
print("mean:", np.array(mean))
print("meanMinusRf:", meanMinusRf)

cov = np.array(cov) # Step 2
inv_cov_matrix = np.linalg.inv(cov)
weights = inv_cov_matrix @ meanMinusRf
print("Unnormalised tangent weights:", weights)

weights /= np.sum(weights) # Step 3

# Minimal Variance Portfolio Weights
vector = np.ones(7)
num = inv_cov_matrix @ vector
den = np.dot(num, vector)

minWeights = num/den
print("tangentWeights:", weights, "\nsum", np.sum(weights), "\nminWeights:", minWeights)
print("Return (tangent):", round(np.sum(mean * weights), 3))
print("Return:", round(np.sum(annualReturn * weights), 3))
print("Standard Deviation: ", round((np.dot(weights.T, np.dot(cov, weights)))**0.5, 3))

print("Return:", round(np.sum(mean * minWeights), 4))
print("Return:", round(np.sum(annualReturn * minWeights), 3))
print("Standard Deviation: ", round((np.dot(minWeights.T, np.dot(cov, minWeights)))**0.5, 3))