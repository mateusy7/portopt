import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("3DailyReturns.csv", index_col=0, usecols=[1,2,3,4,5,6,7,8])
simpleReturn = pd.read_csv("2SimpleReturn.csv", usecols=[1])
simpleReturn = simpleReturn.iloc[:,0].to_numpy()/100

data = data - 1
mean = data.mean() * 252
cov = data.cov() * 252

# Random Portfolio Weights
def random_weights(assets):
    while True:  
        weights = np.random.normal(0, 1.5, assets)
        sum_weights = np.sum(weights)
        
        if abs(sum_weights - 1) < 0.01:  # Check if sum is close to 1
            return weights

# Expected Portfolio Annual Return
def port_returns(weights):
    return np.sum(simpleReturn * weights)

# Expected Portfolio Risk
def port_std(weights, cov):
    varP = np.dot(weights.T, np.dot(cov, weights))
    return varP**(1/2)


# Monte Carlo simulation
returnList = []
riskList = []
for i in range(10000):
    weights = random_weights(7)
    returnList.append(port_returns(weights))
    riskList.append(port_std(weights, cov))
returns = np.array(returnList)
risks = np.array(riskList)

# Tangent portfolio and risk-free rate
tangent_return = 4.97016
tangent_std = 1.275
risk_free_rate = 0.0472

# Compute the slope of the CML
slope = (tangent_return - risk_free_rate) / tangent_std
print("max risks:", max(risks))

# Define the tangent line (Capital Market Line)
std_range = np.linspace(0, max(risks), 100)
cml = risk_free_rate + slope * std_range

# Minimal Variance Portfolio (MVP) coordinates
mvp_return = 0.06658
mvp_std = 0.175

# Plot the opportunity set, tangent line, tangent portfolio, and MVP
sharpe_ratios = (returns-risk_free_rate) / risks
tangent_sharpe = (tangent_return - risk_free_rate) / tangent_std
mvp_sharpe = (mvp_return - risk_free_rate) / mvp_std
print(sharpe_ratios.min(), sharpe_ratios.max())
print("tangent sharpe:", tangent_sharpe)
print("mvp sharpe:", mvp_sharpe)

plt.figure(figsize=(10, 6))
sc = plt.scatter(risks, returns, c=sharpe_ratios, marker='o', cmap='coolwarm')
plt.plot(std_range, cml, color='black', linestyle='--', label="Capital Market Line (CML)")
plt.scatter(tangent_std, tangent_return, c=tangent_sharpe, cmap='coolwarm', edgecolor='black',
            zorder=5, label="Tangent Portfolio", vmin=sharpe_ratios.min(), vmax=sharpe_ratios.max())
plt.scatter(mvp_std, mvp_return, c=mvp_sharpe, cmap='coolwarm', edgecolor='black', zorder=5,
            label="Minimal Variance Portfolio (MVP)", vmin=sharpe_ratios.min(), vmax=sharpe_ratios.max())
plt.scatter(0, risk_free_rate, marker='x', c='black', label='Risk Free Rate')
plt.xlim(0,2)
plt.ylim(-5,10)
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return (x10^2 %)')
plt.colorbar(sc, label='Sharpe Ratio')
plt.legend()
plt.title("Efficient Frontier with the Tangent and Minimal Variance Portfolio")
plt.show()