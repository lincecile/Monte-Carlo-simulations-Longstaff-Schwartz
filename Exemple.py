import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import RegressionEstimator

np.random.seed(42)

T = 3
K = 1.1
r = 6/100

stock_price_paths = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
])

# Cash flow t = 3
CF_t3 = np.maximum(0, K - stock_price_paths[:, -1])

######################################################
######################### T2 #########################
######################################################
ind_stock_price_paths_t2 = [i for i, row in enumerate(stock_price_paths) if np.maximum(0, K - row[2]) != 0]
X = np.array([stock_price_paths[i, 2] for i in ind_stock_price_paths_t2]).reshape(-1, 1)
Y = np.array([CF_t3[i] for i in ind_stock_price_paths_t2]) * np.exp(-r * 1)

estimator = RegressionEstimator(X, Y, degree=2)
continuation_value = estimator.predict(X)

temp = np.zeros(len(stock_price_paths))
for idx, value in zip(ind_stock_price_paths_t2, continuation_value):
    temp[idx] = value
continuation_value = temp

intrinseque_value_t2 = np.maximum(0, K - stock_price_paths[:, -2])
CF_t2 = [intrinseque_value_t2[i] if (intrinseque_value_t2[i] - continuation_value[i] > 0) else 0 for i in range(len(intrinseque_value_t2))]
CF_t3 = [CF_t3[i] if (CF_t2[i] == 0) else 0 for i in range(len(CF_t3))]

######################################################
######################### T1 #########################
######################################################
ind_stock_price_paths_t1 = [i for i, row in enumerate(stock_price_paths) if np.maximum(0, K - row[1]) != 0]
X = np.array([stock_price_paths[i, 1] for i in ind_stock_price_paths_t1]).reshape(-1, 1)
Y = np.array([CF_t2[i] for i in ind_stock_price_paths_t1]) * np.exp(-r * 1)

estimator = RegressionEstimator(X, Y, degree=2)
continuation_value = estimator.predict(X)

temp = np.zeros(len(stock_price_paths))
for idx, value in zip(ind_stock_price_paths_t1, continuation_value):
    temp[idx] = value
continuation_value = temp

intrinseque_value_t1 = np.maximum(0, K - stock_price_paths[:, -3])
CF_t1 = [intrinseque_value_t1[i] if (intrinseque_value_t1[i] - continuation_value[i] > 0) else 0 for i in range(len(intrinseque_value_t1))]
CF_t2 = [CF_t2[i] if (CF_t1[i] == 0) else 0 for i in range(len(CF_t2))]

CF_matrix = pd.DataFrame({'t1': CF_t1, 't2': CF_t2, 't3': CF_t3})
stop_rule = CF_matrix.applymap(lambda x: 1 if x != 0 else 0)

print(CF_matrix)
print()
print(stop_rule)
