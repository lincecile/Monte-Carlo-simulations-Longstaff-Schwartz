
import numpy as np
import scipy.stats as stats
import random
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(42)

T = 3
K = 1.1
r = 6/100

stock_price_paths = [
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
]

stock_price_paths = np.array(stock_price_paths)

# Cash flow t = 3
CF_t3 = np.maximum(0, K - stock_price_paths[:, -1])

######################################################
######################### T2 #########################
######################################################

# chemin ITM à t = 2 sur lesquel faire la regression 
# savoir si on exerce en t=2 ou attendre t=3
ind_stock_price_paths_t2 = [i for i, row in enumerate(stock_price_paths) if np.maximum(0, K - row[2]) != 0]

# Extraire X et Y pour les indices dans ind_stock_price_paths_t2
X = [] # chemin de prix
Y = [] # payoff en date t=3

for i in ind_stock_price_paths_t2:
    X.append([stock_price_paths[i, 2], stock_price_paths[i, 2]**2])
    Y.append(CF_t3[i])

X = np.array(X)
Y = np.array(Y)*np.exp(-r*1) # discount

# Modèle de régression linéaire
X = sm.add_constant(X) # constante 
model = sm.OLS(Y, X)
results = model.fit()

# Affichage des résultats
print(results.summary()) # on a bien  E[ Y / X ] = -1.070 +2.983X - 1 .813X^2.

# prix des stocks en t=2 des chemins ITM
stock_price_t2_used = X[:, 1]  

# Appliquer la formule estimée à chaque prix en t2
continuation_value = 0
for i in range(len(results.params)):
    continuation_value = continuation_value + results.params[i] * stock_price_t2_used**i

temp = np.zeros(len(stock_price_paths))
for idx, value in zip(ind_stock_price_paths_t2, continuation_value):
    temp[idx] = value

continuation_value = temp

# Cash flow t = 2
intrinseque_value_t2 = np.maximum(0, K - stock_price_paths[:, -2])
CF_t2 = [intrinseque_value_t2[i] if (intrinseque_value_t2[i] - continuation_value[i] > 0) else 0 for i in range(len(intrinseque_value_t2))]

# mise à jour CF t3
CF_t3 = [CF_t3[i] if (CF_t2[i] == 0) else 0 for i in range(len(CF_t3))]


######################################################
######################### T1 #########################
######################################################

# chemin ITM à t = 1 sur lesquel faire la regression 
# savoir si on exerce en t=1 ou attendre t=2
ind_stock_price_paths_t1 = [i for i, row in enumerate(stock_price_paths) if np.maximum(0, K - row[1]) != 0]

# Extraire X et Y pour les indices dans ind_stock_price_paths_t1
X = [] # chemin de prix
Y = [] # payoff en date t=2

for i in ind_stock_price_paths_t1:
    X.append([stock_price_paths[i, 1], stock_price_paths[i, 1]**2])
    Y.append(CF_t2[i])

X = np.array(X)
Y = np.array(Y)*np.exp(-r*1) # discount

# Modèle de régression linéaire
X = sm.add_constant(X) # constante 
model = sm.OLS(Y, X)
results = model.fit()

# Affichage des résultats
print(results.summary()) # on a bien  E[ Y 1 X ] = 2.038 - 3.335X + 1.356X^2

# prix des stocks en t=1 des chemins ITM
stock_price_t1_used = X[:, 1]  

# Appliquer la formule estimée à chaque prix en t1
continuation_value = 0
for i in range(len(results.params)):
    continuation_value = continuation_value + results.params[i] * stock_price_t1_used**i

temp = np.zeros(len(stock_price_paths))
for idx, value in zip(ind_stock_price_paths_t1, continuation_value):
    temp[idx] = value

continuation_value = temp

# Cash flow t = 1
intrinseque_value_t1 = np.maximum(0, K - stock_price_paths[:, -3])
CF_t1 = [intrinseque_value_t1[i] if (intrinseque_value_t1[i] - continuation_value[i] > 0) else 0 for i in range(len(intrinseque_value_t1))]

# mise à jour CF t2
CF_t2 = [CF_t2[i] if (CF_t1[i] == 0) else 0 for i in range(len(CF_t2))]


# tableau final
CF_matrix = pd.DataFrame({'t1':CF_t1, 't2':CF_t2, 't3':CF_t3})
stop_rule = CF_matrix.applymap(lambda x: 1 if x != 0 else 0)

print(CF_matrix)
print()
print(stop_rule)
