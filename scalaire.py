
import numpy as np
import scipy.stats as stats
import random
import pandas as pd
random.seed(42)

n = 1000 #precision de l'approximation
N = 1000 # nombre de trajectoire
T = 1
dt = T / N  # Pas de temps


###################################################################################
###################################  Scalaire  ####################################
###################################################################################

T = 3 # maturité
n = 5 # nb de pas
N = 10 # nombre de trajectoire

step = T/n # discretisation

#W0 = 0 valeur initiale du mouvement Brownien standard
W = np.zeros((n+1,N)) # matrice des valeurs du mouvement Brownien

S0 = 100 # valeur initiale de U
S = S0*np.ones((n+1,N)) # matrice n+1 x N rempli de la valeur initiale de  U

K = 10
sigma = 0.2
r = 0.2
q = 0.1

for j in range(N): # N trajectoires representés par les colonnes de la matrice
    for i in range(1,n+1): # n pas representés par les lignes de la matrice

        # W0 = 0 comme la matrice est rempli de 0, on ne change pas à la ligne 0
        # W_ti = W_{ti-1} + sqrt(variance) * G_i 
        # ici variance = T/n = step et avec G_i = loi normale
        uniform_samples = np.random.uniform(0, 1)
        W[i,j] = W[i-1,j]+stats.norm.ppf(uniform_samples) * np.sqrt(step)
        
        # S_0 = 100
        # S_t+dt = S_t * exp((r-q) * dt + sigma*delta_brownian - sig^2 / 2 * dt)
        S[i,j] = S[i-1,j]*np.exp((r - q) * (i - (i-1)) + sigma *(W[i,j]-W[i-1,j]) - sigma**2 / 2 * (i - (i-1)))

S_T = S0*np.exp( (r - q - sigma**2 / 2)*T + sigma*W[-1,-1])

NPV = np.mean(max(0,S_T - K))*np.exp(-r*T)
       
print(pd.DataFrame(W))
print()
print(pd.DataFrame(S))
print(NPV)
