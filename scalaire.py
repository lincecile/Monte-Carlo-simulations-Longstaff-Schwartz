
import numpy as np
import scipy.stats as stats
import random
import pandas as pd

import matplotlib.pyplot as plt


np.random.seed(42)

###################################################################################
###################################  Scalaire  ####################################
###################################################################################

T = 1 # maturité
n = 100 # nb de pas
N = 2000 # nombre de trajectoire

step = T/n # discretisation

#W0 = 0 valeur initiale du mouvement Brownien standard
W = np.zeros((n+1,N)) # matrice des valeurs du mouvement Brownien

S0 = 100 # valeur initiale de U
S = S0*np.ones((n+1,N)) # matrice n+1 x N rempli de la valeur initiale de  U

# valeur option en T
S_T = np.ones((1,N))

# NPV chaque branche
NPV = np.ones((1,N))

K = 10
sigma = 0.2
r = 0.04
q = 0

for j in range(N): # N trajectoires representés par les colonnes de la matrice
    for i in range(1,n+1): # n pas representés par les lignes de la matrice

        # W0 = 0 comme la matrice est rempli de 0, on ne change pas à la ligne 0
        # W_ti = W_{ti-1} + sqrt(variance) * G_i 
        # ici variance = T/n = step et avec G_i = loi normale
        uniform_samples = np.random.uniform(0, 1)
        W[i,j] = W[i-1,j]+stats.norm.ppf(uniform_samples) * np.sqrt(step)
        
        # S_0 = 100
        # S_t+dt = S_t * exp((r-q) * dt + sigma*delta_brownian - sig^2 / 2 * dt)
        # S[i,j] = S[i-1,j]*np.exp((r - q) * (i - (i-1)) + sigma *(W[i,j]-W[i-1,j]) - sigma**2 / 2 * (i - (i-1)))
    
    S_T[0,j] = S0*np.exp( (r - q - sigma**2 / 2)*T + sigma*W[-1,j])
    NPV[0,j] = max(0,S_T[0,j] - K)
    
       
NPV = np.mean(NPV)*np.exp(-r*T)
print(NPV)

exit()

dates=np.linspace(0,T,n+1) # axes des abscisses

fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1 ligne, 2 colonnes

# Premier graphique
axs[0].plot(dates, W)
axs[0].set_title('Mouvement Brownien')
axs[0].set_xlabel('Temps t')

# Deuxième graphique
axs[1].plot(dates, S, linestyle='--')  # Courbes en pointillés
axs[1].set_title('Trajectoire sous jacent')
axs[1].set_xlabel('Temps t')

# Ajustement pour éviter le chevauchement
plt.tight_layout()
plt.show()