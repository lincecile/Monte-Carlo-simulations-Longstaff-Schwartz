
import numpy as np
import scipy.stats as stats
import random
random.seed(42)

n = 1000 #precision de l'approximation
N = 1000 # nombre de trajectoire
T = 1
dt = T / N  # Pas de temps


W_T = np.ones((N,N))
for i in range(N):
    uniform_list = []
    uniform_samples = np.random.uniform(0, 1, N)
    W_T[i] = stats.norm.ppf(uniform_samples) * np.sqrt(dt)
brownian_paths = np.cumsum(W_T, axis=1)
print(brownian_paths)

print(len(brownian_paths))
# brownian_paths = np.cumsum(W_T, axis=1)


W_T = np.ones((N,N))
for i in range(N):
    for j in range(N):
        uniform_samples = np.random.uniform(0, 1)
        W_T[i,j] = stats.norm.ppf(uniform_samples) * np.sqrt(dt)
brownian_paths = np.cumsum(W_T, axis=1)
print(brownian_paths)


exit()
# print(brownian_paths)

W = np.zeros((n+1,N)) 

tildeS0=100 # valeur initiale de l'actif risqué

tildeS=tildeS0*np.ones((n+1,N))

print(tildeS)
print()
S=tildeS0*np.ones((n+1,N))
print(S)
exit()
for j in range(N):
    for i in range(1,n+1):
        # euler scheme W
        # W0 = 0 
        # W_ti = W_{ti-1} + sqrt(variance) * H_i 
        # ici variance = T/n = step et avec H_i = loi normale
        W[i,j]=W[i-1,j]+np.sqrt(step)*sim.randn()
        
        #euler scheme
        # tildeS0 = S0 given
        # tildeS_ti = tildeS_t(i-1) + sigma(U_t(i-1))*tildeS_t(i-1)) * sqrt(T/n) * H_i avec H_i iid N(0,1)
        tildeS[i,j]=tildeS[i-1,j]+sigma(U[i-1,j])*tildeS[i-1,j]*(W[i,j]-W[i-1,j])

        # tildeS_t = S_t / S^0_t par definition
        # S_t = S0_t*tildeS_t 
        S[i,j]=S0t[i,j]*tildeS[i,j]

    






exit()

import numpy as np
import scipy.stats as stats

# Paramètres du modèle
S0 = 100      # Prix initial de l'actif
K = 110       # Prix d'exercice (strike)
r = 0.05      # Taux sans risque
q = 0.00      # Taux de dividende
sigma = 0.2   # Volatilité de l'actif
T = 1         # Temps jusqu'à maturité (1 an)
num_simulations = 10000  # Nombre de simulations Monte Carlo

# Simulation du processus sous risque-neutre
uniform_samples = np.random.uniform(0, 1, num_simulations)
W_T = stats.norm.ppf(uniform_samples) * np.sqrt(T)
brownian_paths = np.cumsum(W_T, axis=1)

# Calcul des prix finaux sous GBM
S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * brownian_paths)

# Calcul du payoff de l'option call européenne
payoffs = np.maximum(S_T - K, 0)

# Actualisation et estimation de la valeur de l'option
NPV = np.mean(payoffs) * np.exp(-r * T)

# Affichage du résultat
# print(NPV)

###################################################################################
###################################  vectoriel  ###################################
###################################################################################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Paramètres du modèle GBM
S0 = 100        # Prix initial du stock
mu = 0.05       # Rendement moyen (5% annuel par exemple)
sigma = 0.2     # Volatilité (20% annuel)
T = 1           # Temps total (1 an)
num_steps = 252 # Nombre de pas (252 jours de bourse)
num_paths = 10  # Nombre de simulations
dt = T / num_steps  # Pas de temps

# Génération du mouvement Brownien
uniform_samples = np.random.uniform(0, 1, (num_paths, num_steps))
normal_increments = stats.norm.ppf(uniform_samples) * np.sqrt(dt)
brownian_paths = np.cumsum(normal_increments, axis=1)

# Calcul du prix du stock avec GBM
time = np.linspace(0, T, num_steps)
drift = (mu - 0.5 * sigma**2) * time
diffusion = sigma * brownian_paths
S_t = S0 * np.exp(drift + diffusion)

# Ajout du point initial S_0
S_t = np.hstack((np.full((num_paths, 1), S0), S_t))

# Affichage des trajectoires des prix de stock
plt.figure(figsize=(10, 6))
plt.plot(S_t.T, alpha=0.7)
plt.title("Simulation de Trajectoire du Prix d'un Stock (GBM)")
plt.xlabel("Temps (jours)")
plt.ylabel("Prix du Stock")
plt.grid(True)
plt.show()

