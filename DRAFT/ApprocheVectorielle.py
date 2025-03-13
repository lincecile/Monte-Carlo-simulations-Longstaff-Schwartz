import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

#np.random.seed(42)

def black_scholes_call(S0, K, T, r, q, sigma):
    """
    Calcule le prix théorique d'un call européen selon Black-Scholes
    
    Paramètres:
    S0 : prix initial du sous-jacent
    K : strike
    T : maturité
    r : taux sans risque
    q : taux de dividende
    sigma : volatilité
    """
    d1 = (np.log(S0/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S0 * np.exp(-q * T) * stats.norm.cdf(d1) - 
                  K * np.exp(-r * T) * stats.norm.cdf(d2))
    
    return call_price

# Paramètres
T = 1           # maturité
n = 1000         # nb de pas
N = 10000        # nombre de trajectoires
step = T/n      # discrétisation
S0 = 100        # valeur initiale
K = 110          # strike
sigma = 0.2     # volatilité
r = 0.04        # taux sans risque
q = 0           # dividende

mc_prices = []
bs_price = black_scholes_call(S0, K, T, r, q, sigma)
trajectories = np.linspace(100, 50000, 50).astype(int)
for N in trajectories:
    np.random.seed(42)
    # Génération vectorielle des mouvements browniens
    uniform_samples = np.random.uniform(0, 1, (n, N))
    Z = stats.norm.ppf(uniform_samples) * np.sqrt(step)
    # Construction du mouvement brownien
    W = np.zeros((n+1, N))
    W[1:, :] = np.cumsum(Z, axis=0)

    # Calcul vectoriel des prix finaux
    S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W[-1, :])

    # Calcul vectoriel des payoffs
    payoffs = np.maximum(S_T - K, 0)

    # Calcul de la NPV
    NPV = np.mean(payoffs) * np.exp(-r * T)
    mc_prices.append(NPV)

print(f"Prix de l'option: {NPV}")
plt.figure(figsize=(10, 6))
plt.plot(trajectories, mc_prices, label='Prix Monte Carlo', color='blue')
plt.axhline(y=bs_price, color='red', linestyle='--', label='Prix Black-Scholes')
plt.xlabel('Nombre de trajectoires')
plt.ylabel('Prix de l\'option')
plt.title('Convergence du prix de l\'option\nMonte Carlo vs Black-Scholes')
plt.legend()
plt.xscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

exit()

# Visualisation
dates = np.linspace(0, T, n+1)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Premier graphique - Mouvement Brownien
axs[0].plot(dates, W)
axs[0].set_title('Mouvement Brownien')
axs[0].set_xlabel('Temps t')

# Deuxième graphique - Prix du sous-jacent
S = S0 * np.exp((r - q - sigma**2/2) * dates.reshape(-1, 1) + sigma * W)
axs[1].plot(dates, S, linestyle='--')
axs[1].set_title('Trajectoire sous jacent')
axs[1].set_xlabel('Temps t')

plt.tight_layout()
plt.show()