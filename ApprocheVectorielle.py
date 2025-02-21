import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Paramètres
T = 1           # maturité
n = 100         # nb de pas
N = 2000        # nombre de trajectoires
step = T/n      # discrétisation
S0 = 100        # valeur initiale
K = 110          # strike
sigma = 0.2     # volatilité
r = 0.04        # taux sans risque
q = 0           # dividende

# Génération vectorielle des mouvements browniens
uniform_samples = np.random.uniform(0, 1, (n, N))
Z = stats.norm.ppf(uniform_samples)
dW = np.sqrt(step) * Z

# Construction du mouvement brownien
W = np.zeros((n+1, N))
W[1:, :] = np.cumsum(dW, axis=0)

# Calcul vectoriel des prix finaux
S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W[-1, :])

# Calcul vectoriel des payoffs
payoffs = np.maximum(S_T - K, 0)

# Calcul de la NPV
NPV = np.mean(payoffs) * np.exp(-r * T)

print(f"Prix de l'option: {NPV}")

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