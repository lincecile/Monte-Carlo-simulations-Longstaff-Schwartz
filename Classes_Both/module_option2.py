import numpy as np
import scipy.stats as si

# Paramètres du problème
S0 = 100  # Spot price
K = 110   # Strike price
T = 1.0   # Maturité (en années)
r = 0.06  # Taux sans risque
sigma = 0.2  # Volatilité
M = 10   # Nombre de pas de temps
N = 1000000  # Nombre de simulations

# Génération des trajectoires de prix
np.random.seed(42)
dt = T / M

dW = np.random.randn(N, M) * np.sqrt(dt)
S = np.zeros((N, M + 1))
S[:, 0] = S0

for t in range(1, M + 1):
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW[:, t - 1])

# Calcul des valeurs d'exercice du put
payoff = np.maximum(K - S, 0)

# Backward Induction pour estimer la valeur d'exercice
V = payoff[:, -1]

def regression_basis(x, degree=5):
    return np.array([x**d for d in range(degree + 1)]).T  # Polynôme jusqu'à degré 5

for t in range(M - 1, 0, -1):
    in_the_money = payoff[:, t] > 0
    X = S[in_the_money, t]
    Y = np.exp(-r * dt) * V[in_the_money]  # Actualisation correcte
    
    if len(X) > 0:
        coeffs = np.polyfit(X, Y, 5)  # Remplacement par polyfit pour plus de stabilité
        continuation_value = np.polyval(coeffs, X)
        exercise = payoff[in_the_money, t] > continuation_value
        V_new = V.copy()
        V_new[in_the_money] = np.where(exercise, payoff[in_the_money, t], Y)
        V = np.exp(-r * dt) * V_new  # Mise à jour après actualisation

# Estimation finale de l'option américaine
put_american = np.mean(V)
print(f"Valeur estimée du put américain: {put_american:.2f}")