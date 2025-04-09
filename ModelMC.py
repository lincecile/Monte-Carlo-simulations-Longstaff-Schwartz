from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Classes_Both.module_option import Option
from Classes_Both.derivatives import OptionDerivatives, OptionDerivativesParameters
import datetime as dt
import numpy as np
import time
from Classes_TrinomialTree.module_arbre_noeud import Arbre
import matplotlib.pyplot as plt
import pandas as pd

# Exemple d'utilisation
from Classes_MonteCarlo_LSM.module_graphique_prix import LSMGraph


### TEST ###
start_date = dt.datetime(2024, 1, 1)
end_date = dt.datetime(2026, 1, 1)

market = DonneeMarche(date_debut= start_date,
volatilite=0.18, 
taux_interet=0.03, 
taux_actualisation=0.03,
# dividends=[{"ex_div_date": dt.datetime(2024, 4, 21), "amount": 3, "rate": 0}], 
dividende_ex_date = dt.datetime(2025, 1, 1),
dividende_montant = 2,
dividende_rate=0,
prix_spot=110)

option = Option(date_pricing=start_date, 
                maturite=end_date, 
                prix_exercice=107, call=True, americaine=True)

pricer = LSM_method(option)

# nombre d'année entre début et fin 
period = (end_date - start_date).days / 365
brownian = Brownian(period, int((end_date - start_date).days / 2), 10000, 1)
price, std_error, intevalles = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="Polynomial")
print("Prix Vecteur polynomial degree 2 : ", price)

trajectoires = pricer.Price(market, brownian, method='vector', antithetic=False)

# Initialisation de la classe avec les objets option et marché
lsm_graph = LSMGraph(option=option, market=market)

# Pour afficher uniquement les trajectoires de prix
fig_prix = lsm_graph.afficher_trajectoires_prix(trajectoires, brownian)
fig_prix.show()

# Pour afficher uniquement les mouvements browniens
fig_brownian = lsm_graph.afficher_mouvements_browniens(brownian)
fig_brownian.show()

# Pour afficher les deux graphiques côte à côte
fig_combinee = lsm_graph.afficher_deux_graphiques(trajectoires, brownian)
fig_combinee.show()

exit()
def plot_mc_intervals(option, market):
    """
    Affiche le prix d'une option pour différentes valeurs de seeds en incluant un intervalle de confiance.
    
    Args:
        option_pricer: Instance contenant la méthode LSM pour calculer le prix.
        market: Instance de DonneeMarche avec les paramètres du marché.
        nb_seeds: Nombre de simulations avec différentes graines (seeds) pour le Brownien.
    """
    
    prices = []
    std_devs = []
    vector_times = []
    #seeds = np.arange(1, nb_seeds + 1)
    nb_points = 250  # Tu peux ajuster ce nombre

    # Intervalle souhaité (de 10 à 1 millions)
    start = 10**1  # 10
    end = 1 * 500  # 1 millions

    # Générer une échelle exponentielle entre 10 et 20 millions
    paths = np.logspace(np.log10(start), np.log10(end), num=nb_points, dtype=int)
    degrees = np.arange(2,8)
    poly_types = ["polynomial", "laguerre", "hermite", "linear", "logarithmic", "exponential"]
    methods = [ "vector", "scalar"]


    for method in methods:
        pricer = LSM_method(option)
        brownian = Brownian(period, 200, 1000, 1)
        start_time_vector = time.time()
        price, std_error, intervall = pricer.LSM(brownian, market, method=method)
        end_time_vector = time.time()
        vector_time = end_time_vector - start_time_vector
        vector_times.append(vector_time)
        prices.append(price)
        std_devs.append(std_error)

    prices = np.array(prices)
    std_devs = np.array(std_devs)

    # Calcul des intervalles de confiance à 95% (± 2 écart-types)
    lower_bound = prices - 2 * std_devs
    upper_bound = prices + 2 * std_devs
    legend_labels = [f"{method} ({time:.2f}s)" for method, time in zip(methods, vector_times)]

    # 📊 **Graphique**
    plt.figure(figsize=(10, 5))
    plt.errorbar(methods, prices, yerr=2 * std_devs, fmt='o', color='blue', ecolor='blue', elinewidth=1, capsize=3)
    plt.axhline(y=np.mean(prices), color='black', linestyle='dashed')  # Ligne moyenne des prix
    plt.xlabel("Methods")
    plt.ylabel("Price")
    plt.title("Intervals around MC prices (+/- 2 std dev)")
    plt.legend(title="Methods (Time in s)", labels=legend_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_mc_intervals2(option, market):
    """
    Affiche l'évolution du prix d'une option en fonction du nombre de paths (simulations).
    
    Args:
        option: Instance contenant la méthode LSM pour calculer le prix.
        market: Instance de DonneeMarche avec les paramètres du marché.
    """

    nb_points = 50  # Nombre de points sur le graphe

    # Générer une échelle exponentielle entre 10 et 20 millions
    paths = np.logspace(1, 3, num=nb_points, dtype=int)  # De 10 à 10 millions
    
    prices_scalar = []
    std_devs_scalar = []
    times_scalar = []

    prices_vector = []
    std_devs_vector = []
    times_vector = []

    methods = ["scalar", "vector"]  

    for method_item in methods:
        print(method_item)
        method_prices = []
        method_std_devs = []
        method_times = []

        for path in paths:
            pricer = LSM_method(option)
            brownian = Brownian(period, 200, path, 1)
            
            start_time = time.time()
            price, std_error = pricer.LSM(brownian, market, method=method_item)
            end_time = time.time()
            
            execution_time = end_time - start_time

            method_prices.append(price)
            method_std_devs.append(std_error)
            method_times.append(execution_time)

        if method_item == "scalar":
            print("method scalar")
            prices_scalar = method_prices
            std_devs_scalar = method_std_devs
            times_scalar = method_times
        else:
            prices_vector = method_prices
            std_devs_vector = method_std_devs
            times_vector = method_times

    # 📊 **Graphique**
    plt.figure(figsize=(10, 5))

    # Courbe Scalaire (Bleu)
    plt.errorbar(paths, prices_scalar, yerr=2 * np.array(std_devs_scalar),
                fmt='-o', color='blue', ecolor='blue', elinewidth=1, capsize=3, label=f"Scalar (Avg: {np.mean(times_scalar):.2f}s)")

    # Courbe Vectorielle (Rouge)
    plt.errorbar(paths, prices_vector, yerr=2 * np.array(std_devs_vector),
                fmt='-o', color='red', ecolor='red', elinewidth=1, capsize=3, label=f"Vector (Avg: {np.mean(times_vector):.2f}s)")

    # Ligne Black-Scholes (27.74)
    plt.axhline(y=27.74, color='black', linestyle='dashed', label="Black-Scholes Price (27.74)")

    # Ajustements
    plt.xscale("log")  # Echelle logarithmique sur X
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Price")
    plt.title("Monte Carlo Pricing Evolution (± 2 std dev)")
    plt.legend(title="Methods & Total Execution Time")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()


### TEST ###
start_date = dt.datetime(2024, 1, 1)
end_date = dt.datetime(2026, 1, 1)

market = DonneeMarche(date_debut= start_date,
volatilite=0.2, 
taux_interet=0.15, 
taux_actualisation=0.15,
# dividends=[{"ex_div_date": dt.datetime(2024, 4, 21), "amount": 3, "rate": 0}], 
dividende_ex_date = dt.datetime(2024, 4, 21),
dividende_montant = 0,
dividende_rate=0,
prix_spot=100)

option = Option(date_pricing=start_date, 
                maturite=end_date, 

                prix_exercice=90, call=False, americaine=True)

period = (end_date - start_date).days / 365
#brownian = Brownian(period, 11, 1000000, 1)
# avec 5 step europeen marche pas
#start_time_vector = time.time()
#priceV2 = option.payoff_LSM(brownian, market, method='vector')
#priceV3 = option.payoff_LSMBBB(brownian, market, method='vector')
#priceV4 = option.LSM(brownian, market, method='vector',antithetic=False)

# brownian = Brownian(period, 10, 1000000, 1)
# priceV5 = option.LSM2(brownian, market, method='vector')

# end_time_vector = time.time()
# vector_time = end_time_vector - start_time_vector
#print("Temps exe méthode vectorielle : ",vector_time)
#print("Prix Vecteur : ", priceV4)
#print("Prix Vecteur : ", priceV2, priceV3, priceV4,priceV5)

# option_deriv = OptionDerivatives(option, market)  
# print("Prix :", option_deriv.price(option_deriv.parameters))
# print("Delta :", option_deriv.delta())
# print("Vega :", option_deriv.vega()/100)
# print("Theta :", option_deriv.theta()/100)
# print("Gamma :", option_deriv.gamma())



# Create a pricing engine for this option
pricer = LSM_method(option)
# Use pricing methods
brownian = Brownian(period, 251, 100000, 1)
print('ici')
price, std_error = pricer.LSM(brownian, market, method='vector',antithetic=True)
print("Prix Vecteur2 : ", price)

nb_pas_arbre = 300
arbre = Arbre(nb_pas_arbre, market, option, pruning = True)
arbre.pricer_arbre()
print(f"Prix option {arbre.prix_option}")

###### test unique
pricer = LSM_method(option)
brownian = Brownian(period, int((end_date - start_date).days / 2), 10000, 1)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="polynomial")
print("Prix Vecteur polynomial degree 2 : ", price)

###### test boucle
from itertools import product
liste_chemin = [10000 * i for i in range(1, 11)]
liste_pas_chemin = [(10000 * i, int((end_date - start_date).days / 2)) for i in range(1, 11)]
liste_pas = [10 * i for i in range(1, 31)]
combinations = list(product(liste_chemin,liste_pas))
dico_price = {}

def test_boucle(combi):
    for (path,pas) in combi:
        brownian = Brownian(period, pas, path, 1)
        price, std_error, intervalle = pricer.LSM(brownian, market, method='vector', antithetic=True, poly_degree=2, model_type="polynomial")
        if intervalle[0] <= arbre.prix_option <= intervalle[1]:
            if int(arbre.prix_option * 100) / 100 == int(price * 100) / 100:# and std_error < 0.01:
                dico_price[(pas,path)] = {'vrai prix': arbre.prix_option,'price': round(price, 4) ,'ecart-type': round(std_error, 4), 
                'min': round(intervalle[0], 4), 'max':round(intervalle[1], 4), 'proche': "✅    "}
                break
            if int(price * 100) / 100 - 0.01 <= int(arbre.prix_option * 100) / 100 <= int(price * 100) / 100 + 0.01:# and std_error < 0.01:
                dico_price[(pas,path)] = {'vrai prix': arbre.prix_option,'price': round(price, 4) ,'ecart-type': round(std_error, 4), 
                'min': round(intervalle[0], 4), 'max':round(intervalle[1], 4),'proche':'❌    '}
        print(pas,path, price)

test_boucle(combinations)
test_boucle(liste_pas_chemin)
print(pd.DataFrame(dico_price))

exit()


brownian = Brownian(period, int((end_date - start_date).days / 2), 10000, 1)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=True,poly_degree=2, model_type="polynomial")
print("Prix Vecteur polynomial degree 2 : ", price)





exit()


brownian = Brownian(period, int((end_date - start_date).days / 2), 100, 1)
price, std_error = pricer.LSM(brownian, market, method='scalar', antithetic=False,poly_degree=2, model_type="polynomial")
print("Prix Vecteur polynomial degree 2 : ", price)


brownian = Brownian(period, 10, 100, 1)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=3, model_type="polynomial")
print("Prix Vecteur polynomial degree 3: ", price)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="linear")
print("Prix Vecteur lineaire: ", price)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="hermite")
print("Prix Vecteur hermite: ", price)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="laguerre")
print("Prix Vecteur laguerre: ", price)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="logarithmic")
print("Prix Vecteur logarithmic: ", price)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="exponential")
print("Prix Vecteur exponential: ", price)
exit()
option_deriv = OptionDerivatives(option, market, pricer)  
#print("Prix :", option_deriv.price(option_deriv.parameters))
#print("Delta :", option_deriv.delta())
#print("Vega :", option_deriv.vega())
print("Theta :", option_deriv.theta())
#print("Gamma :", option_deriv.gamma())


brownian = Brownian(10, 1000000, 1)
start_time_vector = time.time()
priceV = option.payoff_intrinseque_classique(brownian, market, method='vector')
end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV)

brownian = Brownian(10, 1000000, 1)
start_time_scalar = time.time()
priceS = option.payoff_intrinseque_classique(brownian, market, method='scalaire')
end_time_scalar = time.time()
scalar_time = end_time_scalar - start_time_scalar
print("Temps exe méthode scalaire : ",scalar_time)
print("Prix Scalaire : ", priceS)

print("Diff : ", priceV-priceS)
