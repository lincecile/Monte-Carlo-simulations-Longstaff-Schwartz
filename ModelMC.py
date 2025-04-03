from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Classes_Both.module_option import Option
from Classes_Both.derivatives import OptionDerivatives, OptionDerivativesParameters
import datetime as dt
import numpy as np
import time
from Classes_TrinomialTree.module_arbre_noeud import Arbre

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

nb_pas_arbre = 300
arbre = Arbre(nb_pas_arbre, market, option, pruning = True)
arbre.pricer_arbre()
print(f"Prix option {arbre.prix_option}")

pricer = LSM_method(option)


from itertools import product
liste_chemin = [10000*i for i in range(1,11)]
liste_pas_chemin = [(int((end_date - start_date).days / 2),x) for x in liste_chemin]
liste_pas = [10*i for i in range(1,31)]

combinations = list(product(liste_chemin,liste_pas))
combinations2 = list(product(liste_pas_chemin, liste_chemin))
dico_price = {}
for (path,pas) in combinations:
    brownian = Brownian(period, pas, path, 1)
    price, std_error, intervalle = pricer.LSM(brownian, market, method='vector', antithetic=True, poly_degree=2, model_type="polynomial")
    # print("Prix Vecteur polynomial degree 2 : ", price)
    # print(int(arbre.prix_option * 100) / 100, int(price * 100) / 100)
    if int(arbre.prix_option * 100) / 100 == int(price * 100) / 100 and intervalle[0] <= arbre.prix_option <= intervalle[1]:# and std_error < 0.01:
        dico_price[(pas,path)] = {'price': round(price, 4) ,'ecart-type': round(std_error, 4), 
        'min': round(intervalle[0], 4), 'max':round(intervalle[1], 4)}
        break
    print(pas,path, price)
    
print(dico_price)

exit()
brownian = Brownian(period, int((end_date - start_date).days / 2), 10000, 1)
price, std_error = pricer.LSM(brownian, market, method='vector', antithetic=False,poly_degree=2, model_type="polynomial")
print("Prix Vecteur polynomial degree 2 : ", price)

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