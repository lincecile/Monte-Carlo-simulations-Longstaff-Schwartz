from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_Both.module_option import Option
import datetime as dt
import numpy as np
import time

### TEST ###
market = DonneeMarche(date_debut= dt.datetime(2024, 1, 1),
volatilite=0.2, 
taux_interet=0.06, 
taux_actualisation=0.06,
# dividends=[{"ex_div_date": dt.datetime(2024, 4, 21), "amount": 3, "rate": 0}], 
dividende_ex_date = dt.datetime(2024, 4, 21),
dividende_montant = 0,
dividende_rate=0,
prix_spot=1)

option = Option(date_pricing=dt.datetime(2024, 1, 1), 
                maturite=dt.datetime(2025, 1, 1), 
                prix_exercice=1.1, call=False, americaine=True)

brownian = Brownian(3, 8, 42)

start_time_vector = time.time()
priceV2 = option.payoff_LSM(brownian, market, method='vector')

end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV2)

exit()

start_time_vector = time.time()
priceV = option.payoff_intrinseque_classique(brownian, market, method='vector')
end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV)


start_time_scalar = time.time()
priceS = option.payoff_intrinseque_classique(brownian, market, method='scalaire')
end_time_scalar = time.time()
scalar_time = end_time_scalar - start_time_scalar
print("Temps exe méthode scalaire : ",scalar_time)
print("Prix Scalaire : ", priceS)

print("Diff : ", priceV-priceS)