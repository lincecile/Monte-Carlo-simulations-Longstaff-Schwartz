from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_Both.module_option import Option
import datetime as dt
import numpy as np
import time

### TEST ###
start_date = dt.datetime(2024, 1, 1)
end_date = dt.datetime(2026, 1, 1)

market = DonneeMarche(date_debut= start_date,
volatilite=0.2, 
taux_interet=0.06, 
taux_actualisation=0.06,
# dividends=[{"ex_div_date": dt.datetime(2024, 4, 21), "amount": 3, "rate": 0}], 
dividende_ex_date = dt.datetime(2024, 4, 21),
dividende_montant = 0,
dividende_rate=0,
prix_spot=100)

option = Option(date_pricing=start_date, 
                maturite=end_date, 
                prix_exercice=90, call=True, americaine=True)

period = (end_date - start_date).days / 365
brownian = Brownian(period, 10, 1000000, 1)
# avec 5 step europeen marche pas
start_time_vector = time.time()
#priceV2 = option.payoff_LSM(brownian, market, method='vector')
#priceV3 = option.payoff_LSMBBB(brownian, market, method='vector')
priceV4 = option.LSM(brownian, market, method='vector',antithetic=True)

# brownian = Brownian(period, 10, 1000000, 1)
# priceV5 = option.LSM(brownian, market, method='vector')

end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV4)
#print("Prix Vecteur : ", priceV2, priceV3, priceV4,priceV5)


exit()
brownian = Brownian(10, 1000000, 1)
start_time_vector = time.time()
priceV = option.payoff_intrinseque_classique(brownian, market, method='vector')
end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV)
exit()

brownian = Brownian(10, 1000000, 1)
start_time_scalar = time.time()
priceS = option.payoff_intrinseque_classique(brownian, market, method='scalaire')
end_time_scalar = time.time()
scalar_time = end_time_scalar - start_time_scalar
print("Temps exe méthode scalaire : ",scalar_time)
print("Prix Scalaire : ", priceS)

print("Diff : ", priceV-priceS)