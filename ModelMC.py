from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Classes_Both.module_option import Option
from Classes_Both.derivatives import OptionDerivatives, OptionDerivativesParameters
import datetime as dt
import numpy as np
import time

### TEST ###
start_date = dt.datetime(2024, 1, 1)
end_date = dt.datetime(2025, 1, 1)

market = DonneeMarche(date_debut= start_date,
volatilite=0.2, 
taux_interet=0.15, 
taux_actualisation=0.15,
# dividends=[{"ex_div_date": dt.datetime(2024, 4, 21), "amount": 3, "rate": 0}], 
dividende_ex_date = dt.datetime(2024, 2, 21),
dividende_montant = 0,
dividende_rate=0,
prix_spot=100)

option = Option(date_pricing=start_date, 
                maturite=end_date, 
                prix_exercice=80, call=True, americaine=True)

period = (end_date - start_date).days / 365
brownian = Brownian(period, 11, 1000000, 1)
# avec 5 step europeen marche pas
start_time_vector = time.time()
#priceV2 = option.payoff_LSM(brownian, market, method='vector')
#priceV3 = option.payoff_LSMBBB(brownian, market, method='vector')
priceV4 = option.LSM(brownian, market, method='vector',antithetic=False)

# brownian = Brownian(period, 10, 1000000, 1)
# priceV5 = option.LSM2(brownian, market, method='vector')

end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV4)
#print("Prix Vecteur : ", priceV2, priceV3, priceV4,priceV5)

option_deriv = OptionDerivatives(option, market)  
#print("Prix :", option_deriv.price(option_deriv.parameters))
#print("Delta :", option_deriv.delta())
#print("Vega :", option_deriv.vega())
print("Theta :", option_deriv.theta())
#print("Gamma :", option_deriv.gamma())

# Create a pricing engine for this option
pricer = LSM_method(option)
# Use pricing methods
brownian = Brownian(period, 10, 1000000, 1)

print('ici')
price, std_error = pricer.LSM(brownian, market, method='vector')
print("Prix Vecteur2 : ", price)


exit()
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