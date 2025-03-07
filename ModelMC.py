from Market import Market
from Brownian import Brownian
from Option import Option
import datetime as dt
import numpy as np
import time

### TEST ###
market = Market(
sigma=0.2, 
r=0.04, 
dividends=[{"ex_div_date": dt.datetime(2025, 4, 21), "amount": 0, "rate": 0}], 
price=100)

option = Option(pricing_date=dt.datetime(2024, 1, 1), maturity_date=dt.datetime(2025, 1, 1), strike=110)

brownian = Brownian(252, 25000, 42)

start_time_vector = time.time()
priceV = option.payoff(brownian, market, method='vector')
end_time_vector = time.time()
vector_time = end_time_vector - start_time_vector
print("Temps exe méthode vectorielle : ",vector_time)
print("Prix Vecteur : ", priceV)

start_time_scalar = time.time()
priceS = option.payoff(brownian, market, method='scalaire')
end_time_scalar = time.time()
scalar_time = end_time_scalar - start_time_scalar
print("Temps exe méthode scalaire : ",scalar_time)
print("Prix Scalaire : ", priceS)

print("Diff : ", priceV-priceS)