from Market import Market
from Brownian import Brownian
from Option import Option
import datetime as dt
import numpy as np

### TEST ###
market = Market(
sigma=0.2, 
r=0.04, 
dividends=[{"ex_div_date": dt.datetime(2025, 4, 21), "amount": 0, "rate": 0}], 
price=100)

option = Option(pricing_date=dt.datetime(2024, 1, 1), maturity_date=dt.datetime(2025, 1, 1), strike=110)

brownian = Brownian(10, 100)


priceV = option.payoff(brownian, market, method='vector')
print("Prix Vecteur : ", priceV)

priceS = option.payoff(brownian, market, method='scalaire')
print("Prix Scalaire : ", priceS)

print("Diff : ", priceV-priceS)