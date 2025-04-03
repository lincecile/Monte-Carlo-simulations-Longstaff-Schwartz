import datetime as dt
from dataclasses import dataclass
from typing import Callable
from Classes_Both.module_option import Option
from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from copy import deepcopy, copy

"""
A data class to represent the parameters of an option derivative.
Attributes:
    price (float): The current price of the underlying asset.
    strike (float): The strike price of the option.
    maturity (float): The time to maturity of the option in years.
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
"""
@dataclass
class OptionDerivativesParameters:
    option: Option  # Objet option
    market: DonneeMarche  # Objet marché
    
    def __getitem__(self, item):
        if item == "price":
            return self.market.prix_spot
        elif item == "strike":
            return self.option.prix_exercice
        elif item == "maturite":
            return self.option.maturite
        elif item == "maturity":
            return self.option.maturity
        elif item == "r":
            return self.market.taux_interet
        elif item == "sigma":
            return self.market.volatilite
        else:
            raise KeyError(f"Paramètre inconnu: {item}")

    def __setitem__(self, item, value):
        if item == "price":
            self.market.prix_spot = value
        elif item == "strike":
            self.option.prix_exercice = value
        elif item == "maturite":
            self.option.maturite = value
        elif item == "maturity":
            self.option.maturity = value
        elif item == "r":
            self.market.taux_interet = value
        elif item == "sigma":
            self.market.volatilite = value
        else:
            raise KeyError(f"Paramètre inconnu: {item}")

    

class OneDimDerivative: 
    """
    A class to compute the first derivative of a given function with respect to one of its parameters.
    -----------
    Attributes:
    f : Callable
        The function for which the derivative is to be computed.
    parameters : OptionDerivativesParameters
        The parameters with which we want to compute the option derivative.
    shift : float, optional
        The shift value used for finite difference approximation (default is 1).
    -----------
    Methods:
    first(along: str) -> float:
        Computes the first derivative of the function with respect to the specified parameter.
    """
    
    def __init__(self, function:Callable, parameters: OptionDerivativesParameters, shift:float=1) -> None:
        self.f = function
        self.parameters = parameters
        self.shift = shift
        
    def first(self, along: str) -> float:
        params_u = deepcopy(self.parameters)
        params_u[along] += self.shift
        
        params_d = deepcopy(self.parameters)
        params_d[along] -= self.shift
        return (self.f(params_u)[0] - self.f(params_d)[0]) / (2 * self.shift)

class TwoDimDerivatives:
    """
    A class to compute second-order partial derivatives of a given function with respect to two parameters.
    -----------
    Attributes:
        f (Callable): The function for which the derivatives are to be computed.
        parameters (OptionDerivativesParameters): The parameters of the function.
        shift (float): The shift value used for finite difference approximation. Default is 1.
    -----------
    Methods:
        second(along1: str, along2: str) -> float:
            Computes the second-order mixed partial derivative of the function with respect to the given parameters.
    """
    def __init__(self, function: Callable, parameters: OptionDerivativesParameters, shift: float = 1) -> None:
        self.f = function
        self.parameters = parameters
        self.shift = shift

    def second(self, along1: str, along2: str) -> float:
        params_uu = deepcopy(self.parameters)
        params_uu[along1] += self.shift
        params_uu[along2] += self.shift

        params_ud = deepcopy(self.parameters)
        params_ud[along1] += self.shift
        params_ud[along2] -= self.shift

        params_du = deepcopy(self.parameters)
        params_du[along1] -= self.shift
        params_du[along2] += self.shift

        params_dd = deepcopy(self.parameters)
        params_dd[along1] -= self.shift
        params_dd[along2] -= self.shift
        return (self.f(params_uu)[0] - self.f(params_ud)[0] - self.f(params_du)[0] + self.f(params_dd)[0]) / (4 * self.shift * self.shift)


class OptionDerivatives: 
    """
    A class to compute the price and the Greeks (delta, vega, theta, gamma) of an option using numerical methods.
    Attributes:
        option (BaseOption): The option for which the derivatives are computed.
        parameters (OptionDerivativesParameters): The parameters of the option.
        pricer_options (dict): Other options for the pricer
    """
    def __init__(self, option: Option, market: DonneeMarche, pricer : LSM_method):
        self.option = deepcopy(option)
        self.market = deepcopy(market)
        self.pricer = pricer
        self.parameters = OptionDerivativesParameters(option, market)
        
    def price(self, params: OptionDerivativesParameters):
    # Mise à jour des valeurs
        self.market.prix_spot = params["price"]
        self.market.volatilite = params["sigma"]
        self.market.taux_interet = params["r"]
        self.option.prix_exercice = params["strike"]
        self.option.maturite = params["maturite"]
        self.option.maturity = params["maturity"]

        period = (self.option.maturite - self.option.date_pricing).days / 365
        # print("Period : ", period)
        # print(self.option.maturity)
        brownian = Brownian(period, 200, 100000, 1)
        pricer = LSM_method(self.option)

        # Pricing avec LSM
        return pricer.LSM(brownian, self.market, method='vector')


    def delta(self): 
        return OneDimDerivative(self.price, self.parameters, shift=1).first("price")
    
    def vega(self):
        return OneDimDerivative(self.price, self.parameters, shift=0.01).first("sigma")
    
    def theta(self):
        return -OneDimDerivative(self.price, self.parameters, shift=1/365).first("maturity")
    
    def gamma(self):
        return TwoDimDerivatives(self.price, self.parameters).second("price", "price")