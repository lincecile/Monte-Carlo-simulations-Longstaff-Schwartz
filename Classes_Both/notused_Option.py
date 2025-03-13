from dataclasses import dataclass
import datetime as dt
import numpy as np
from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import pandas as pd

@dataclass
class Option:
    date_pricing: dt.datetime
    maturite: dt.datetime
    prix_exercice: float
    call: bool
    american: bool

    @property
    def maturity(self) -> float:
        return (self.maturite - self.date_pricing).days / 365.0 # changer à 252 ? faut mettre partout la même chose mais .days() ne prend pas que les BD
    
    def get_dividend(self, market : DonneeMarche, brownian: Brownian, inf, sup):
        if 0 < (market.dividende_ex_date - self.date_pricing).days / 365 <= (self.maturite - self.date_pricing).days / 365:
            return market.dividende_rate
        return 0


    def compute_cash_flows(stock_price_paths, K=1.1, r=0.06):
        T = stock_price_paths.shape[1] - 1
        
        # Initialisation des cash flows
        CF_matrix = np.zeros_like(stock_price_paths)
        CF_matrix[:, -1] = np.maximum(0, K - stock_price_paths[:, -1])
        
        discount_factor = np.exp(-r)  # Calculé une seule fois
        
        for t in range(T - 1, 0, -1):
            intrinsic_value = np.maximum(0, K - stock_price_paths[:, t])
            in_the_money = intrinsic_value > 0
            
            if np.any(in_the_money):  # Vérifie s'il y a des valeurs ITM
                X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                Y = CF_matrix[in_the_money, t + 1] * discount_factor
                estimator = RegressionEstimator(X, Y, degree=2)
                continuation_value = np.zeros_like(intrinsic_value)
                continuation_value[in_the_money] = estimator.get_estimator(X)
            else:
                continuation_value = np.zeros_like(intrinsic_value)

            exercise = intrinsic_value > continuation_value
            CF_matrix[:, t] = np.where(exercise, intrinsic_value, 0)
            CF_matrix[:, t + 1] *= ~exercise  # Annule les cash flows futurs si exercé
        
        stop_rule = (CF_matrix > 0).astype(int)
        columns = [f't{t}' for t in range(1, T + 1)]
        
        return pd.DataFrame(CF_matrix[:, 1:], columns=columns), pd.DataFrame(stop_rule[:, 1:], columns=columns)

    
    def Price(self, market : DonneeMarche, brownian: Brownian, method: str = 'vector') -> float:
        """
        Calcule le payoff de l'option en utilisant un mouvement brownien.
        
        Args:
            brownian: Le mouvement brownien (sortie de Brownian.Scalaire() ou Brownian.Vecteur())
            market: Instance de la classe DonneeMarche contenant les paramètres du marché
            method: 'vector' ou 'scalar' selon la méthode de calcul du brownian
            
        Returns:
            float: La valeur actualisée du payoff moyen
        """
        # Extraction des paramètres du marché
        S0 = market.prix_spot
        r = market.taux_interet
        sigma = market.volatilite
        q = market.dividende_rate  # Taux de dividende total
        T = 1
        
        if method == 'vector':
            # Pour la méthode vectorielle
            W = brownian.Vecteur()
            S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W[:, -1]) - self.get_dividend(market, brownian, 0, 1)
            
        else:
            # Pour la méthode scalaire
            S_T = np.zeros(brownian.N)
            for i in range(brownian.N):
                W = brownian.Scalaire() 
                S_T[i] = S0*np.exp( (r - q - sigma**2 / 2)*T + sigma* W[-1]) - self.get_dividend( market, brownian, 0, 1)
        return S_T

    def payoff(self, brownian : Brownian, market: DonneeMarche, method : str = 'vector') -> float:
        """
        Calcule le payoff de l'option en utilisant un mouvement brownien.
        
        Args:
            brownian: Le mouvement brownien (sortie de Brownian.Scalaire() ou Brownian.Vecteur())
            market: Instance de la classe DonneeMarche contenant les paramètres du marché
            method: 'vector' ou 'scalar' selon la méthode de calcul du brownian
            
        Returns:
            float: La valeur actualisée du payoff moyen
        """
        # Extraction des paramètres du marché
        #S0 = market.price
        #sigma = market.sigma
        #q = sum(div["rate"] for div in market.dividends)  # Taux de dividende total
        #n = 10
        #N = 100
        r = market.taux_interet
        T = self.maturity
        S_T = self.Price(market, brownian, method=method)

        # Calcul des payoffs selon le type d'option (call ou put)
        if self.call:
            payoffs = np.maximum(S_T - self.prix_exercice, 0)
        else:
            payoffs = np.maximum(self.prix_exercice - S_T, 0)

        # Calcul de la NPV (Net Present Value)
        NPV = np.mean(payoffs) * np.exp(-r * T)

        return NPV
