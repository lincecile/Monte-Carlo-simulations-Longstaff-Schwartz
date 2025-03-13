#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional

from Classes_TrinomialTree.module_barriere import Barriere

from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import numpy as np

#%% Classes

@dataclass
class Option : 
    """Classe utilisée pour représenter une option et ses paramètres.
    """
    
    maturite : dt.date
    prix_exercice : float
    barriere : Optional[Barriere] = None
    americaine : bool = False
    call : bool = True
    date_pricing : dt.date = dt.date.today() 

    @property
    def maturity(self) -> float:
        return (self.maturite - self.date_pricing).days / 365.0 # changer à 252 ? faut mettre partout la même chose mais .days() ne prend pas que les BD
    
    def get_dividend(self, market : DonneeMarche, brownian: Brownian, inf, sup):
        if 0 < (market.dividende_ex_date - self.date_pricing).days / 365 <= (self.maturite - self.date_pricing).days / 365:
            return market.dividende_rate
        return 0
    
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
