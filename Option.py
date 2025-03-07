from dataclasses import dataclass
import datetime as dt
import numpy as np
from Market import Market
from Brownian import Brownian

@dataclass
class Option:
    pricing_date: dt.datetime
    maturity_date: dt.datetime
    strike: float
    call: bool = True
    american: bool = False

    @property
    def maturity(self) -> float:
        return (self.maturity_date - self.pricing_date).days / 365.0

    def payoff(self, brownian : Brownian, market: Market, method: str = 'vector') -> float:
        """
        Calcule le payoff de l'option en utilisant un mouvement brownien.
        
        Args:
            brownian: Le mouvement brownien (sortie de Brownian.Scalaire() ou Brownian.Vecteur())
            market: Instance de la classe Market contenant les paramètres du marché
            method: 'vector' ou 'scalar' selon la méthode de calcul du brownian
            
        Returns:
            float: La valeur actualisée du payoff moyen
        """
        # Extraction des paramètres du marché
        #S0 = market.price
        #r = market.r
        #sigma = market.sigma
        #q = sum(div["rate"] for div in market.dividends)  # Taux de dividende total
        #T = self.maturity
        #n = 10
        #N = 100
        r = market.r
        T = self.maturity
        S_T = market.Price(brownian, method=method)

        if method == 'vector':
            # Pour la méthode vectorielle
            payoffs = np.maximum(S_T - self.strike, 0)
            #payoffs = np.maximum(market.Price(brownian) - self.strike, 0)
            #print(market.Price(brownian))
            #print(payoffs)
        else:
            # Pour la méthode scalaire
            payoffs = np.maximum(S_T - self.strike, 0)
            #payoffs = np.ones(N)
            #for i in range(0,N):
                #payoffs[i] = max(0,market.Price(brownian, "scalaire")[i] - self.strike)
                #print(market.Price(brownian, "scalaire"))
                #x = input()
            #print(payoffs)
                
            #S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * brownian[-1])
            #S_T = np.array([S_T])  # Conversion en array pour uniformiser le traitement

        # Calcul des payoffs selon le type d'option (call ou put)
        #if self.call:
         #   payoffs = np.maximum(S_T - self.strike, 0)
        #else:
         #   payoffs = np.maximum(self.strike - S_T, 0)

        # Calcul de la NPV (Net Present Value)
        NPV = np.mean(payoffs) * np.exp(-r * T)
        return NPV
