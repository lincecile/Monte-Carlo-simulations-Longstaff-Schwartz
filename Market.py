from dataclasses import dataclass
from Brownian import Brownian
import numpy as np
@dataclass
class Market:
    sigma: float
    r: float
    dividends: list
    price: float

    def Price(self, brownian: Brownian, method: str = 'vector') -> float:
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
        S0 = self.price
        r = self.r
        sigma = self.sigma
        q = sum(div["rate"] for div in self.dividends)  # Taux de dividende total
        T = 1
        n = brownian.n
        N = brownian.N
       

        if method == 'vector':
            # Pour la méthode vectorielle
            S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * brownian.Vecteur()[-1, :])
        
            
        else:
            # Pour la méthode scalaire
            S_T = np.ones(N)
            for i in range(1,N):
                #print(brownian.Scalaire()[-1])
                S_T[i] = S0*np.exp( (r - q - sigma**2 / 2)*T + sigma*brownian.Scalaire()[-1])
        return S_T
               