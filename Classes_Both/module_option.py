#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional

from Classes_TrinomialTree.module_barriere import Barriere

from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import numpy as np
import pandas as pd
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
        T = self.maturity
        import pandas as pd
        if method == 'vector':
            # Pour la méthode vectorielle
            W = brownian.Vecteur()
            S_T = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W[:, :]) - self.get_dividend(market, brownian, 0, 1)
            S_T[:,0] = S0
        else:
            # Pour la méthode scalaire
            S_T = np.ones((brownian.N,brownian.n+1))*S0
            for i in range(brownian.N):
                W = brownian.Scalaire() 
                for j in range(1,brownian.n+1):
                    S_T[i,j] = S0*np.exp( (r - q - sigma**2 / 2)*T + sigma* W[j]) - self.get_dividend( market, brownian, 0, 1)

        ## Exemple
        S_T = np.array([
            [1.00, 1.09, 1.08, 1.34],
            [1.00, 1.16, 1.26, 1.54],
            [1.00, 1.22, 1.07, 1.03],
            [1.00, 0.93, 0.97, 0.92],
            [1.00, 1.11, 1.56, 1.52],
            [1.00, 0.76, 0.77, 0.90],
            [1.00, 0.92, 0.84, 1.01],
            [1.00, 0.88, 1.22, 1.34]
        ])

        return S_T
    
    def payoff_LSM(self, brownian : Brownian, market: DonneeMarche, method='vector'):
        stock_price_paths = self.Price(market, brownian, method=method)

        # Initialisation des cash flows
        CF_vect = np.zeros(len(stock_price_paths))
        print(pd.DataFrame(stock_price_paths))
        
        for t in range(brownian.n , -1, -1):
            print()
            print('temps ',t)
            if self.call:
                intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
            else:
                intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
        
            in_the_money = intrinsic_value > 0
            print(CF_vect)
            # print(intrinsic_value)
            # print(in_the_money)
            # x = input()
            continuation_value = np.zeros_like(intrinsic_value)
            if np.any(in_the_money) and t != brownian.n:  # Vérifie s'il y a des valeurs ITM
                #print(pd.DataFrame(stock_price_paths))
                # print('in')
                X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                Y = CF_vect[in_the_money] * np.exp(-market.taux_interet*brownian.step)
                print(X)
                print(Y)
                estimator = RegressionEstimator(X, Y, degree=2)
                continuation_value[in_the_money] = estimator.get_estimator(X)
                x = input()
                # print(pd.DataFrame(continuation_value))
             
            # x = input()
            exercise = intrinsic_value > continuation_value
            print(sum(exercise))
            # print(pd.DataFrame(exercise))
            CF_vect = np.where(exercise, intrinsic_value, CF_vect*np.exp(-market.taux_interet*brownian.step))
            print(t,brownian.step)
            
            print(np.mean(CF_vect)*np.exp(-market.taux_interet*(brownian.step*t)))
            x = input()

        
        # CF_vect = CF_vect*np.exp(-market.taux_interet*(brownian.step))
        
        return np.mean(CF_vect)

    def payoff_intrinseque_classique(self, brownian : Brownian, market: DonneeMarche, method : str = 'vector') -> float:
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
            payoffs = np.maximum(S_T[:,-1] - self.prix_exercice, 0)
        else:
            payoffs = np.maximum(self.prix_exercice - S_T[:,-1], 0)
        
        # Calcul de la NPV (Net Present Value)
        NPV = np.mean(payoffs) * np.exp(-r * T)

        return NPV
