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
    
    def Price(self, market : DonneeMarche, brownian: Brownian, method: str = 'vector', antithetic : bool=False) -> float:
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
            if antithetic:
                W_neg = -W
                S_T_pos = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W[:, :]) - self.get_dividend(market, brownian, 0, 1)
                S_T_neg = S0 * np.exp((r - q - sigma**2/2) * T + sigma * W_neg[:, :]) - self.get_dividend(market, brownian, 0, 1)
                S_T_pos[:,0] = S0
                S_T_neg[:,0] = S0
                return S_T_pos, S_T_neg
            else:
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
        """ S_T = np.array([
            [1.00, 1.09, 1.08, 1.34],
            [1.00, 1.16, 1.26, 1.54],
            [1.00, 1.22, 1.07, 1.03],
            [1.00, 0.93, 0.97, 0.92],
            [1.00, 1.11, 1.56, 1.52],
            [1.00, 0.76, 0.77, 0.90],
            [1.00, 0.92, 0.84, 1.01],
            [1.00, 0.88, 1.22, 1.34]
        ]) """

        return S_T
    
    # marche call euro et americain
    def payoff_LSM(self, brownian : Brownian, market: DonneeMarche, method='vector'):
        stock_price_paths = self.Price(market, brownian, method=method)

        # Initialisation des cash flows
        CF_vect = np.zeros(len(stock_price_paths))
        if self.call:
            CF_vect = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        else:
            CF_vect = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        
        for t in range(brownian.n , 0, -1):

            if self.call:
                intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
            else:
                intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
        
            in_the_money = intrinsic_value > 0

            if t < brownian.n - 1:

                continuation_value = np.zeros_like(intrinsic_value)
                if np.sum(in_the_money) > 0:#np.any(in_the_money) and t != brownian.n:  # Vérifie s'il y a des valeurs ITM

                    X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                    Y = CF_vect[in_the_money] * np.exp(-market.taux_interet*brownian.step)

                    estimator = RegressionEstimator(X, Y, degree=2)
                    continuation_value[in_the_money] = estimator.get_estimator(X)

                #x = input()
                exercise = (intrinsic_value > continuation_value) & in_the_money

                CF_vect = np.where(exercise, intrinsic_value, CF_vect*np.exp(-market.taux_interet*brownian.step))
    
        CF_vect = CF_vect*np.exp(-market.taux_interet*brownian.step)
        
        return np.mean(CF_vect)
    
    # marche call et put euro
    def payoff_LSMBBB(self, brownian : Brownian, market: DonneeMarche, method='vector'):
        stock_price_paths = self.Price(market, brownian, method=method)

        # Initialisation des cash flows
        CF_vect = np.zeros(len(stock_price_paths))

        # Final payoff calculation based on option type
        if self.call:
            CF_vect = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        else:
            CF_vect = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        
        # Early exercise logic for American options
        if self.americaine:
            for t in range(brownian.n, 0, -1):
                # Calculate intrinsic value at current time step
                if self.call:
                    intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
                else:
                    intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
            
                in_the_money = intrinsic_value > 0
                
                # if t < brownian.n - 1 and np.sum(in_the_money) > 0:
                if t < brownian.n - 1:

                    continuation_value = np.zeros_like(intrinsic_value)
                    if np.sum(in_the_money) > 0:
                        # Regression to estimate continuation value
                        X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                        Y = CF_vect[in_the_money] * np.exp(-market.taux_interet * brownian.step)
                        
                        estimator = RegressionEstimator(X, Y, degree=2)
                        # continuation_value = np.zeros_like(intrinsic_value)
                        continuation_value[in_the_money] = estimator.get_estimator(X)
                    
                    # Decide whether to exercise early
                    exercise = (intrinsic_value > continuation_value) & in_the_money
                    CF_vect = np.where(exercise, intrinsic_value, CF_vect * np.exp(-market.taux_interet * brownian.step))
            # CF_vect = CF_vect*np.exp(-market.taux_interet*brownian.step)
        
        # Discounting the final cash flows
        return np.mean(CF_vect * np.exp(-market.taux_interet * self.maturity))
    

    # call US ok 1cts d'ecart
    # call euro ok
    # put euro ok

    def payoff_LSM3(self, brownian : Brownian, market: DonneeMarche, method='vector', antithetic: bool = False):
        # Générer les trajectoires des prix

        if antithetic:
            stock_price_paths_pos, stock_price_paths_neg = self.Price(market, brownian, method=method, antithetic=antithetic)
            stock_price_paths = np.concatenate([stock_price_paths_pos, stock_price_paths_neg], axis=0)
        else:
            stock_price_paths = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul du payoff final
        if self.call:
            final_payoff = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        else:
            final_payoff = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        
        # Pour les options européennes, retourner simplement le payoff actualisé
        if not self.americaine:
            prix = np.mean(final_payoff * np.exp(-market.taux_interet * self.maturity))
            std_prix = np.std(final_payoff * np.exp(-market.taux_interet * self.maturity)) / np.sqrt(len(final_payoff))
            print("Nb chemins :",len(final_payoff))
            print("Prix min :",prix - 2*std_prix)
            print("Prix max :",prix + 2*std_prix)
            return (prix, std_prix)
        
        # Pour les options américaines
        CF_vect = final_payoff.copy()
        
        # Parcourir les étapes à l'envers
        for t in range(brownian.n - 1, 0, -1):
            # Calculer la valeur intrinsèque à l'étape courante
            if self.call:
                intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
            else:
                intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
            
            # Sélectionner uniquement les chemins intéressants
            in_the_money = intrinsic_value > 0
            
            if np.sum(in_the_money) > 0:
                # Préparer les données pour la régression
                X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                Y = CF_vect[in_the_money] * np.exp(-market.taux_interet * brownian.step)
                
                # Estimer la valeur de continuation
                estimator = RegressionEstimator(X, Y, degree=2)
                continuation_value = np.zeros_like(intrinsic_value)
                continuation_value[in_the_money] = estimator.get_estimator(X)
                
                # Décider de l'exercice anticipé
                exercise = (intrinsic_value > continuation_value) & in_the_money
                
                # Mettre à jour les cash-flows
                CF_vect[exercise] = intrinsic_value[exercise]
            
            # Actualiser tous les cash-flows
            CF_vect *= np.exp(-market.taux_interet * brownian.step)

        if not antithetic:
            prix = np.mean(CF_vect)
            std_prix = np.std(CF_vect) / np.sqrt(len(CF_vect))
            print("Nb chemins :",len(CF_vect))
            print("Prix min :",prix - 2*std_prix)
            print("Prix max :",prix + 2*std_prix)

        moitie = len(CF_vect) // 2
        CF_vect_final = (CF_vect[:moitie] + CF_vect[moitie:]) / 2
        prix = np.mean(CF_vect_final)
        std_prix = np.std(CF_vect_final) / np.sqrt(len(CF_vect_final))
        print("Nb chemins :",len(CF_vect))
        print("Prix min :",prix - 2*std_prix)
        print("Prix max :",prix + 2*std_prix)

        return (prix, std_prix)


    def payoff_LSM4(self, brownian: Brownian, market: DonneeMarche, method='vector'):
    
        stock_price_paths = self.Price(market, brownian, method=method)

        # Calcul du payoff final
        if self.call:
            final_payoff = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        else:
            final_payoff = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        
        # Pour les options européennes, retourner simplement le payoff actualisé
        if not self.americaine:
            return np.mean(final_payoff * np.exp(-market.taux_interet * self.maturity))
        
        # Pour les options américaines
        CF_vect = final_payoff.copy()
        
        # Parcourir les étapes à l'envers
        for t in range(brownian.n - 1, 0, -1):
            # Calculer la valeur intrinsèque à l'étape courante
            if self.call:
                intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
            else:
                intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
            
            # Sélectionner uniquement les chemins intéressants
            in_the_money = intrinsic_value > 0
            
            if np.sum(in_the_money) > 0:
                # Préparer les données pour la régression
                X = stock_price_paths[in_the_money, t].reshape(-1, 1)
                Y = CF_vect[in_the_money] * np.exp(-market.taux_interet * brownian.step)
                
                # Estimer la valeur de continuation
                estimator = RegressionEstimator(X, Y, degree=2)
                continuation_value = np.zeros_like(intrinsic_value)
                continuation_value[in_the_money] = estimator.get_estimator(X)
                
                # Décider de l'exercice anticipé
                exercise = (intrinsic_value > continuation_value) & in_the_money
                
                # Mettre à jour les cash-flows
                CF_vect[exercise] = intrinsic_value[exercise]
            
            # Actualiser tous les cash-flows
            CF_vect *= np.exp(-market.taux_interet * brownian.step)
        
        return np.mean(CF_vect)

    def payoff_LSM2(self, brownian : Brownian, market: DonneeMarche, method='vector'):
        stock_price_paths = self.Price(market, brownian, method=method)
        stock_price_paths = np.array(stock_price_paths)
        if self.call:
            intrinsic_value_paths = np.maximum(0, stock_price_paths - self.prix_exercice)
            print('in call')
        else:
            intrinsic_value_paths = np.maximum(0, self.prix_exercice - stock_price_paths)
            print('in put')

        # Initialisation des cash flows
        CF_vect = intrinsic_value_paths[:,-1]
        print(pd.DataFrame(stock_price_paths))
        print(pd.DataFrame(intrinsic_value_paths))
        
        for t in range(brownian.n-1 , 0, -1):
            print()
            print('temps ',t)
            stock_price_t = stock_price_paths[:, t]
            intrinsic_value_t = intrinsic_value_paths[:,t]

            in_the_money = intrinsic_value_t > 0
            continuation_value = np.zeros_like(intrinsic_value_t)
            print(in_the_money)
                        
            if np.any(in_the_money):# and t != brownian.n:  # Vérifie s'il y a des valeurs ITM
                print('in')
                X = stock_price_t[in_the_money]
                print('X',X)
                print('step',brownian.step)
                print('t',t)
                print('market.taux_interet',market.taux_interet,-market.taux_interet*brownian.step)
                print('IV', CF_vect[in_the_money])
                Y = CF_vect[in_the_money] * np.exp(-market.taux_interet*brownian.step)
                print('Y',Y)
                estimator = RegressionEstimator(X, Y, degree=2)
                continuation_value[in_the_money] = estimator.get_estimator(X)
                print(continuation_value)
        
            # x = input()
            exercise = intrinsic_value_t > continuation_value
            print('nb exercise', sum(exercise))
            print(CF_vect*np.exp(-market.taux_interet*brownian.step))
            # if sum(exercise) > brownian.N*20/100:
            #     print(t)
            #     x = input()
            CF_vect = np.where(exercise, intrinsic_value_t, CF_vect*np.exp(-market.taux_interet*brownian.step))
            print(CF_vect)
        print('t=0')
        print(np.mean(CF_vect*np.exp(-market.taux_interet*(brownian.step))))
            # x = input()
        
        return np.mean(CF_vect)*np.exp(-market.taux_interet*(brownian.step))

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
