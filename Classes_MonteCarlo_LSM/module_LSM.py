#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional

from Classes_TrinomialTree.module_barriere import Barriere

from Classes_Both.module_marche import DonneeMarche
from Classes_Both.module_option import Option
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial, Laguerre, Hermite

#%% Classes

class LSM_method : 
    """Classe utilisée pour calculer le prix d'une option."""
    
    def __init__(self, option: Option):
        self.option = option
        
    def get_dividend(self, market: DonneeMarche, brownian: Brownian, inf, sup):
        if 0 < (market.dividende_ex_date - self.option.date_pricing).days / 365 <= (self.option.maturite - self.option.date_pricing).days / 365:
            return market.dividende_rate
        return 0
    
    def __calcul_position_div (self, market: DonneeMarche, brownian: Brownian):
        """Nous permet de calculer la position du dividende dans l'arbre

        Returns:
            float: nous renvoie la position d'ex-date du div, exprimé en nombre de pas dans l'arbre.
        """
        nb_jour_detachement = (market.dividende_ex_date - self.option.date_pricing).days
        position_div = nb_jour_detachement / 365 / brownian.step
        # print("position_div", position_div)
        # print('market.step', brownian.step)
        
        return position_div
    
    def Price(self, market: DonneeMarche, brownian: Brownian, method: str = 'vector', antithetic: bool = False):
        """
        Calcule le val_intriseque de l'option en utilisant un mouvement brownien.
        
        Args:
            brownian: Le mouvement brownien (sortie de Brownian.Scalaire() ou Brownian.Vecteur())
            market: Instance de la classe DonneeMarche contenant les paramètres du marché
            method: 'vector' ou 'scalar' selon la méthode de calcul du brownian
            
        Returns:
            float: La valeur actualisée du val_intriseque moyen
        """

        # Extraction des paramètres du marché
        S0 = market.prix_spot
        taux_interet = market.taux_interet
        sigma = market.volatilite
        q = market.dividende_montant  # Taux de dividende total
        T = self.option.maturity
        
        if method == 'vector':
            # Pour la méthode vectorielle
            W, timedelta = brownian.Vecteur()
            S_T = S0 * np.exp((taux_interet - sigma**2/2) * timedelta + sigma * W)
            # print('la')
            # print(pd.DataFrame(S_T))
            if q > 0:
                print('in dividende price')
                position_div = self.__calcul_position_div(market=market, brownian=brownian)
                # print(position_div)
                S_T[:, int(position_div)+1] = S_T[:, int(position_div)+1] - q
                S_T[:, int(position_div) + 2:] = S_T[:, int(position_div) + 1][:, np.newaxis] * np.exp(
                        (taux_interet - sigma**2 / 2) * (timedelta[int(position_div) + 2:] - timedelta[int(position_div) + 1]) + sigma * (
                        W[:, int(position_div) + 2:] - W [:, int(position_div) + 1][:, np.newaxis]))
            
            S_T[:,0] = S0
            # print(pd.DataFrame(S_T))
            # x = input()
            
            if antithetic:
                print('in antithetic price')
                W_neg = -W
                S_T_pos = S0 * np.exp((taux_interet - sigma**2/2) * timedelta + sigma * W)
                S_T_neg = S0 * np.exp((taux_interet - sigma**2/2) * timedelta + sigma * W_neg)
                S_T_pos[:,0] = S0
                S_T_neg[:,0] = S0
                return S_T_pos, S_T_neg
        else:
            # Pour la méthode scalaire
            S_T = np.ones((brownian.nb_trajectoire, brownian.nb_step+1)) * S0
            for i in range(brownian.nb_trajectoire):
                W = brownian.Scalaire() 
                for j in range(1, brownian.nb_step+1):
                    S_T[i,j] = S0 * np.exp((taux_interet - q - sigma**2 / 2) * T + sigma * W[j]) - self.get_dividend(market, brownian, 0, 1)

        return S_T

    def LSM(self, brownian: Brownian, market: DonneeMarche, poly_degree=2, model_type="polynomial", method='vector', antithetic: bool=False):
        # Prix du sous-jacent simulé
        if antithetic:
            stock_price_paths_pos, stock_price_paths_neg = self.Price(market, brownian, method=method, antithetic=antithetic)
            Spot_simule = np.concatenate([stock_price_paths_pos, stock_price_paths_neg], axis=0)
        else:
            Spot_simule = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul valeur intrinsèque à chaque pas de temps
        if self.option.call:
            val_intriseque = np.maximum(Spot_simule[:,-1] - self.option.prix_exercice, 0.0)
        else:
            val_intriseque = np.maximum(self.option.prix_exercice - Spot_simule[:,-1], 0.0)

        # Valeur de l'option européenne
        if not self.option.americaine:
            prix = np.mean(val_intriseque * np.exp(-market.taux_interet * self.option.maturity))
            std_prix = np.std(val_intriseque * np.exp(-market.taux_interet * self.option.maturity)) / np.sqrt(len(val_intriseque))
            print("Nb chemins :", len(val_intriseque))
            print("Prix min :", prix - 2*std_prix)
            print("Prix max :", prix + 2*std_prix)
            return (prix, std_prix)
        
        # Matrice des cash flows
        CF_Vect = val_intriseque
        
        # Algo LSM
        for t in range(brownian.nb_step - 1, 0, -1): 
            
            # CF en t1 actualisé
            discounted_CF_next = CF_Vect * np.exp(-market.taux_interet * self.option.maturity / brownian.nb_step)
            
            if self.option.call:
                val_intriseque2 = np.maximum(Spot_simule[:,t] - self.option.prix_exercice, 0.0)
            else:
                val_intriseque2 = np.maximum(self.option.prix_exercice - Spot_simule[:,t], 0.0)

            # Chemins dans la monnaie en t            
            in_the_money = val_intriseque2 > 0

            # CF en t1 actualisé en t par défaut
            CF_Vect = discounted_CF_next.copy()
            
            # Si des chemins sont dans la monnaie en t, on fait la regression
            if np.any(in_the_money):  
                
                X = Spot_simule[in_the_money, t]       # prix du sous jacent en t
                Y = discounted_CF_next[in_the_money]   # CF des chemins dans la monnaie en t1 actualisé en t
                
                # CF espérés en t pour les chemins dans la monnaie si on n'exerce pas
                continuation_values = RegressionEstimator(X,Y, degree=poly_degree, model_type=model_type).Regression()

                # Exercice anticipé en t si valeur en t est supérieure à la valeur espérée
                exercise = val_intriseque2[in_the_money] > continuation_values
                
                # Mise à jour des CF en t pour les chemins dans la monnaie
                CF_Vect[in_the_money]  = np.where(exercise, val_intriseque2[in_the_money], discounted_CF_next[in_the_money])
        
        # Valeur en t0
        CF_t0 = CF_Vect * np.exp(-market.taux_interet * self.option.maturity / brownian.nb_step)
        
        if not antithetic:
            print("non antithetic")
            prix = np.mean(CF_t0)
            std_prix = np.std(CF_t0) / np.sqrt(len(CF_t0))
            print("Nb chemins :", len(CF_t0))
            print("Prix min non antithetic:", prix - 2*std_prix)
            print("Prix max non antithetic:", prix + 2*std_prix)
            return (prix, std_prix)
        
        moitie = len(CF_t0) // 2
        print('antithetic')
        CF_vect_final = (CF_t0[:moitie] + CF_t0[moitie:]) / 2
        prix = np.mean(CF_vect_final)
        std_prix = np.std(CF_vect_final) / np.sqrt(len(CF_vect_final))
        print("Nb chemins :", len(CF_t0))
        print("Prix min antithetic:", prix - 2*std_prix)
        print("Prix max antithetic:", prix + 2*std_prix)

        return (prix, std_prix)
