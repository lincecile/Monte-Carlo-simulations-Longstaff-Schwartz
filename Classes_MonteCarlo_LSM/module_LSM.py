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
        print("position_div", position_div)
        print('market.step', brownian.step)
        
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
                print('iin')
                position_div = self.__calcul_position_div(market=market, brownian=brownian)
                print(position_div)
                S_T[:, int(position_div)+1] = S_T[:, int(position_div)+1] - q
                S_T[:, int(position_div) + 2:] = S_T[:, int(position_div) + 1][:, np.newaxis] * np.exp(
                        (taux_interet - sigma**2 / 2) * (timedelta[int(position_div) + 2:] - timedelta[int(position_div) + 1]) + sigma * (
                        W[:, int(position_div) + 2:] - W [:, int(position_div) + 1][:, np.newaxis]))
            
            S_T[:,0] = S0
            print(pd.DataFrame(S_T))
            x = input()
            
            if antithetic:
                print('in antithetic')
                W_neg = -W
                S_T_pos = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W)
                S_T_neg = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W_neg)
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
    
    def payoff_LSM(self, brownian: Brownian, market: DonneeMarche, method='vector'):
        stock_price_paths = self.Price(market, brownian, method=method)
        if self.option.call:
            val_intriseque = np.maximum(stock_price_paths - self.option.prix_exercice, 0.0)
        else:
            val_intriseque = np.maximum(self.option.prix_exercice - stock_price_paths, 0.0)

        # Initialisation des cash flows
        CF_vect = np.zeros(len(stock_price_paths))
        print("vect", CF_vect)
        
        for t in range(brownian.nb_step, 0, -1):
            intrinsic_value = val_intriseque[:,t]
            in_the_money = intrinsic_value > 0
            intrinsic_value_discount = intrinsic_value * np.exp(-market.taux_interet*brownian.step)

            if t < brownian.nb_step - 1:
                continuation_value = np.zeros_like(intrinsic_value)
                if np.sum(in_the_money) > 0:
                    X = stock_price_paths[in_the_money, t]
                    Y = intrinsic_value_discount[in_the_money]

                    estimator = RegressionEstimator(X, Y, degree=2)
                    continuation_value[in_the_money] = estimator.get_estimator(X)

                exercise = (intrinsic_value > continuation_value) & in_the_money
                CF_vect = np.where(exercise, intrinsic_value, CF_vect*np.exp(-market.taux_interet*brownian.step))
    
        CF_vect = CF_vect*np.exp(-market.taux_interet*brownian.step)
        
        return np.mean(CF_vect)
    
    def payoff_LSM3(self, brownian: Brownian, market: DonneeMarche, method='vector', antithetic: bool = False):
        # Générer les trajectoires des prix
        if antithetic:
            stock_price_paths_pos, stock_price_paths_neg = self.Price(market, brownian, method=method, antithetic=antithetic)
            stock_price_paths = np.concatenate([stock_price_paths_pos, stock_price_paths_neg], axis=0)
        else:
            stock_price_paths = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul du payoff final
        if self.option.call:
            final_payoff = np.maximum(0, stock_price_paths[:, -1] - self.option.prix_exercice)
        else:
            final_payoff = np.maximum(self.option.prix_exercice - stock_price_paths[:, -1], 0)
        print("val_intriseque:", final_payoff)
        
        # Pour les options européennes, retourner simplement le val_intriseque actualisé
        if not self.option.americaine:
            prix = np.mean(final_payoff * np.exp(-market.taux_interet * self.option.maturity))
            std_prix = np.std(final_payoff * np.exp(-market.taux_interet * self.option.maturity)) / np.sqrt(len(final_payoff))
            print("Nb chemins :", len(final_payoff))
            print("Prix min :", prix - 2*std_prix)
            print("Prix max :", prix + 2*std_prix)
            return (prix, std_prix)
        
        # Pour les options américaines
        CF_vect = final_payoff.copy()
        
        # Parcourir les étapes à l'envers
        for t in range(brownian.nb_step - 1, 0, -1):
            # Calculer la valeur intrinsèque à l'étape courante
            if self.option.call:
                intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.option.prix_exercice)
            else:
                intrinsic_value = np.maximum(self.option.prix_exercice - stock_price_paths[:, t], 0)
            
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
            print("Nb chemins :", len(CF_vect))
            print("Prix min :", prix - 2*std_prix)
            print("Prix max :", prix + 2*std_prix)
            return (prix, std_prix)

        moitie = len(CF_vect) // 2
        CF_vect_final = (CF_vect[:moitie] + CF_vect[moitie:]) / 2
        prix = np.mean(CF_vect_final)
        std_prix = np.std(CF_vect_final) / np.sqrt(len(CF_vect_final))
        print("Nb chemins :", len(CF_vect))
        print("Prix min :", prix - 2*std_prix)
        print("Prix max :", prix + 2*std_prix)

        return (prix, std_prix)

    def LSM(self, brownian: Brownian, market: DonneeMarche, poly_degree=2, poly_type="standard", method='vector', antithetic: bool=False):
        # Prix du sous-jacent simulé
        if antithetic:
            stock_price_paths_pos, stock_price_paths_neg = self.Price(market, brownian, method=method, antithetic=antithetic)
            Spot_simule = np.concatenate([stock_price_paths_pos, stock_price_paths_neg], axis=0)
        else:
            Spot_simule = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul valeur intrinsèque à chaque pas de temps
        if self.option.call:
            val_intriseque = np.maximum(Spot_simule - self.option.prix_exercice, 0.0)
        else:
            val_intriseque = np.maximum(self.option.prix_exercice - Spot_simule, 0.0)
        
        # Valeur de l'option européenne
        if not self.option.americaine:
            prix = np.mean(val_intriseque[:,-1] * np.exp(-market.taux_interet * self.option.maturity))
            std_prix = np.std(val_intriseque[:,-1] * np.exp(-market.taux_interet * self.option.maturity)) / np.sqrt(len(val_intriseque[:,-1]))
            print("Nb chemins :", len(val_intriseque[:,-1]))
            print("Prix min :", prix - 2*std_prix)
            print("Prix max :", prix + 2*std_prix)
            return (prix, std_prix)
        
        # Matrice des cash flows
        CF = np.zeros_like(Spot_simule)
        CF[:, -1] = val_intriseque[:, -1]  # Cash flows à la maturité, valeur intrinsèque
        
        # Algo LSM
        for t in range(brownian.nb_step - 1, 0, -1): 
            
            # CF en t1 actualisé
            discounted_CF_next = CF[:, t+1] * np.exp(-market.taux_interet * self.option.maturity / brownian.nb_step)

            # Chemins dans la monnaie en t            
            in_the_money = val_intriseque[:, t] > 0
            
            # CF en t1 actualisé en t par défaut
            CF[:, t] = discounted_CF_next

            # Si des chemins sont dans la monnaie en t, on fait la regression
            if np.any(in_the_money):  
                
                X = Spot_simule[in_the_money, t]        # prix du sous jacent en t
                Y = discounted_CF_next[in_the_money]    # CF des chemins dans la monnaie en t1 actualisé en t
                
                # CF espérés en t pour les chemins dans la monnaie si on n'exerce pas
                continuation_values = RegressionEstimator(X, Y, degree=poly_degree, poly_type=poly_type).Regression(X, Y)

                # Exercice anticipé en t si valeur en t est supérieure à la valeur espérée
                exercise = val_intriseque[in_the_money, t] > continuation_values
                
                # Mise à jour des CF en t pour les chemins dans la monnaie
                CF[in_the_money, t] = np.where(exercise, val_intriseque[in_the_money, t], discounted_CF_next[in_the_money])
        
        # Valeur en t0
        CF0 = CF[:, 1] * np.exp(-market.taux_interet * self.option.maturity / brownian.nb_step)
        if not antithetic:
            prix = np.mean(CF0)
            std_prix = np.std(CF0) / np.sqrt(len(CF0))
            print("Nb chemins :", len(CF0))
            print("Prix min non antithetic:", prix - 2*std_prix)
            print("Prix max non antithetic:", prix + 2*std_prix)
            return (prix, std_prix)
        
        moitie = len(CF0) // 2
        CF_vect_final = (CF0[:moitie] + CF0[moitie:]) / 2
        prix = np.mean(CF_vect_final)
        std_prix = np.std(CF_vect_final) / np.sqrt(len(CF_vect_final))
        print("Nb chemins :", len(CF0))
        print("Prix min antithetic:", prix - 2*std_prix)
        print("Prix max antithetic:", prix + 2*std_prix)

        return (prix, std_prix)