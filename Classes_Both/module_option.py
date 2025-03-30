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
    
    def Price(self, market : DonneeMarche, brownian: Brownian, method: str = 'vector',  antithetic : bool=False) -> float:
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
        q = market.dividende_rate  # Taux de dividende total
        T = self.maturity
        if method == 'vector':
            # Pour la méthode vectorielle
            W, timedelta = brownian.Vecteur()
            S_T = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W) - self.get_dividend(market, brownian, 0, 1)
            S_T[:,0] = S0
            if antithetic:
                W_neg = -W
                S_T_pos = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W) - self.get_dividend(market, brownian, 0, 1)
                S_T_neg = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W_neg) - self.get_dividend(market, brownian, 0, 1)
                S_T_pos[:,0] = S0
                S_T_neg[:,0] = S0
                return S_T_pos, S_T_neg
            else:
                S_T = S0 * np.exp((taux_interet - q - sigma**2/2) * timedelta + sigma * W) - self.get_dividend(market, brownian, 0, 1)
                S_T[:,0] = S0
        else:
            # Pour la méthode scalaire
            S_T = np.ones((brownian.nb_trajectoire,brownian.nb_step+1))*S0
            for i in range(brownian.nb_trajectoire):
                W = brownian.Scalaire() 
                for j in range(1,brownian.nb_step+1):
                    S_T[i,j] = S0*np.exp( (taux_interet - q - sigma**2 / 2)*T + sigma* W[j]) - self.get_dividend( market, brownian, 0, 1)

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
        if self.call:
            val_intriseque =  np.maximum(stock_price_paths - self.prix_exercice, 0.0)
        else:
            val_intriseque =  np.maximum(self.prix_exercice - stock_price_paths, 0.0)

        # Initialisation des cash flows
        CF_vect = np.zeros(len(stock_price_paths))
        # if self.call:
        #     CF_vect = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        # else:
        #     CF_vect = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        print("vect",CF_vect)
        
        for t in range(brownian.nb_step , 0, -1):

            # if self.call:
            #     intrinsic_value = np.maximum(0, stock_price_paths[:, t] - self.prix_exercice)
            # else:
            #     intrinsic_value = np.maximum(self.prix_exercice - stock_price_paths[:, t], 0)
            intrinsic_value = val_intriseque[:,t]
            in_the_money = intrinsic_value > 0
            intrinsic_value_discount = intrinsic_value * np.exp(-market.taux_interet*brownian.step)

            if t < brownian.nb_step - 1:

                continuation_value = np.zeros_like(intrinsic_value)
                if np.sum(in_the_money) > 0:#np.any(in_the_money) and t != brownian.nb_step:  # Vérifie s'il y a des valeurs ITM

                    X = stock_price_paths[in_the_money, t]
                    Y = intrinsic_value_discount[in_the_money]

                    estimator = RegressionEstimator(X, Y, degree=2)
                    continuation_value[in_the_money] = estimator.get_estimator(X)

                #x = input()
                exercise = (intrinsic_value > continuation_value) & in_the_money

                CF_vect = np.where(exercise, intrinsic_value, CF_vect*np.exp(-market.taux_interet*brownian.step))
    
        CF_vect = CF_vect*np.exp(-market.taux_interet*brownian.step)
        
        return np.mean(CF_vect)
    

    # call US ok 1cts d'ecart
    # call euro ok
    # put euro ok

    def payoff_LSM3(self, brownian : Brownian, market: DonneeMarche, method='vector'):
        # Générer les trajectoires des prix
        stock_price_paths = self.Price(market, brownian, method=method)
        print("stock_price_paths:",stock_price_paths)
        # exit()
        # Calcul du val_intriseque final
        if self.call:
            final_payoff = np.maximum(0, stock_price_paths[:, -1] - self.prix_exercice)
        else:
            final_payoff = np.maximum(self.prix_exercice - stock_price_paths[:, -1], 0)
        print("val_intriseque:",final_payoff)
        # exit()
        # Pour les options européennes, retourner simplement le val_intriseque actualisé
        if not self.americaine:
            return np.mean(final_payoff * np.exp(-market.taux_interet * self.maturity))
        
        # Pour les options américaines
        CF_vect = final_payoff.copy()
        
        # Parcourir les étapes à l'envers
        for t in range(brownian.nb_step - 1, 0, -1):
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

        prix = np.mean(CF_vect)
        std_prix = np.std(CF_vect) / np.sqrt(len(CF_vect))
        print("Nb chemins :",len(CF_vect))
        print("Prix min :",prix - 2*std_prix)
        print("Prix max :",prix + 2*std_prix)

        return (prix, std_prix)


    def LSM(self, brownian: Brownian, market: DonneeMarche, poly_degree=2, poly_type="standard", method='vector', antithetic : bool=False):

        # Prix du sous-jacent simulé
        Spot_simule = self.Price(market, brownian, method=method)

        if antithetic:
            stock_price_paths_pos, stock_price_paths_neg = self.Price(market, brownian, method=method, antithetic=antithetic)
            Spot_simule = np.concatenate([stock_price_paths_pos, stock_price_paths_neg], axis=0)
        else:
            Spot_simule = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul valeur intrinsèque à chaque pas de temps
        if self.call:
            val_intriseque = np.maximum(Spot_simule - self.prix_exercice, 0.0)
        else:
            val_intriseque = np.maximum(self.prix_exercice - Spot_simule, 0.0)
        
        # Valeur de l'option européenne
        if not self.americaine:
            # return np.mean(val_intriseque[:,-1] * np.exp(-market.taux_interet * self.maturity))
            prix = np.mean(val_intriseque[:,-1] * np.exp(-market.taux_interet * self.maturity))
            std_prix = np.std(val_intriseque[:,-1] * np.exp(-market.taux_interet * self.maturity)) / np.sqrt(len(val_intriseque[:,-1]))
            print("Nb chemins :",len(val_intriseque[:,-1]))
            print("Prix min :",prix - 2*std_prix)
            print("Prix max :",prix + 2*std_prix)
            return (prix, std_prix)
        
        # Matrice des cash flows
        CF = np.zeros_like(Spot_simule)
        CF[:, -1] = val_intriseque[:, -1]  # Cash flows à la maturité, valeur intrinsèque
        # print(pd.DataFrame(val_intriseque))
        # x = input()
        # Algo LSM
        for t in range(brownian.nb_step - 1, -1, -1): 
            
            # CF en t1 actualisé
            discounted_CF_next = CF[:, t+1] * np.exp(-market.taux_interet * self.maturity / brownian.nb_step)
            print("discounted_CF_next",discounted_CF_next, CF[:, t+1])
            # x = input()
            # Chemins dans la monnaie en t            
            in_the_money = val_intriseque[:, t] > 0
            
            # CF en t1 actualisé en t par défaut
            CF[:, t] = discounted_CF_next
            # Si des chemins sont dans la monnaie en t, on fait la regression
            if np.any(in_the_money):  
                
                X = Spot_simule[in_the_money, t]        # prix du sous jacent en t
                Y = discounted_CF_next[in_the_money]    # CF des chemins dans la monnaie en t1 actualisé en t
                
                # CF espérés en t pour les chemins dans la monnaie si on n'exerce pas
                continuation_values = RegressionEstimator(X, Y, degree=2,poly_type=poly_type).Regression(X,Y)

                # Exercice anticipé en t si valeur en t est supérieure à la valeur espérée
                exercise = val_intriseque[in_the_money, t] > continuation_values
                
                # Mise à jour des CF en t pour les chemins dans la monnaie
                CF[in_the_money, t] = np.where(exercise, val_intriseque[in_the_money, t], discounted_CF_next[in_the_money])
        
        # Valeur en t0
        CF0 = CF[:, 0]
        if not antithetic:
            prix = np.mean(CF0)
            std_prix = np.std(CF0) / np.sqrt(len(CF0))
            print("Nb chemins :",len(CF0))
            print("Prix min non antithetic:",prix - 2*std_prix)
            print("Prix max non antithetic:",prix + 2*std_prix)
        
        moitie = len(CF0) // 2
        CF_vect_final = (CF0[:moitie] + CF0[moitie:]) / 2
        prix = np.mean(CF_vect_final)
        std_prix = np.std(CF_vect_final) / np.sqrt(len(CF_vect_final))
        print("Nb chemins :",len(CF0))
        print("Prix min antithetic:",prix - 2*std_prix)
        print("Prix max antithetic:",prix + 2*std_prix)

        return (prix, std_prix)

    def LSM2(self, brownian: Brownian, market: DonneeMarche, poly_degree=2, poly_type="standard", method='vector'):
        # Prix du sous-jacent simulé
        Spot_simule = self.Price(market, brownian, method=method)
        
        # Calcul valeur intrinsèque à chaque pas de temps
        if self.call:
            val_intriseque = np.maximum(Spot_simule[:,-1] - self.prix_exercice, 0.0)
        else:
            val_intriseque = np.maximum(self.prix_exercice - Spot_simule[:,-1], 0.0)
        
        # Valeur de l'option européenne
        if not self.americaine:
            return np.mean(val_intriseque * np.exp(-market.taux_interet * self.maturity))
        
        # Matrice des cash flows
        CF = np.zeros_like(Spot_simule)
        CF[:, -1] = val_intriseque  # Cash flows à la maturité, valeur intrinsèque
        CF_Vect = val_intriseque.copy()

        # Algo LSM
        for t in range(brownian.nb_step - 1, -1, -1): 

            if self.call:
                print('innnn')
                val_intriseque = np.maximum(Spot_simule[:,t] - self.prix_exercice, 0.0)
                val_intriseque1 = np.maximum(Spot_simule[:,t+1] - self.prix_exercice, 0.0)
                print('val_intriseque1',val_intriseque1)
            else:
                val_intriseque = np.maximum(self.prix_exercice - Spot_simule[:,t], 0.0)
                val_intriseque1 = np.maximum(self.prix_exercice - Spot_simule[:,t+1], 0.0)
            
            # CF en t1 actualisé
            discounted_CF_next = val_intriseque1 * np.exp(-market.taux_interet * self.maturity / brownian.nb_step)
            print("val_intriseque1",val_intriseque1)
            x = input()
            # Chemins dans la monnaie en t
            in_the_money = val_intriseque > 0

            # Si des chemins sont dans la monnaie en t, on fait la regression
            if np.any(in_the_money):  
                
                X = Spot_simule[in_the_money, t]        # prix du sous jacent en t
                Y = discounted_CF_next[in_the_money]    # CF des chemins dans la monnaie en t1 actualisé en t
                
                # CF espérés en t pour les chemins dans la monnaie si on n'exerce pas
                continuation_values = RegressionEstimator(X, Y, degree=2,poly_type=poly_type).Regression(X,Y)

                # Exercice anticipé en t si valeur en t est supérieure à la valeur espérée
                exercise = val_intriseque[in_the_money] > continuation_values
                
                # Mise à jour des CF en t pour les chemins dans la monnaie
                CF_Vect[in_the_money] = np.where(exercise, val_intriseque[in_the_money], discounted_CF_next[in_the_money])
        
        
        # Valeur en t0
        CF0 = CF_Vect
        prix = np.mean(CF0)
        std_prix = np.std(CF0) / np.sqrt(len(CF0))
        print("Nb chemins :",len(CF0))
        print("Prix min :",prix - 2*std_prix)
        print("Prix max :",prix + 2*std_prix)

        return (prix, std_prix)