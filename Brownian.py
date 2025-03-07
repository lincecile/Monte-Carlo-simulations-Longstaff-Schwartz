import numpy as np
import scipy.stats as stats
import pandas as pd
#from Option import Option


class Brownian:
    def __init__(self, n , N, seed):
        self.n = n
        self.N = N
        self.step = 1 / n
        self.seed = seed
        self._generator : np.random.Generator = np.random.default_rng(self.seed)

        # à mettre ici pour générer les mêmes mouvements browniens ??
        #self.uniform_samples = self._generator.uniform(0, 1, (self.n, self.N))
        #self.gaussian_increments = stats.norm.ppf(self.uniform_samples) * np.sqrt(self.step)

    def Scalaire(self):
        #np.random.seed(42)
        #W0 = 0 valeur initiale du mouvement Brownien standard
        W = np.zeros(self.n+1) # matrice des valeurs du mouvement Brownien
        # NPV chaque branche
        for i in range(1,self.n+1): # n pas representés par les lignes de la matrice
            uniform_samples = self._generator.uniform(0, 1)
            W[i] = W[i-1]+stats.norm.ppf(uniform_samples) * np.sqrt(self.step)
        
        # print(pd.DataFrame(W))
        # x = input("Press Enter to continue...")
        return W


    def Vecteur(self):
        # Génération vectorielle des mouvements browniens
        uniform_samples = self._generator.uniform(0, 1, (self.N,self.n)) 
        Z = stats.norm.ppf(uniform_samples)
        dW = np.sqrt(self.step) * Z

        # Construction du mouvement brownien
        W = np.zeros((self.N, self.n+1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        
        self._generator : np.random.Generator = np.random.default_rng(self.seed)

        return W


