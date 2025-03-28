import numpy as np
import scipy.stats as stats
import pandas as pd

class Brownian:
    def __init__(self, n , N, seed):
        self.n = n
        self.N = N
        self.step = 1 / n
        self.seed = seed
        self._generator : np.random.Generator = np.random.default_rng(self.seed)

    def Scalaire(self):
        # Mouvement Brownien
        W = np.zeros(self.n+1) 
        for i in range(1,self.n+1): 
            uniform_samples = self._generator.uniform(0, 1)
            W[i] = W[i-1]+stats.norm.ppf(uniform_samples) * np.sqrt(self.step)
        return W


    def Vecteur(self):
        # Génération vectorielle des mouvements browniens
        uniform_samples = self._generator.uniform(0, 1, (self.N,self.n)) 
        Z = stats.norm.ppf(uniform_samples)
        dW = np.sqrt(self.step) * Z

        # Construction du mouvement brownien
        W = np.zeros((self.N, self.n+1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        
        # self._generator = np.random.default_rng(self.seed)

        return W


