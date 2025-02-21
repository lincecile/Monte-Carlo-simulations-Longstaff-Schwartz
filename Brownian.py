import numpy as np
import scipy.stats as stats
import random
import pandas as pd
#from Option import Option
np.random.seed(42)

class Brownian:
    def __init__(self, n , N):
        self.n = n
        self.N = N
        self.step = 1 / n

    def Scalaire(self):
        #W0 = 0 valeur initiale du mouvement Brownien standard
        W = np.zeros(self.n+1) # matrice des valeurs du mouvement Brownien

        # NPV chaque branche
        for i in range(1,self.n+1): # n pas representés par les lignes de la matrice
            uniform_samples = np.random.uniform(0, 1)
            W[i] = W[i-1]+stats.norm.ppf(uniform_samples) * np.sqrt(self.step)
            
        return W


    def Vecteur(self):
        # Génération vectorielle des mouvements browniens
        uniform_samples = np.random.uniform(0, 1, (self.n, self.N))
        Z = stats.norm.ppf(uniform_samples)
        dW = np.sqrt(self.step) * Z

        # Construction du mouvement brownien
        W = np.zeros((self.n+1, self.N))
        W[1:, :] = np.cumsum(dW, axis=0)
        return W