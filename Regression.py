# Initialisation du générateur de nombres aléatoires
from numpy.random import default_rng, SeedSequence
seq = SeedSequence()
rng = default_rng(seq)

def gaussian(n_time, n_path, d=1, random_state:np.random.Generator = rng):
            """
            Génère un mouvement brownien avec une distribution gaussienne.
            
            Parameters:
            - n_time : int : Nombre de pas de temps
            - n_path : int : Nombre de trajectoires
            - d : int : Dimension (par défaut 1)
            - random_state : np.random.Generator (optionnel) : Générateur aléatoire
            
            Returns:
            - np.ndarray : Tableau de nombres générés selon une loi normale
            """
            gauss = random_state.standard_normal((d, n_time, n_path))
            return gauss