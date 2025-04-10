import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from matplotlib.gridspec import GridSpec
import seaborn as sns
from Classes_Both.module_option import Option
from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from copy import deepcopy


class LSMGraph:
    """
    Classe pour la visualisation graphique des résultats des méthodes de pricing d'options
    par la méthode Least Square Monte Carlo (LSM).
    """
    
    def __init__(self, option : Option, market : DonneeMarche):
        """
        Initialisation de la classe LSMGraph.
        
        Args:
            option: Instance d'Option à évaluer
            market: Instance de DonneeMarche avec les paramètres du marché
        """
        self.option = deepcopy(option)
        self.market = deepcopy(market)
        self.period = (option.maturite - option.date_pricing).days / 365
        
        # Configuration des styles pour les graphiques
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.markers = ['o', 's', '^', 'D', 'x', '*', '+', 'v', '<', '>']
        self.line_styles = ['-', '--', '-.', ':']
        
        # Style Seaborn pour de meilleurs graphiques
        sns.set_style("whitegrid")
        self.palette = sns.color_palette("muted")
        
    def afficher_trajectoires_prix(self, trajectoires, brownian : Brownian, nb_trajectoires=5):
        """
        Affiche les trajectoires de prix pour un nombre limité de simulations.
        
        Args:
            trajectoires: Array numpy contenant les trajectoires de prix
            brownian: Instance de Brownian utilisée pour générer les trajectoires
            nb_trajectoires: Nombre de trajectoires à afficher (défaut: 5)
            
        Returns:
            fig: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Obtenir les dates/temps et sélectionner des trajectoires aléatoires
        times = np.linspace(0, self.period, brownian.nb_step + 1)
        indices = np.random.choice(trajectoires.shape[0], min(nb_trajectoires, trajectoires.shape[0]), replace=False)
        
        for i, idx in enumerate(indices):
            color = self.colors[i % len(self.colors)]
            ax.plot(times, trajectoires[idx, :], color=color, alpha=0.8, 
                   label=f"Trajectoire {idx+1}")
        
        # Ajout du prix spot initial
        ax.axhline(y=self.market.prix_spot, color='black', linestyle='--', 
                  label=f"Prix spot initial ({self.market.prix_spot:.2f})")
        
        # Ajout du strike
        ax.axhline(y=self.option.prix_exercice, color='red', linestyle='--', 
                  label=f"Prix d'exercice ({self.option.prix_exercice:.2f})")
        
        ax.set_xlabel("Temps (années)")
        ax.set_ylabel("Prix du sous-jacent")
        ax.set_title("Simulation de trajectoires de prix")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def afficher_mouvements_browniens(self, brownian : Brownian, nb_trajectoires=5):

        """
        Affiche les mouvements browniens pour un nombre limité de simulations.
        
        Args:
            brownian: Instance de Brownian
            nb_trajectoires: Nombre de trajectoires à afficher (défaut: 5)
            
        Returns:
            fig: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = np.linspace(0, self.period, brownian.nb_step + 1)
        
        # Générer tous les mouvements browniens en une seule fois
        brownian_paths = brownian.Vecteur()
        
        # Sélectionner aléatoirement nb_trajectoires parmi celles générées
        indices = np.random.choice(brownian_paths.shape[0], 
                                min(nb_trajectoires, brownian_paths.shape[0]), 
                                replace=False)
        
        for i, idx in enumerate(indices):
            color = self.colors[i % len(self.colors)]
            ax.plot(times, brownian_paths[idx, :], color=color, alpha=0.8, 
                label=f"Brownien {idx+1}")
        
        ax.axhline(y=0, color='black', linestyle='--', label="Niveau 0")
        
        ax.set_xlabel("Temps (années)")
        ax.set_ylabel("Valeur du mouvement brownien")
        ax.set_title("Simulation de mouvements browniens")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    
    def comparer_methodes(self, methods=None, nb_paths=1000, nb_steps=200):
        """
        Compare différentes méthodes de calcul du prix d'une option.
        
        Args:
            methods: Liste des méthodes à comparer ['vector', 'scalar']
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas de temps pour la simulation
            
        Returns:
            fig: Figure matplotlib
        """
        if methods is None:
            methods = ["vector", "scalar"]
            
        prices = []
        std_devs = []
        execution_times = []

        for method in methods:
            pricer = LSM_method(self.option)
            brownian = Brownian(self.period, nb_steps, nb_paths, 1)
            start_time = time.time()
            price, std_error, interval = pricer.LSM(brownian, self.market, method=method)
            end_time = time.time()
            execution_time = end_time - start_time
            
            execution_times.append(execution_time)
            prices.append(price)
            std_devs.append(std_error)

        prices = np.array(prices)
        std_devs = np.array(std_devs)

        # Calcul des intervalles de confiance à 95% (± 2 écart-types)
        lower_bound = prices - 2 * std_devs
        upper_bound = prices + 2 * std_devs
        legend_labels = [f"{method} ({time:.2f}s)" for method, time in zip(methods, execution_times)]

        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(methods, prices, yerr=2 * std_devs, fmt='o', color='blue', 
                   ecolor='blue', elinewidth=1, capsize=5, markersize=8)
        ax.axhline(y=np.mean(prices), color='black', linestyle='dashed', 
                  label=f"Prix moyen: {np.mean(prices):.4f}")
        
        ax.set_xlabel("Méthodes")
        ax.set_ylabel("Prix")
        ax.set_title("Intervalles de confiance des prix (± 2 écart-types)")
        ax.legend(title="Méthodes (Temps en s)", labels=legend_labels)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def comparer_convergence_paths(self, reference_price=None, methods=None, nb_points=20, max_paths=100000):
        """
        Compare la convergence du prix en fonction du nombre de paths pour différentes méthodes.
        
        Args:
            reference_price: Prix de référence (ex: price_BS ou price_tree)
            methods: Liste des méthodes à comparer ['vector', 'scalar']
            nb_points: Nombre de points sur le graphique
            max_paths: Nombre maximum de paths
            
        Returns:
            fig: Figure matplotlib
        """
        if methods is None:
            methods = ["scalar", "vector"]

        # Générer une échelle logarithmique pour le nombre de paths
        paths = np.logspace(1, np.log10(max_paths), num=nb_points, dtype=int)
        
        results = {}
        for method in methods:
            method_prices = []
            method_std_devs = []
            method_times = []
            
            print(f"Traitement de la méthode: {method}")
            for path in paths:
                pricer = LSM_method(self.option)
                brownian = Brownian(self.period, 200, path, 1)
                
                start_time = time.time()
                price, std_error, _ = pricer.LSM(brownian, self.market, method=method)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                method_prices.append(price)
                method_std_devs.append(std_error)
                method_times.append(execution_time)
            
            results[method] = {
                'prices': method_prices,
                'std_devs': method_std_devs,
                'times': method_times
            }

        # Création du graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        for i, method in enumerate(methods):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            avg_time = np.mean(results[method]['times'])
            
            # Graphique des prix
            ax1.errorbar(paths, results[method]['prices'], 
                        yerr=2 * np.array(results[method]['std_devs']),
                        fmt=f'-{marker}', color=color, ecolor=color, elinewidth=1, 
                        capsize=3, label=f"{method} (Moy: {avg_time:.2f}s)")
            
            # Graphique des temps d'exécution
            ax2.plot(paths, results[method]['times'], f'-{marker}', color=color,
                    label=f"{method}")

        # Si un prix de référence est fourni
        if reference_price is not None:
            ax1.axhline(y=reference_price, color='black', linestyle='dashed', 
                      label=f"Prix de référence ({reference_price:.4f})")

        # Ajustements graphique des prix
        ax1.set_xscale("log")  # Échelle logarithmique sur X
        ax1.set_xlabel("Nombre de chemins (échelle log)")
        ax1.set_ylabel("Prix de l'option")
        ax1.set_title("Évolution du prix en fonction du nombre de chemins (± 2 écart-types)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Ajustements graphique des temps d'exécution
        ax2.set_xscale("log")  # Échelle logarithmique sur X
        ax2.set_yscale("log")  # Échelle logarithmique sur Y pour les temps
        ax2.set_xlabel("Nombre de chemins (échelle log)")
        ax2.set_ylabel("Temps d'exécution (s) (échelle log)")
        ax2.set_title("Temps d'exécution en fonction du nombre de chemins")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def comparer_convergence_pas(self, reference_price=None, methods=None, nb_points=20, nb_paths=10000, max_steps=500):
        """
        Compare la convergence du prix en fonction du nombre de pas pour différentes méthodes.
        
        Args:
            reference_price: Prix de référence (ex: price_BS ou price_tree)
            methods: Liste des méthodes à comparer ['vector', 'scalar']
            nb_points: Nombre de points sur le graphique
            nb_paths: Nombre de paths pour la simulation
            max_steps: Nombre maximum de pas
            
        Returns:
            fig: Figure matplotlib
        """
        if methods is None:
            methods = ["scalar", "vector"]

        # Générer une séquence de pas
        steps = np.linspace(5, max_steps, num=nb_points, dtype=int)
        
        results = {}
        for method in methods:
            method_prices = []
            method_std_devs = []
            method_times = []
            
            print(f"Traitement de la méthode: {method}")
            for step in steps:
                pricer = LSM_method(self.option)
                brownian = Brownian(self.period, step, nb_paths, 1)
                
                start_time = time.time()
                price, std_error, _ = pricer.LSM(brownian, self.market, method=method)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                method_prices.append(price)
                method_std_devs.append(std_error)
                method_times.append(execution_time)
            
            results[method] = {
                'prices': method_prices,
                'std_devs': method_std_devs,
                'times': method_times
            }

        # Création du graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        for i, method in enumerate(methods):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            avg_time = np.mean(results[method]['times'])
            
            # Graphique des prix
            ax1.errorbar(steps, results[method]['prices'], 
                        yerr=2 * np.array(results[method]['std_devs']),
                        fmt=f'-{marker}', color=color, ecolor=color, elinewidth=1, 
                        capsize=3, label=f"{method} (Moy: {avg_time:.2f}s)")
            
            # Graphique des temps d'exécution
            ax2.plot(steps, results[method]['times'], f'-{marker}', color=color,
                    label=f"{method}")

        # Si un prix de référence est fourni
        if reference_price is not None:
            ax1.axhline(y=reference_price, color='black', linestyle='dashed', 
                      label=f"Prix de référence ({reference_price:.4f})")

        # Ajustements graphique des prix
        ax1.set_xlabel("Nombre de pas")
        ax1.set_ylabel("Prix de l'option")
        ax1.set_title("Évolution du prix en fonction du nombre de pas (± 2 écart-types)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Ajustements graphique des temps d'exécution
        ax2.set_xlabel("Nombre de pas")
        ax2.set_ylabel("Temps d'exécution (s)")
        ax2.set_title("Temps d'exécution en fonction du nombre de pas")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def comparer_seeds(self, nb_seeds=10, nb_paths=10000, nb_steps=200):
        """
        Compare les résultats pour différentes valeurs de seed.
        
        Args:
            nb_seeds: Nombre de seeds à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure matplotlib
        """
        seeds = np.arange(1, nb_seeds + 1)
        prices = []
        std_devs = []
        
        for seed in seeds:
            pricer = LSM_method(self.option)
            brownian = Brownian(self.period, nb_steps, nb_paths, seed)
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector')
            
            prices.append(price)
            std_devs.append(std_error)
        
        prices = np.array(prices)
        std_devs = np.array(std_devs)
        
        # Calcul des intervalles de confiance à 95% (± 2 écart-types)
        lower_bound = prices - 2 * std_devs
        upper_bound = prices + 2 * std_devs
        
        mean_price = np.mean(prices)
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.errorbar(seeds, prices, yerr=2 * std_devs, fmt='o', color='blue', 
                   ecolor='blue', elinewidth=1, capsize=5, markersize=8)
        
        ax.axhline(y=mean_price, color='red', linestyle='--', 
                  label=f"Prix moyen: {mean_price:.4f} ± {np.mean(std_devs):.4f}")
        
        ax.fill_between(seeds, lower_bound, upper_bound, color='blue', alpha=0.2)
        
        ax.set_xlabel("Valeur de la seed")
        ax.set_ylabel("Prix de l'option")
        ax.set_title("Influence de la seed sur le prix (± 2 écart-types)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def comparer_polynomes(self, reference_price=None, poly_degrees=None, poly_types=None, 
                          nb_paths=10000, nb_steps=200):
        """
        Compare les résultats pour différents types et degrés de polynômes.
        
        Args:
            reference_price: Prix de référence (ex: price_BS ou price_tree)
            poly_degrees: Liste des degrés polynomiaux à tester
            poly_types: Liste des types de polynômes à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure matplotlib
            df: DataFrame avec les résultats
        """
        if poly_degrees is None:
            poly_degrees = [2, 3, 4, 5, 6, 7]
        
        if poly_types is None:
            poly_types = ["Polynomial", "Laguerre", "Hermite", "Linear", 
                         "Logarithmic", "Exponential"]
        
        results = {}
        
        # Test des degrés polynomiaux (pour le type "polynomial")
        for degree in poly_degrees:
            brownian = Brownian(self.period, nb_steps, nb_paths, 1)
            pricer = LSM_method(self.option)
            
            start_time = time.time()
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                           antithetic=False, poly_degree=degree, 
                                           model_type="Polynomial")
            end_time = time.time()
            execution_time = end_time - start_time
            
            key = f"Polynomial (deg={degree})"
            results[key] = {
                'price': price, 
                'std_error': std_error,
                'time': execution_time
            }
            
        # Test des types de modèles (avec degré polynomial fixé à 2)
        for poly_type in poly_types:
            if poly_type == "Polynomial":
                continue  # Déjà testé ci-dessus avec différents degrés
                
            brownian = Brownian(self.period, nb_steps, nb_paths, 1)
            pricer = LSM_method(self.option)
            
            start_time = time.time()
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                           antithetic=False, poly_degree=2, 
                                           model_type=poly_type)
            end_time = time.time()
            execution_time = end_time - start_time
            
            results[poly_type] = {
                'price': price, 
                'std_error': std_error,
                'time': execution_time
            }
        
        # Création d'un DataFrame pour visualisation
        df_results = pd.DataFrame({
            'Model': list(results.keys()),
            'Price': [results[k]['price'] for k in results],
            'Std Error': [results[k]['std_error'] for k in results],
            'Time (s)': [results[k]['time'] for k in results]
        })
        
        # Création du graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        models = df_results['Model']
        prices = df_results['Price']
        errors = df_results['Std Error']
        times = df_results['Time (s)']
        
        # Graphique des prix avec barres d'erreur
        for i, (model, price, error) in enumerate(zip(models, prices, errors)):
            color = self.colors[i % len(self.colors)]
            ax1.errorbar(i, price, yerr=2*error, fmt='o', color=color, 
                        ecolor=color, capsize=5, markersize=8)
        
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        if reference_price is not None:
            ax1.axhline(y=reference_price, color='red', linestyle='dashed', 
                       label=f"Prix de référence: {reference_price:.4f}")
        else:
            ax1.axhline(y=np.mean(prices), color='red', linestyle='dashed', 
                       label=f"Prix moyen: {np.mean(prices):.4f}")
        
        ax1.set_ylabel('Prix de l\'option')
        ax1.set_title('Comparaison des prix par type/degré de modèle (± 2 écart-types)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Graphique des temps d'exécution
        bar_positions = range(len(models))
        ax2.bar(bar_positions, times, color=self.colors[:len(models)])
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Temps d\'exécution (s)')
        ax2.set_title('Temps d\'exécution par type/degré de modèle')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        return fig, df_results
    
    def comparer_degres_par_type(self, poly_types=None, poly_degrees=None, 
                               nb_paths=10000, nb_steps=200):
        """
        Compare les résultats pour différents degrés pour chaque type de polynôme.
        
        Args:
            poly_types: Liste des types de polynômes à tester
            poly_degrees: Liste des degrés polynomiaux à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure matplotlib
            df: DataFrame avec les résultats
        """
        if poly_types is None:
            poly_types = ["Polynomial", "Laguerre", "Hermite"]
        
        if poly_degrees is None:
            poly_degrees = [2, 3, 4, 5, 6]
        
        results = {}
        
        for poly_type in poly_types:
            for degree in poly_degrees:
                brownian = Brownian(self.period, nb_steps, nb_paths, 1)
                pricer = LSM_method(self.option)
                
                try:
                    start_time = time.time()
                    price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                                   antithetic=False, poly_degree=degree, 
                                                   model_type=poly_type)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    key = f"{poly_type} (deg={degree})"
                    results[key] = {
                        'type': poly_type,
                        'degree': degree,
                        'price': price, 
                        'std_error': std_error,
                        'time': execution_time
                    }
                except Exception as e:
                    print(f"Erreur pour {poly_type} degré {degree}: {e}")
        
        # Création d'un DataFrame pour visualisation
        df_results = pd.DataFrame([results[k] for k in results])
        
        # Création du graphique
        fig, axes = plt.subplots(len(poly_types), 1, figsize=(12, 5*len(poly_types)))
        
        if len(poly_types) == 1:
            axes = [axes]
        
        for i, poly_type in enumerate(poly_types):
            type_df = df_results[df_results['type'] == poly_type]
            
            ax = axes[i]
            degrees = type_df['degree']
            prices = type_df['price']
            errors = type_df['std_error']
            
            ax.errorbar(degrees, prices, yerr=2*errors.values, fmt='o-', color=self.colors[i], 
                       ecolor=self.colors[i], capsize=5, markersize=8)
            
            ax.set_xlabel('Degré polynomial')
            ax.set_ylabel('Prix de l\'option')
            ax.set_title(f'Résultats pour le type {poly_type}')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, df_results
    
    