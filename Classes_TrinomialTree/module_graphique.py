#%% Imports

import networkx as nx
import plotly.graph_objects as go

    
from Classes_TrinomialTree.module_arbre_noeud import Arbre
  
  
#%% Classes  
    
class ArbreGraph :
    def __init__(self, arbre : Arbre, dimension_chart : tuple = (2000, 1500)):
        """Initialisation de la classe Arbre_Graph

        Args:
            arbre (Arbre): l'arbre pour lequel nous souhaitons réaliser le graphique
            dimension_chart (tuple, optional): la dimension de la fen^tre de sortie de notre graphique. Défaut à (16, 12).
        """
        self.arbre = arbre
        self.dimension_chart = dimension_chart
         
        #Dans le cas où l'arbre fournit en input n'aurait pas été pricé
        if self.arbre.prix_option is None : 
            self.arbre.pricer_arbre()
            
        if self.arbre.nb_pas > 100 : 
            raise ValueError("Nombre de pas dans l'arbre trop important pour être affiché. Veuillez choisir un nombre de pas inférieur à 10.")
        
    def afficher_arbre(self) -> None:
        """Fonction nous permettant de réaliser le graphique en positionnant les nœuds selon le prix sous-jacent."""
        
        # Initialisation de l'objet issu de la librairie networkx
        G = nx.DiGraph()
        labels = {}
        positions = {}
        queue = [self.arbre.racine]
        
        # le niveau de la barrière
        niveau_barriere = self.arbre.option.barriere.niveau_barriere

        # Déterminer le prix max et min des sous-jacents pour ajuster l'échelle
        prix_min, prix_max = float('inf'), float('-inf')
        
        while queue:
            noeud = queue.pop(0)
            
            # Utilisation du prix sous-jacent pour la coordonnée y
            y = noeud.prix_sj

            # Mise à jour des limites des prix
            prix_min = min(prix_min, y)
            prix_max = max(prix_max, y)
            
            # Création d'un label pour chaque nœud
            if not (noeud.valeur_intrinseque == 0 and noeud.prix_sj == 0 and noeud.p_cumule == 0):
                noeud_label = f"Valeur intrinsèque : {noeud.valeur_intrinseque:.2f}<br>Prix sous-jacent : {noeud.prix_sj:.2f}<br>Probabilité cumulée : {noeud.p_cumule:.6f}"
                labels[noeud] = noeud_label
                positions[noeud] = (noeud.position_arbre, y)
                G.add_node(noeud)
            
            # Itération sur chaque futur nœud
            for direction, futur in zip(["bas", "centre", "haut"], [noeud.futur_bas, noeud.futur_centre, noeud.futur_haut]):
                if futur is not None:
                    # Probabilité correspondante à chaque direction
                    if direction == "bas":
                        prob = noeud.p_bas
                    elif direction == "centre":
                        prob = noeud.p_mid
                    elif direction == "haut":
                        prob = noeud.p_haut
                    
                    # Formatage de la probabilité
                    prob_label = f"{prob:.4f}"
                    
                    # Ajout de la ligne avec la probabilité en label
                    if not (futur.valeur_intrinseque == 0 and futur.prix_sj == 0 and futur.p_cumule == 0):
                        G.add_edge(noeud, futur, label=prob_label)
                    
                    # Ajout du nœud futur à la queue pour itérer dessus 
                    if futur not in queue: 
                        queue.append(futur)

        # Ajuster les limites de l'axe des y selon les prix sous-jacents
        y_margin = (prix_max - prix_min) * 0.05  # Ajout d'une marge de 10% en haut et en bas

        # Dessiner les nœuds et leurs labels
        nx.draw_networkx_nodes(G, pos=positions, node_size=2500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=10)
        
        # Extraction des labels des lignes
        liaison_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edges(G, pos=positions, arrows=True, arrowstyle='-|>', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=liaison_labels, font_color='red', font_size=8, label_pos=0.3)
            
        # Preparation des liaison pour plotly
        liaison_x = []
        liaison_y = []
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            liaison_x += [x0, x1, None]
            liaison_y += [y0, y1, None]


        liaison_trace = go.Scatter(
            x=liaison_x, y=liaison_y,
            mode='lines',
            line=dict(width=1, color='#888'),
        )

        noeud_x = [positions[node][0] for node in G.nodes()]
        noeud_y = [positions[node][1] for node in G.nodes()]
        
        labels_noeuds = [labels[node] for node in G.nodes()]

        noeuds_trace = go.Scatter(
            x=noeud_x, y=noeud_y,
            mode='markers',
            text=labels_noeuds,
            hoverinfo='text',
            marker=dict(
                colorscale='YlGnBu',
                color=[],
                size=15
            ))

        # Ajout d'un titre au graphique, on fait s'adapter le titre à ce qu'on graphe
        
        if self.arbre.option.call : 
            type_option = "call"
        else : 
            type_option = "put"
            
        if self.arbre.option.americaine : 
            exercice_option = "américaine"
        else : 
            exercice_option = "européenne"
            
        strike = self.arbre.option.prix_exercice
        
        barriere_titre=""
        
        if self.arbre.option.barriere.direction_barriere : 
            barriere_titre = f", barrière {self.arbre.option.barriere.type_barriere.value} {self.arbre.option.barriere.direction_barriere.value} {round(self.arbre.option.barriere.niveau_barriere, 2)}"
        
        graphique_titre = f"Arbre Trinomial, option {type_option} {exercice_option}, strike {strike}{barriere_titre}"

        fig = go.Figure(data=[liaison_trace, noeuds_trace],
                        layout=go.Layout(
                        title=f"{graphique_titre}",
                        titlefont_size=30,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
        
        if self.arbre.option.barriere.direction_barriere != None and self.arbre.option.barriere.type_barriere != None : 
            fig.add_shape(
                type='line',
                x0=min(noeud_x),  # Position X de départ
                y0=niveau_barriere,  # position en Y de la barriere
                x1=max(noeud_x),  # Fin en X
                y1=niveau_barriere,  # position en Y de la barriere
                line=dict(
                    color='red',
                    width=2,
                    dash='dash',  # ligne pointillée
                )
            )

        fig.update_layout(width=self.dimension_chart[0], height=self.dimension_chart[1])
        
        return fig
