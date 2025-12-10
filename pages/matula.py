import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
import time  # AJOUT IMPORT TIME

# استخدام backend غير تفاعلي لـ matplotlib
matplotlib.use('Agg')



# تخصيص الألوان والتصميم باستخدام CSS من Code 1
st.markdown("""
<style>
    .stApp {
        background-color: white;
        font-family: 'Arial', sans-serif;
    }
            
    .stButton>button {
        background-color: darkred;
        color: white;
        border-radius: 18px;
        font-size: 10px;
        padding: 15px 30px;
        width: 170px;
        height:45px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #8B0000;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .title {
        font-size: 20px;
        font-weight: bold;
        color: black;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: darkred;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .content {
        font-size: 18px;
        color: black;
        line-height: 1.6;
    }
    .bullet {
        margin-left: 20px;
    }
    
    .input-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border: 0.5px solid #ccc;
        margin: 10px 0;
        font-size: 14px;
    }
    .checkbox-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 0.5px solid #ddd;
        margin: 10px 0;
        font-size: 14px;
    }
    .matrix-container {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #b3d9f2;
        margin: 10px 0;
    }
    
    /* تخصيص border لحقول إدخال القيم - الحل الصحيح */
    .stTextInput>div>div>input {
        border: 2px solid darkred !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.3) !important;
    }
    
    /* بديل: استخدام data-testid */
    [data-testid="stTextInput"] input {
        border: 2px solid darkred !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.3) !important;
    }
            
.stTextInput label {
    color: darkred !important;
    font-weight: bold !important;
    font-size: 16px !important;
}

/* تخصيص عرض خصائص الشجرة */
.properties-container {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 2px solid darkred;
    margin: 15px 0;
}

.property-item {
    font-size: 16px;
    margin: 8px 0;
    padding: 5px;
    border-left: 4px solid darkred;
    padding-left: 10px;
    background-color: white;
}
    
    /* تخصيص textarea من Code 1 */
    .stTextArea>div>textarea {
        border: 2px solid darkred !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    
    .stTextArea>div>textarea:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.3) !important;
    }
    
    [data-testid="stTextArea"] textarea {
        border: 2px solid darkred !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    
    [data-testid="stTextArea"] textarea:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.3) !important;
    }
    
    .stTextArea label {
        color: darkred !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    /* تخصيص dataframes */
    .stDataFrame {
        border: 2px solid darkred !important;
        border-radius: 8px !important;
    }
    
    /* تخصيص headers */
    h1, h2, h3, h4 {
        color: darkred !important;
    }
    
    /* تخصيص info boxes */
    .stAlert {
        border: 2px solid darkred !important;
        border-radius: 8px !important;
    }
    
    /* تخصيص success messages */
    .stSuccess {
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
    }
    
    /* تخصيص error messages */
    .stError {
        border: 2px solid #f44336 !important;
        border-radius: 8px !important;
    }
    
    /* تخصيص warning messages */
    .stWarning {
        border: 2px solid #ff9800 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

class MatulaVisualization:
    def __init__(self, vertices, adj_matrix):
         # Données d'entrée
        self.vertices = vertices      # Liste des noms de sommets
        self.adj_matrix = adj_matrix  # matrice d'adjacence
        # Structure du graphe 
        self.G = nx.Graph()  # Objet graphe NetworkX
        self.steps = []     # Historique des étapes
        self.coloring_steps = []     # Étapes de coloration
        self.final_colors = {}     # Résultat final des couleurs
        
        # Construire le graphe à partir de la matrice d'adjacence
        self.build_graph()
    
    def build_graph(self):
        """Construit le graphe à partir de la matrice d'adjacence"""
        self.G.add_nodes_from(self.vertices)
        n = len(self.vertices)
        
        for i in range(n):
            for j in range(i+1, n):  # Pour un graphe non orienté
                if self.adj_matrix[i][j] == 1:
                    self.G.add_edge(self.vertices[i], self.vertices[j])
    
    def run_matula_algorithm(self):
        """Exécute l'algorithme de Matula étape par étape"""
        st.markdown('<div class="section-title">🔍 Étape 1: Calcul des Degrés Initiaux</div>', unsafe_allow_html=True)
        
        # Calcul des degrés initiaux
        initial_degrees = dict(self.G.degree())
        
        # Afficher les degrés dans un conteneur stylisé
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        deg_text = " | ".join([f"<strong>{sommet}: {deg}</strong>" for sommet, deg in initial_degrees.items()])
        st.markdown(f"**Degrés initiaux :** {deg_text}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🔄 Étape 2: Smallest-Last Ordering</div>', unsafe_allow_html=True)
        
        # Phase de suppression
        temp_graph = self.G.copy()
        classement = []
        step_count = 1
        
        # Tableau pour afficher les étapes
        steps_data = []
        
        while temp_graph.nodes():
            # Calculer les degrés actuels
            current_degrees = dict(temp_graph.degree())
            
            # Trouver le sommet de plus petit degré
            min_degree = min(current_degrees.values())
            min_vertices = [v for v, deg in current_degrees.items() if deg == min_degree]
            min_vertex = sorted(min_vertices)[0]
            
            # Ajouter le sommet au classement
            classement.append(min_vertex)
            
            # Enregistrer l'étape
            steps_data.append({
                'Étape': step_count,
                'Graphe Restant': ', '.join(sorted(temp_graph.nodes())),
                'Degrés': str(current_degrees),
                'Sommet Supprimé': min_vertex,
                'Classement': ', '.join(classement)
            })
            
            # Mettre à jour le graphe temporaire
            temp_graph.remove_node(min_vertex)
            step_count += 1
        
        # Afficher le tableau des étapes
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        steps_df = pd.DataFrame(steps_data)
        st.dataframe(steps_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Inversion du classement
        st.markdown('<div class="section-title">📋Inversion du Classement</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.subheader("Classement Original")
            st.markdown(f"**Ordre de suppression :** {' → '.join(classement)}")
            st.info("Cet ordre représente l'ordre dans lequel les sommets ont été supprimés (du plus petit degré au plus grand)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.subheader("Classement Inversé")
            classement_inverse = list(reversed(classement))
            st.markdown(f"**Ordre de coloration :** {' → '.join(classement_inverse)}")
            st.success("Cet ordre sera utilisé pour la coloration gloutonne (du dernier supprimé au premier supprimé)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🎨 Étape 3: Coloration Gloutonne</div>', unsafe_allow_html=True)
        
        # Phase de coloration - INVERSE du classement
        colors = {}
        color_names = ["Rouge", "Bleu", "Vert", "Jaune", "Orange", "Violet", "Rose", "Marron"]
        coloring_data = []
        
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.info(f"**Coloration suivant l'ordre inversé :** {' → '.join(classement_inverse)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, vertex in enumerate(classement_inverse):
            # Trouver les couleurs des voisins
            used_colors = set()
            for neighbor in self.G.neighbors(vertex):
                if neighbor in colors:
                    used_colors.add(colors[neighbor])
            
            # Attribuer la plus petite couleur disponible
            color = 0
            while color in used_colors:
                color += 1
            
            colors[vertex] = color
            
            # Enregistrer l'étape de coloration
            colored_neighbors = [f"{n}({color_names[colors[n]]})" 
                               for n in self.G.neighbors(vertex) if n in colors]
            
            coloring_data.append({
                'Étape Coloration': i + 1,
                'Sommet': vertex,
                'Couleur Attribuée': color_names[color],
                'Voisins Colorés': ', '.join(colored_neighbors) if colored_neighbors else "Aucun"
            })
        
        # Afficher les étapes de coloration
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        coloring_df = pd.DataFrame(coloring_data)
        st.dataframe(coloring_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        self.final_colors = colors
        return colors, classement, classement_inverse
    
    def visualize_graph(self, colors=None, highlight_node=None, title="Graphe"):
        """Visualise le graphe avec matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Définir les positions des nœuds
        pos = nx.spring_layout(self.G, seed=42)
        
        # Définir les couleurs
        color_palette = ['darkred', 'navy', 'darkgreen', 'goldenrod', 'darkorange', 'purple', 'deeppink', 'saddlebrown']
        
        if colors:
            node_colors = [color_palette[colors[node] % len(color_palette)] for node in self.G.nodes()]
        else:
            node_colors = ['lightblue' for _ in self.G.nodes()]
        
        # Dessiner le graphe
        nx.draw_networkx_nodes(self.G, pos, 
                              node_color=node_colors,
                              node_size=1500, 
                              ax=ax,
                              edgecolors='black',
                              linewidths=2)
        
        nx.draw_networkx_edges(self.G, pos, ax=ax, width=3, edge_color='gray')
        
        # Dessiner les labels avec contraste
        nx.draw_networkx_labels(self.G, pos, ax=ax, 
                               font_size=14, 
                               font_weight='bold',
                               font_color='white')
        
        ax.set_title(title, fontsize=18, fontweight='bold', color='darkred')
        ax.axis('off')
        plt.tight_layout()
        
        return fig

def main():
    st.set_page_config(page_title="Algorithme Matula", page_icon="🎨", layout="wide")
    
    # Titre avec style
    st.markdown('<div class="title">🎨 Algorithme de Matula - Coloration de Graphes</div>', unsafe_allow_html=True)
    
    # === CHAMPS DANS LA PAGE PRINCIPALE ===
    st.markdown('<div class="section-title">📊 Saisie du Graphe</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        # Exemple pré-rempli
        default_vertices = "A, B, C, D, E, F"
        vertices_input = st.text_area("Sommets (séparés par des virgules):", 
                                     value=default_vertices, height=120)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== MATRICE D'ADJACENCE =====
    if vertices_input:
        vertices = [v.strip() for v in vertices_input.split(',') if v.strip()]
        
        if vertices:
            st.success(f"✅ Sommets définis: {', '.join(vertices)}")
            
            # Initialiser la matrice d'adjacence
            n = len(vertices)
            if 'matula_adj_matrix' not in st.session_state or st.session_state.get('prev_vertices') != vertices:
                st.session_state.matula_adj_matrix = np.zeros((n, n), dtype=int)
                st.session_state.prev_vertices = vertices.copy()
            
            with col2:
                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                st.write("**Matrice d'adjacence (graphe non orienté):**")
                st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
                
                # En-têtes des colonnes
                cols = st.columns(n + 1)
                with cols[0]:
                    st.write("**De\\À**")
                for j in range(n):
                    with cols[j + 1]:
                        st.write(f"**{vertices[j]}**")
                
                # Lignes de la matrice
                for i in range(n):
                    cols = st.columns(n + 1)
                    with cols[0]:
                        st.write(f"**{vertices[i]}**")
                    
                    for j in range(n):
                        with cols[j + 1]:
                            if i == j:
                                st.write("0")
                            elif j > i:  # Triangle supérieur seulement
                                key = f"matula_adj_{i}_{j}"
                                checked = st.checkbox("", 
                                                    value=bool(st.session_state.matula_adj_matrix[i][j]), 
                                                    key=key,
                                                    label_visibility="collapsed")
                                st.session_state.matula_adj_matrix[i][j] = 1 if checked else 0
                                st.session_state.matula_adj_matrix[j][i] = 1 if checked else 0
                            else:
                                # Afficher la valeur symétrique
                                value = st.session_state.matula_adj_matrix[i][j]
                                st.write("✓" if value else "")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # === BOUTON STYLISÉ ===
    
    if st.button("🚀Lancer l'algorithme", type="primary", use_container_width=True):
        if not vertices_input:
            st.error("❌ Veuillez d'abord saisir les sommets")
        else:
            vertices = [v.strip() for v in vertices_input.split(',') if v.strip()]
            
            if not vertices:
                st.error("❌ Aucun sommet valide saisi")
            else:
                try:
                    # Vérifier si au moins une arête existe
                    has_edges = False
                    n = len(vertices)
                    for i in range(n):
                        for j in range(i+1, n):
                            if st.session_state.matula_adj_matrix[i][j] == 1:
                                has_edges = True
                                break
                        if has_edges:
                            break
                    
                    if not has_edges:
                        st.warning("⚠️ Le graphe n'a aucune arête. Ajoutez au moins une arête dans la matrice d'adjacence.")
                    else:
                        # Début du chronomètre - AJOUTÉ ICI
                        start_time = time.time()
                        
                        # Initialiser la visualisation avec la matrice d'adjacence
                        matula_viz = MatulaVisualization(vertices, st.session_state.matula_adj_matrix)
                        
                        # Afficher le graphe initial
                        st.markdown('<div class="section-title">📈 Graphe Initial</div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_initial = matula_viz.visualize_graph(title="Graphe Initial - Non Coloré")
                            st.pyplot(fig_initial)
                        
                        with col2:
                            st.markdown('<div class="properties-container">', unsafe_allow_html=True)
                            st.subheader("📊 Informations du Graphe")
                            st.markdown(f'<div class="property-item">🎯 <strong>Nombre de sommets:</strong> {len(vertices)}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="property-item">🔗 <strong>Nombre d\'arêtes:</strong> {matula_viz.G.number_of_edges()}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="property-item">📈 <strong>Densité:</strong> {nx.density(matula_viz.G):.3f}</div>', unsafe_allow_html=True)
                        
                        # Exécuter l'algorithme
                        colors, classement, classement_inverse = matula_viz.run_matula_algorithm()
                        
                        # Calcul du temps d'exécution - AJOUTÉ ICI
                        execution_time = time.time() - start_time
                        
                        # Afficher le résultat final
                        st.markdown('<div class="section-title">✅ Résultat Final</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Graphe coloré final
                            fig_final = matula_viz.visualize_graph(colors=colors, 
                                                                 title="Graphe Final Coloré - Algorithme Matula")
                            st.pyplot(fig_final)
                        
                        with col2:
                            st.markdown('<div class="properties-container">', unsafe_allow_html=True)
                            st.subheader("🎯 Résumé de la Coloration")
                            
                            # Compter les couleurs utilisées
                            color_count = len(set(colors.values()))
                            st.markdown(f'<div class="property-item">🎨 <strong>Nombre de couleurs utilisées:</strong> {color_count}</div>', unsafe_allow_html=True)
                            
                            # AJOUT DU TEMPS D'EXÉCUTION - AJOUTÉ ICI
                            st.markdown(f'<div class="property-item">⏱️ <strong>Temps d\'exécution:</strong> {execution_time:.4f} secondes</div>', unsafe_allow_html=True)
                            
                            # Afficher les ordres
                            st.markdown(f'<div class="property-item">📋 <strong>Ordre de suppression :</strong> {" → ".join(classement)}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="property-item">📋 <strong>Ordre de coloration :</strong> {" → ".join(classement_inverse)}</div>', unsafe_allow_html=True)
                            
                            # Distribution des couleurs
                            st.subheader("📊 Distribution des Couleurs")
                            color_dist = defaultdict(list)
                            for vertex, color in colors.items():
                                color_dist[color].append(vertex)
                            
                            color_names = ["Rouge", "Bleu", "Vert", "Jaune", "Orange", "Violet", "Rose", "Marron"]
                            for color, vertices_list in color_dist.items():
                                color_name = color_names[color % len(color_names)]
                                st.markdown(f'<div class="property-item" style="border-left-color: {color_names[color].lower() if color < len(color_names) else "darkred"};">'
                                           f'<strong>{color_name}:</strong> {", ".join(vertices_list)}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Tableau des couleurs par sommet
                            st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
                            st.subheader("📋 Tableau des Couleurs")
                            color_df = pd.DataFrame([
                                {'Sommet': vertex, 'Couleur': color_names[colors[vertex] % len(color_names)]} 
                                for vertex in colors
                            ])
                            st.dataframe(color_df, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    st.markdown('<div class="input-container">', unsafe_allow_html=True)
                    st.info("💡 Vérifiez que vous avez bien sélectionné des arêtes dans la matrice d'adjacence.")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # === BOUTON RÉINITIALISATION ===
    st.markdown("---")
    if st.button("🧹 Réinitialiser tout", use_container_width=True):
        keys_to_clear = ['matula_adj_matrix', 'prev_vertices']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()