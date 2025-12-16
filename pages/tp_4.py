# pages/tp4.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="TP4 - Algorithmes PCC et Coloration", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
    .stApp {
        background-color: white;
        font-family: 'Arial', sans-serif;
    }
    
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: darkred;
        text-align: center;
        padding: 15px;
        border: 2px solid darkred;
        border-radius: 8px;
        margin: 15px auto;
        width: 80%;
        background-color: #f8f9fa;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: darkred;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid darkred;
    }
    
    .stButton>button {
        background-color: darkred;
        color: white;
        border-radius: 18px;
        font-size: 16px;
        padding: 10px 20px;
        margin: 5px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #8B0000;
    }
    
    .iteration-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
    }
    
    .iteration-table th, .iteration-table td {
        border: 2px solid darkred;
        padding: 10px;
        text-align: center;
        background-color: white;
    }
    
    .iteration-table th {
        background-color: darkred;
        color: white;
        font-weight: bold;
    }
    
    .final-table {
        width: 60%;
        border-collapse: collapse;
        margin: 20px auto;
        font-size: 16px;
    }
    
    .final-table th, .final-table td {
        border: 2px solid darkred;
        padding: 12px;
        text-align: center;
    }
    
    .final-table th {
        background-color: darkred;
        color: white;
        font-weight: bold;
    }
    
    .final-table td {
        background-color: #f9f9f9;
    }
    
    .star {
        color: red;
        font-weight: bold;
    }
    
    .negative-weight {
        color: blue;
        font-weight: bold;
    }
    
    /* Styles pour l'algorithme de coloration */
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
        color: #262730 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: darkred;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Ensure main content text is visible */
    .main .block-container {
        color: #262730;
    }
    
    .main .block-container .stMarkdown p:not(.stButton *),
    .main .block-container .stText:not(.stButton *) {
        color: #262730 !important;
    }
    
    /* Input fields text color */
    .main [data-testid="stTextInput"] input,
    .main [data-testid="stTextArea"] textarea {
        color: #262730 !important;
    }
    
    /* Ensure buttons keep white text */
    .main .stButton>button {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== TITRE PRINCIPAL =====
st.markdown('<div class="main-title">TP4 - Algorithmes PCC et Coloration</div>', unsafe_allow_html=True)

# ===== ONGLETS =====
tab1, tab2 = st.tabs(["🔍 Algorithme de Bellman-Ford", "🎨 Algorithme de Coloration"])

def bellman_ford_with_iterations(vertices, edges, source):
    """Implémentation CORRIGÉE de Bellman-Ford """
    n = len(vertices)
    
    # Initialisation
    distance = {v: float('inf') for v in vertices}
    predecessor = {v: None for v in vertices}
    distance[source] = 0
    
    # Stocker l'historique de chaque itération
    iterations = []
    
    # État initial
    initial_state = {v: distance[v] for v in vertices}
    initial_changes = {v: False for v in vertices}
    initial_changes[source] = True
    iterations.append(("Initialisation", initial_state.copy(), initial_changes.copy()))
    
    # Relaxation des arcs (n-1 fois) - VERSION CORRIGÉE POUR POIDS NÉGATIFS
    for k in range(1, n):
        changes = {v: False for v in vertices}
        previous_distance = distance.copy()  # Sauvegarder l'état précédent
        
        # Appliquer la relaxation sur tous les arcs
        for u, v, w in edges:
            # Vérifier si on peut améliorer la distance en utilisant l'arc (u, v)
            if previous_distance[u] != float('inf') and previous_distance[u] + w < distance[v]:
                distance[v] = previous_distance[u] + w
                predecessor[v] = u
                changes[v] = True
        
        # Stocker l'état de cette itération
        current_state = {v: distance[v] for v in vertices}
        iterations.append((f"Itération {k}", current_state.copy(), changes.copy()))
        
        # Vérifier si aucune modification n'a été apportée (optimisation)
        no_changes = True
        for v in vertices:
            if changes[v]:
                no_changes = False
                break
                
        if no_changes:
            # Remplir les itérations restantes avec les mêmes valeurs
            for remaining_k in range(k + 1, n):
                iterations.append((f"Itération {remaining_k}", current_state.copy(), {v: False for v in vertices}))
            break
    
    # Vérification des cycles de poids négatif (étape supplémentaire)
    has_negative_cycle = False
    negative_cycle_info = ""
    
    for u, v, w in edges:
        if distance[u] != float('inf') and distance[u] + w < distance[v]:
            has_negative_cycle = True
            negative_cycle_info = f"Cycle négatif détecté sur l'arc {u}→{v} (poids: {w})"
            break
    
    return distance, predecessor, has_negative_cycle, negative_cycle_info, iterations

def get_shortest_path(predecessor, target):
    """Reconstruit le chemin le plus court depuis la source jusqu'à target"""
    if predecessor[target] is None:
        return []  # Aucun chemin depuis la source
    
    path = []
    current = target
    
    # Remonter les prédecesseurs jusqu'à la source
    while current is not None:
        path.append(current)
        current = predecessor[current]
    
    # Inverser pour avoir source -> target
    path.reverse()
    return path

def build_shortest_path_tree(vertices, predecessor, source, edges):
    """Construit le graphe partiel des plus courts chemins (arborescence) avec les poids"""
    G_shortest = nx.DiGraph()
    G_shortest.add_nodes_from(vertices)
    
    # Reconstruire les poids des arcs du graphe partiel
    edge_weights = {}
    for u, v, w in edges:
        edge_weights[(u, v)] = w
    
    # Ajouter les arcs qui font partie des plus courts chemins avec leurs poids
    for v in vertices:
        if v != source and predecessor[v] is not None:
            u = predecessor[v]
            weight = edge_weights.get((u, v), 1)  # Utiliser le poids original
            G_shortest.add_edge(u, v, weight=weight)
    
    return G_shortest

def format_value(value, changed=False):
    """Formate une valeur pour l'affichage dans le tableau"""
    if value == float('inf'):
        return "∞"
    elif value == int(value):
        str_value = str(int(value))
    else:
        str_value = f"{value:.1f}"
    
    return f"{str_value}{'*' if changed else ''}"

def format_weight_display(weight):
    """Formate l'affichage des poids (spécial pour poids négatifs)"""
    if weight == float('inf'):
        return "∞"
    elif weight == int(weight):
        return str(int(weight))
    else:
        return f"{weight:.1f}"

with tab1:
    st.markdown('<div class="section-title">Algorithme de Bellman-Ford</div>', unsafe_allow_html=True)
    
    # Configuration du graphe
    st.markdown("### 1. Configuration du graphe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Saisie des sommets
        st.subheader("Saisie des sommets")
        vertices_input = st.text_input(
            "Entrez les sommets (séparés par des virgules):",
            value="S,U,X,V,Y",
            placeholder="Ex: A,B,C,D ou S,U,X,V,Y",
            help="Saisissez les noms des sommets séparés par des virgules"
        )
        
        if vertices_input:
            vertices = [v.strip() for v in vertices_input.split(',') if v.strip()]
            st.success(f"Sommets définis: {', '.join(vertices)}")
            
            # Initialiser les matrices quand les sommets changent
            if 'prev_vertices' not in st.session_state or st.session_state.prev_vertices != vertices:
                n = len(vertices)
                st.session_state.adj_matrix = np.zeros((n, n), dtype=int)
                st.session_state.weight_matrix = np.full((n, n), float('inf'))
                for i in range(n):
                    st.session_state.weight_matrix[i][i] = 0
                st.session_state.prev_vertices = vertices.copy()
        else:
            vertices = []
            st.info("Veuillez saisir les sommets")
    
    with col2:
        # Sélection du sommet source
        st.subheader("Sommet source")
        source_vertex = st.selectbox(
            "Choisissez le sommet source:",
            vertices if vertices else ["Aucun sommet disponible"],
            help="Sélectionnez le sommet de départ pour l'algorithme"
        )
    
    # Matrices d'adjacence et de pondération
    if vertices:
        st.markdown("### 2. Matrices du graphe")
        
        n = len(vertices)
        
        # Initialisation des matrices
        if 'adj_matrix' not in st.session_state:
            st.session_state.adj_matrix = np.zeros((n, n), dtype=int)
        if 'weight_matrix' not in st.session_state:
            st.session_state.weight_matrix = np.full((n, n), float('inf'))
            for i in range(n):
                st.session_state.weight_matrix[i][i] = 0
        
        # Matrice d'adjacence
        st.markdown("#### Matrice d'adjacence")
        st.write("Cochez les cases pour indiquer les arcs existants:")
        
        # En-têtes des colonnes
        cols = st.columns(n + 1)
        with cols[0]:
            st.write("**From/To**")
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
                    else:
                        key = f"adj_{i}_{j}"
                        checked = st.checkbox("", 
                                            value=bool(st.session_state.adj_matrix[i][j]), 
                                            key=key,
                                            label_visibility="collapsed")
                        st.session_state.adj_matrix[i][j] = 1 if checked else 0
        
        # Matrice de pondération (AVEC SUPPORT DES POIDS NÉGATIFS)
        st.markdown("#### Matrice de pondération")
        st.markdown("<p class='negative-weight'></p>", unsafe_allow_html=True)
        st.write("Entrez les poids des arcs (laisser vide si pas d'arc):")
        
        # En-têtes des colonnes
        cols = st.columns(n + 1)
        with cols[0]:
            st.write("**From/To**")
        for j in range(n):
            with cols[j + 1]:
                st.write(f"**{vertices[j]}**")
        
        # Lignes de la matrice de pondération
        for i in range(n):
            cols = st.columns(n + 1)
            with cols[0]:
                st.write(f"**{vertices[i]}**")
            
            for j in range(n):
                with cols[j + 1]:
                    if i == j:
                        st.write("0")
                    else:
                        key = f"weight_{i}_{j}"
                        current_val = st.session_state.weight_matrix[i][j]
                        
                        display_value = ""
                        if current_val != float('inf'):
                            display_value = format_weight_display(current_val)
                        
                        weight = st.text_input("", 
                                             value=display_value, 
                                             key=key, 
                                             placeholder="∞", 
                                             label_visibility="collapsed",
                                             help=f"Poids pour {vertices[i]}→{vertices[j]} (peut être négatif)")
                        
                        if weight and weight.strip():
                            if weight == "∞":
                                st.session_state.weight_matrix[i][j] = float('inf')
                            else:
                                try:
                                    # Autoriser les nombres négatifs
                                    weight_value = float(weight)
                                    st.session_state.weight_matrix[i][j] = weight_value
                                except ValueError:
                                    st.error(f"Valeur invalide pour {vertices[i]}→{vertices[j]}. Doit être un nombre.")
        
        # Exemple pré-rempli avec possibilité de poids négatifs
        with st.expander("💡 Exemples pré-remplis (cliquez pour charger)"):
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                if st.button("Charger l'exemple standard"):
                    example_vertices = ["S", "U", "X", "V", "Y"]
                    example_adj = [
                        [0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 1],
                        [0, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0]
                    ]
                    example_weights = [
                        [0, 10, 5, float('inf'), float('inf')],
                        [float('inf'), 0, 2, 1, float('inf')],
                        [float('inf'), 3, 0, 9, 2],
                        [float('inf'), float('inf'), float('inf'), 0, 4],
                        [7, float('inf'), float('inf'), 6, 0]
                    ]
                    
                    st.session_state.adj_matrix = np.array(example_adj)
                    st.session_state.weight_matrix = np.array(example_weights)
                    st.session_state.prev_vertices = example_vertices.copy()
                    st.rerun()
            
            with col_ex2:
                if st.button("Charger exemple avec poids négatifs"):
                    example_vertices = ["A", "B", "C", "D"]
                    example_adj = [
                        [0, 1, 1, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]
                    ]
                    example_weights = [
                        [0, 4, 5, float('inf')],
                        [float('inf'), 0, -2, 3],
                        [float('inf'), float('inf'), 0, 1],
                        [2, float('inf'), float('inf'), 0]
                    ]
                    
                    st.session_state.adj_matrix = np.array(example_adj)
                    st.session_state.weight_matrix = np.array(example_weights)
                    st.session_state.prev_vertices = example_vertices.copy()
                    st.rerun()
        
        # Boutons d'action
        st.markdown("### 3. Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Construire le Graphe", use_container_width=True):
                # Validation
                valid = True
                for i in range(n):
                    for j in range(n):
                        if i != j and st.session_state.adj_matrix[i][j] == 1:
                            if st.session_state.weight_matrix[i][j] == float('inf'):
                                st.error(f"Poids manquant pour {vertices[i]}→{vertices[j]}")
                                valid = False
                
                if valid:
                    # Construction du graphe
                    G = nx.DiGraph()
                    G.add_nodes_from(vertices)
                    edges = []
                    for i in range(n):
                        for j in range(n):
                            if st.session_state.adj_matrix[i][j] == 1 and i != j:
                                weight = st.session_state.weight_matrix[i][j]
                                G.add_edge(vertices[i], vertices[j], weight=weight)
                                edges.append((vertices[i], vertices[j], weight))
                    
                    # Visualisation avec coloration pour poids négatifs
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
                    nx.draw_networkx_labels(G, pos, font_weight='bold')
                    
                    # Colorer les arcs selon le poids
                    edge_colors = []
                    for u, v, data in G.edges(data=True):
                        if data['weight'] < 0:
                            edge_colors.append('red')  # Rouge pour poids négatifs
                        else:
                            edge_colors.append('gray')
                    
                    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20)
                    
                    # Labels des arêtes
                    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                    ax.set_title("Graphe Construit (rouge = poids négatifs)")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.session_state.graph = G
                    st.session_state.edges = edges
                    st.session_state.vertices = vertices
                    st.success("Graphe construit avec succès!")
        
        with col2:
            if st.button("⚡ Appliquer Bellman-Ford", use_container_width=True):
                if 'graph' not in st.session_state:
                    st.error("Veuillez d'abord construire le graphe")
                else:
                    # Application de Bellman-Ford
                    vertices = st.session_state.vertices
                    edges = st.session_state.edges
                    
                    distance, predecessor, has_negative_cycle, negative_cycle_info, iterations = bellman_ford_with_iterations(
                        vertices, edges, source_vertex
                    )
                    
                    st.markdown("### 4. Tableau des itérations de Bellman-Ford")
                    
                    if has_negative_cycle:
                        st.error(f"❌ {negative_cycle_info}")
                        st.warning("L'algorithme a détecté un cycle de poids négatif. Les résultats peuvent être incorrects.")
                    else:
                        st.success("✅ Aucun cycle de poids négatif détecté")
                    
                    # Création du tableau d'itérations
                    table_html = """
                    <table class="iteration-table">
                        <thead>
                            <tr>
                                <th>k</th>
                    """
                    
                    # En-têtes des colonnes (sommets)
                    for v in vertices:
                        table_html += f'<th>λ({v})</th>'
                    table_html += "</tr></thead><tbody>"
                    
                    # Lignes du tableau (itérations)
                    for iter_name, state, changes in iterations:
                        table_html += f"<tr><td><strong>{iter_name}</strong></td>"
                        
                        for v in vertices:
                            value = state[v]
                            formatted_value = format_value(value, changes[v])
                            table_html += f"<td>{formatted_value}</td>"
                        
                        table_html += "</tr>"
                    
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)
                    
                    # Légende
                    st.markdown("<p class='star'>* : valeur mise à jour lors de cette itération</p>", unsafe_allow_html=True)
                    
                    # Résultats finaux
                    st.markdown("### 5. Résultats finaux")
                    
                    # Utiliser un dataframe pandas pour un affichage propre
                    import pandas as pd
                    
                    results_data = []
                    for v in vertices:
                        path = get_shortest_path(predecessor, v)
                        if not path:
                            if v == source_vertex:
                                path_str = v
                            else:
                                path_str = "Aucun chemin"
                        else:
                            path_str = " → ".join(path)
                        
                        dist_display = "∞" if distance[v] == float('inf') else f"{distance[v]:.1f}"
                        
                        results_data.append({
                            "Sommet": v,
                            "Distance finale": dist_display,
                            "Chemin le plus court": path_str
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # GRAPHE PARTIEL DES PLUS COURS CHEMINS
                    st.markdown("### 6. Graphe Partiel G° (Arborescence des plus courts chemins)")
                    
                    # Construire le graphe partiel avec les poids
                    G_shortest = build_shortest_path_tree(vertices, predecessor, source_vertex, edges)
                    
                    # Visualiser le graphe partiel
                    fig2, ax2 = plt.subplots(figsize=(12, 10))  # Plus grand pour mieux voir
                    pos_shortest = nx.spring_layout(G_shortest, seed=42, k=3, iterations=50)  # Meilleur espacement
                    
                    # Dessiner les nœuds - PLUS GRANDS
                    nx.draw_networkx_nodes(G_shortest, pos_shortest, node_color='lightgreen', 
                                          node_size=2000, alpha=0.8, edgecolors='darkgreen', linewidths=2)
                    
                    # Dessiner les arcs
                    nx.draw_networkx_edges(G_shortest, pos_shortest, edge_color='darkgreen', 
                                          arrows=True, arrowsize=25, width=3, alpha=0.8,
                                          arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
                    
                    # Labels des noms des nœuds (plus grands) - D'ABORD les noms
                    nx.draw_networkx_labels(G_shortest, pos_shortest, font_size=16, 
                                          font_weight='bold', font_color='darkblue')
                    
                    # PUIS les distances AU-DESSUS des nœuds - CORRECTION ICI
                    for v, pos in pos_shortest.items():
                        # Ajuster la position pour être AU-DESSUS du nœud
                        label_pos = (pos[0], pos[1] + 0.2)  # Décalage vertical
                        dist_text = f"d({v}) = {distance[v]:.1f}" if distance[v] != float('inf') else f"d({v}) = ∞"
                        ax2.text(label_pos[0], label_pos[1], dist_text, 
                                fontsize=12, fontweight='bold', ha='center', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                        edgecolor='darkgreen', alpha=0.9))
                    
                    # Ajouter les poids des arcs
                    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G_shortest.edges(data=True)}
                    nx.draw_networkx_edge_labels(G_shortest, pos_shortest, edge_labels=edge_labels,
                                               font_size=12, font_weight='bold', 
                                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                    
                    ax2.set_title(f"Graphe Partiel G° - Arborescence des plus courts chemins depuis {source_vertex}", 
                                fontsize=16, fontweight='bold', pad=20)
                    ax2.axis('off')
                    
                    # Ajuster les limites pour mieux voir tous les labels
                    ax2.set_xlim([min(x for x, y in pos_shortest.values()) - 0.3, 
                                max(x for x, y in pos_shortest.values()) + 0.3])
                    ax2.set_ylim([min(y for x, y in pos_shortest.values()) - 0.3, 
                                max(y for x, y in pos_shortest.values()) + 0.3])
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Informations supplémentaires
                    st.info("""
                    **Légende du graphe partiel:**
                    - 🟢 **Nœuds verts** : Sommets accessibles depuis la source
                    - 🟢 **Arcs verts** : Arcs appartenant aux plus courts chemins
                    - **Au-dessus des nœuds** : Distance finale depuis la source
                    - **Sur les arcs** : Poids des arêtes
                    - **Dans les nœuds** : Nom du sommet
                    """)
        
        with col3:
            if st.button("🧹 Réinitialiser", use_container_width=True):
                keys_to_clear = ['adj_matrix', 'weight_matrix', 'graph', 'edges', 'vertices', 'prev_vertices']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

with tab2:
    st.markdown('<div class="section-title">🎨 Algorithme de Matula - Coloration de Graphes</div>', unsafe_allow_html=True)
    
    # Classe MatulaVisualization
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
            st.markdown('<div class="section-title">📋 Inversion du Classement</div>', unsafe_allow_html=True)
            
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
            if 'matula_adj_matrix' not in st.session_state or st.session_state.get('prev_vertices_coloring') != vertices:
                st.session_state.matula_adj_matrix = np.zeros((n, n), dtype=int)
                st.session_state.prev_vertices_coloring = vertices.copy()
            
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
    
    if st.button("🚀 Lancer l'algorithme de coloration", type="primary", use_container_width=True):
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
                        # Début du chronomètre
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
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Exécuter l'algorithme
                        colors, classement, classement_inverse = matula_viz.run_matula_algorithm()
                        
                        # Calcul du temps d'exécution
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
                            
                            # Temps d'exécution
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
                                st.markdown(f'<div class="property-item"><strong>{color_name}:</strong> {", ".join(vertices_list)}</div>', unsafe_allow_html=True)
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
    if st.button("🧹 Réinitialiser", use_container_width=True, key="reset_coloring"):
        keys_to_clear = ['matula_adj_matrix', 'prev_vertices_coloring']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ===== PIED DE PAGE =====
st.markdown("---")
st.markdown("*TP4 - Algorithmes PCC et Coloration - Group 4*")