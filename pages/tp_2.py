import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import io

matplotlib.use('Agg')

# ============================
#  Arbre 2-3 (B-arbre ordre 3)
# ============================

class Node23:
    def __init__(self, keys=None, children=None):
        self.keys = list(keys) if keys else []  # 1 or 2 integers
        self.children = list(children) if children else []  # 0, 2 or 3 Node23

    def is_leaf(self):
        return len(self.children) == 0

    def num_keys(self):
        return len(self.keys)

    def num_children(self):
        return len(self.children)

    def __repr__(self):
        return f"Node23(keys={self.keys})"


class Tree23:
    def __init__(self):
        self.root = None

    # ======== SEARCH ========
    def search(self, key):
        def _search(node, key):
            if node is None:
                return False, None
            # keys sorted
            for k in node.keys:
                if key == k:
                    return True, node
            if node.is_leaf():
                return False, None
            if key < node.keys[0]:
                return _search(node.children[0], key)
            if node.num_keys() == 1 or key < node.keys[1]:
                return _search(node.children[1], key)
            return _search(node.children[2], key)

        return _search(self.root, key)

    # ======== INSERT ========
    def insert(self, key):
        if self.root is None:
            self.root = Node23([key])
            return self.root

        def _insert(node, key):
            if node.is_leaf():
                # insert in sorted order
                node.keys.append(key)
                node.keys.sort()
                if node.num_keys() <= 2:
                    return None  # no split
                # split 3 keys into 2 nodes and promote middle
                return self._split_node(node)

            # choose child
            if key < node.keys[0]:
                idx = 0
            elif node.num_keys() == 1 or key < node.keys[1]:
                idx = 1
            else:
                idx = 2

            split_result = _insert(node.children[idx], key)
            if split_result is None:
                return None

            left, middle_key, right = split_result
            # insert promoted key and redistribute children
            node.keys.insert(idx, middle_key)
            node.children[idx] = left
            node.children.insert(idx + 1, right)

            if node.num_keys() <= 2:
                return None
            return self._split_node(node)

        res = _insert(self.root, key)
        if res is not None:
            left, middle_key, right = res
            self.root = Node23([middle_key], [left, right])
        return self.root

    def _split_node(self, node):
        # node has 3 keys and 0/4 children → split around middle key
        k0, k1, k2 = node.keys
        if node.children:
            c0, c1, c2, c3 = node.children
            left = Node23([k0], [c0, c1])
            right = Node23([k2], [c2, c3])
        else:
            left = Node23([k0])
            right = Node23([k2])
        return left, k1, right

    # ======== DELETE ========
    def delete(self, key):
        if self.root is None:
            return

        def _get_successor(node):
            # go to right child then all the way left
            cur = node.children[1] if node.num_keys() == 1 else node.children[2]
            while not cur.is_leaf():
                cur = cur.children[0]
            return cur.keys[0]

        def _merge_with_sibling(parent, idx):
            # merge child at idx with a sibling: prefer left sibling if exists
            if idx > 0:
                # merge left sibling (idx-1), bring down parent key at idx-1
                left = parent.children[idx - 1]
                right = parent.children[idx]
                sep = parent.keys.pop(idx - 1)
                # resulting 2-key node
                merged_keys = left.keys + [sep] + right.keys
                merged_children = []
                if left.children or right.children:
                    merged_children = left.children + right.children
                parent.children[idx - 1] = Node23(merged_keys, merged_children)
                parent.children.pop(idx)
                return idx - 1
            else:
                # merge with right sibling
                left = parent.children[idx]
                right = parent.children[idx + 1]
                sep = parent.keys.pop(idx)
                merged_keys = left.keys + [sep] + right.keys
                merged_children = []
                if left.children or right.children:
                    merged_children = left.children + right.children
                parent.children[idx] = Node23(merged_keys, merged_children)
                parent.children.pop(idx + 1)
                return idx

        def _borrow_from_sibling(parent, idx):
            # try borrow from left sibling
            if idx > 0 and parent.children[idx - 1].num_keys() == 2:
                left_sib = parent.children[idx - 1]
                child = parent.children[idx]
                # rotate right: move parent key down to child, left_sib max up to parent
                down_key = parent.keys[idx - 1]
                up_key = left_sib.keys.pop()
                parent.keys[idx - 1] = up_key
                child.keys.insert(0, down_key)
                if left_sib.children:
                    child.children.insert(0, left_sib.children.pop())
                return True
            # try borrow from right sibling
            if idx < len(parent.children) - 1 and parent.children[idx + 1].num_keys() == 2:
                right_sib = parent.children[idx + 1]
                child = parent.children[idx]
                down_key = parent.keys[idx]
                up_key = right_sib.keys.pop(0)
                parent.keys[idx] = up_key
                child.keys.append(down_key)
                if right_sib.children:
                    child.children.append(right_sib.children.pop(0))
                    return True
            return False

        def _delete(node, key):
            # return whether root may shrink handled outside
            # Case 1: key is in this node
            if key in node.keys:
                pos = node.keys.index(key)
                if node.is_leaf():
                    node.keys.pop(pos)
                else:
                    # replace with successor
                    if node.num_keys() == 1:
                        succ = _min_key(node.children[1])
                        node.keys[pos] = succ
                        _delete(node.children[1], succ)
                    else:
                        succ = _min_key(node.children[pos + 1])
                        node.keys[pos] = succ
                        _delete(node.children[pos + 1], succ)
            else:
                # descend to appropriate child
                if node.is_leaf():
                    return
                if key < node.keys[0]:
                    idx = 0
                elif node.num_keys() == 1 or key < node.keys[1]:
                    idx = 1
                else:
                    idx = 2
                child = node.children[idx]
                _delete(child, key)

            # fix underflow if any child has 0 keys
            if not node.is_leaf():
                for idx, child in enumerate(list(node.children)):
                    if child.num_keys() == 0:
                        if not _borrow_from_sibling(node, idx):
                            _merge_with_sibling(node, idx)

        def _min_key(node):
            cur = node
            while not cur.is_leaf():
                cur = cur.children[0]
            return cur.keys[0]

        _delete(self.root, key)

        # shrink root if empty and has a child
        if self.root and self.root.num_keys() == 0:
            if self.root.children:
                self.root = self.root.children[0]
            else:
                self.root = None

    # ======== DISPLAY ========
    def display(self):
        if not self.root:
            return []

        levels = []
        def dfs(node, depth=0):
            if len(levels) <= depth:
                levels.append([])
            levels[depth].append(node)
            for c in node.children:
                dfs(c, depth + 1)
        dfs(self.root)
        return [[n.keys for n in lvl] for lvl in levels]

    # ======== BALANCE CHECK ========
    def is_balanced(self):
        """Vérifie si toutes les feuilles sont à la même profondeur"""
        if not self.root:
            return True
        
        leaf_depths = []
        def dfs_depth(node, depth=0):
            if node.is_leaf():
                leaf_depths.append(depth)
            else:
                for child in node.children:
                    dfs_depth(child, depth + 1)
        
        dfs_depth(self.root)
        return len(set(leaf_depths)) == 1

    # ======== TREE STATISTICS ========
    def get_stats(self):
        """Retourne les statistiques de l'arbre"""
        if not self.root:
            return {
                'total_nodes': 0,
                'total_keys': 0,
                'height': 0,
                'is_balanced': True,
                'leaf_depths': []
            }
        
        total_nodes = 0
        total_keys = 0
        leaf_depths = []
        max_depth = 0
        
        def dfs_stats(node, depth=0):
            nonlocal total_nodes, total_keys, max_depth
            total_nodes += 1
            total_keys += node.num_keys()
            max_depth = max(max_depth, depth)
            
            if node.is_leaf():
                leaf_depths.append(depth)
            else:
                for child in node.children:
                    dfs_stats(child, depth + 1)
        
        dfs_stats(self.root)
        
        return {
            'total_nodes': total_nodes,
            'total_keys': total_keys,
            'height': max_depth,
            'is_balanced': len(set(leaf_depths)) == 1,
            'leaf_depths': leaf_depths
        }


# ============================
#  Visualisation (Streamlit)
# ============================

st.markdown("""
<style>
    .stApp { background-color: white; font-family: 'Arial', sans-serif; }
    [data-testid="stSidebar"] { background-color: darkred; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton>button { background-color: darkred; color: white; border-radius: 18px; font-size: 14px; padding: 10px 20px; width: 200px; height:42px; margin: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stButton>button:hover { background-color: #8B0000; box-shadow: 0 6px 12px rgba(0,0,0,0.3); }
    .section-title { font-size: 24px; font-weight: bold; color: darkred; margin-top: 10px; margin-bottom: 10px; }
    .input-container { background-color: #f0f2f6; padding: 12px; border-radius: 8px; border: 0.5px solid #ccc; margin: 10px 0; font-size: 14px; }
    .properties-container { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 2px solid darkred; margin: 15px 0; }
    .property-item { font-size: 16px; margin: 8px 0; padding: 5px; border-left: 4px solid darkred; padding-left: 10px; background-color: white; }
    .node-rect { border-radius: 10px; }
    
    /* Réduire les marges pour utiliser toute la largeur */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
    .stContainer { padding: 0; }
    .element-container { padding: 0; }
    
    /* Beautiful form button styling */
    .stForm .stButton>button {
        background-color: darkred !important;
        color: white !important;
        border-radius: 18px !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        width: 200px !important;
        height: 42px !important;
        margin: 6px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        border: none !important;
    }
    
    .stForm .stButton>button:hover {
        background-color: #8B0000 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }
    
    .stForm .stButton[data-testid="baseButton-secondary"]>button {
        background-color: darkred !important;
    }
    
    .stForm .stButton[data-testid="baseButton-secondary"]>button:hover {
        background-color: #8B0000 !important;
    }
    
    /* Make "érer le graph" button tall and full width */
    button[kind="secondary"], .stButton[key="finalize_tree"] button {
        height: 65px !important;
        width: 100% !important;
        font-size: 18px !important;
        padding: 20px 30px !important;
        background-color: darkred !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .stButton[key="finalize_tree"] button:hover {
        background-color: #8B0000 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }
    
    /* Make "Browse files" button much smaller */
    .stFileUploader [data-testid="stFileUploadDropzone"] button,
    .uploadedFile button,
    button[data-testid="baseButton-secondary"] {
        height: 35px !important;
        width: 150px !important;
        padding: 5px 12px !important;
        font-size: 12px !important;
    }
    
    /* ✅ Ensure main content text is visible - but exclude buttons */
    .main .block-container {
        color: #262730;
    }
    
    /* Only target plain text, not buttons or styled elements */
    .main .block-container .stMarkdown p:not(.stButton *),
    .main .block-container .stText:not(.stButton *) {
        color: #262730 !important;
    }
    
    /* ✅ Input fields text color */
    .main [data-testid="stTextInput"] input {
        color: #262730 !important;
    }
    
    .main [data-testid="stTextInput"] input::placeholder {
        color: #999 !important;
    }
    
    /* ✅ Checkbox labels */
    .main [data-testid="stCheckbox"] label {
        color: #262730 !important;
    }
    
    /* ✅ File uploader text */
    .main [data-testid="stFileUploader"] label {
        color: #262730 !important;
    }
    
    /* ✅ Ensure buttons keep white text */
    .main .stButton>button {
        color: white !important;
    }
    
    /* ✅ Ensure styled titles keep their colors */
    .section-title {
        color: darkred !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'tp2_page' not in st.session_state:
    st.session_state.tp2_page = 'edition'  # 'edition' | 'visualisation'
if 'tree23' not in st.session_state:
    st.session_state.tree23 = Tree23()
if 'insertion_start_time' not in st.session_state:
    st.session_state.insertion_start_time = None
if 'last_action' not in st.session_state:
    st.session_state.last_action = None
if 'last_key' not in st.session_state:
    st.session_state.last_key = None
if 'highlight_node_label' not in st.session_state:
    st.session_state.highlight_node_label = None


def build_graph_from_tree(tree: Tree23):
    G = nx.DiGraph()
    pos = {}
    labels = {}

    if not tree.root:
        return G, pos, labels

    def label_of(node: Node23):
        return "[" + ", ".join(str(k) for k in node.keys) + "]"

    def layout(node, x, y, dx):
        node_label = label_of(node)
        G.add_node(node_label)
        labels[node_label] = node_label
        pos[node_label] = (x, y)
        if node.children:
            count = len(node.children)
            for i, child in enumerate(node.children):
                child_x = x + (i - (count - 1) / 2.0) * dx
                child_y = y - 1.5
                child_label = label_of(child)
                G.add_edge(node_label, child_label)
                layout(child, child_x, child_y, dx / 2.0)

    layout(tree.root, 0.0, 0.0, 6.0)
    return G, pos, labels


def draw_tree(tree: Tree23, highlight_label=None, is_mini=False, figsize=None):
    G, pos, labels = build_graph_from_tree(tree)
    if figsize is None:
        figsize = (8, 5) if is_mini else (16, 9)
    fig, ax = plt.subplots(figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(Arbre vide)", ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

    # Colors
    default_node_color = []
    for node in G.nodes():
        if highlight_label and node == highlight_label:
            if st.session_state.last_action == 'search_found':
                default_node_color.append('green')
            elif st.session_state.last_action == 'insert':
                default_node_color.append('#4ade80')
            elif st.session_state.last_action == 'delete':
                default_node_color.append('#fca5a5')
            else:
                default_node_color.append('darkred')
        else:
            default_node_color.append('darkred')

    node_size = 2200 if is_mini else 2800  # Much bigger nodes for visibility
    font_size = 14 if is_mini else 16  # Bigger font
    
    nx.draw_networkx_nodes(G, pos,
                           node_color=default_node_color,
                           node_size=node_size,
                           ax=ax,
                           edgecolors='black',
                           linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=20, arrowstyle='-|>', width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_weight='bold', font_color='white', ax=ax)
    
    title = "Mini-prévisualisation" if is_mini else "Visualisation de l'arbre 2-3"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


def set_highlight_for_key(tree: Tree23, key: int, found: bool):
    # find node label where key resides (on found) else None
    if not found:
        st.session_state.highlight_node_label = None
        return
    ok, node = tree.search(key)
    if ok and node is not None:
        st.session_state.highlight_node_label = "[" + ", ".join(str(k) for k in node.keys) + "]"
    else:
        st.session_state.highlight_node_label = None


# ======= Pages =======
if st.session_state.tp2_page == 'edition':
    # ===== PAGE A - ÉDITION =====
    st.markdown('<div class="section-title">📝Arebre 2 3 </div>', unsafe_allow_html=True)
    st.markdown("**Construire un arbre 2-3 soit valeur par valeur, soit en important un fichier**")

    # Deux colonnes: à gauche les contrôles (plus large), à droite l'arbre
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Section 1: Insertion manuelle
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("➕ Insertion manuelle")
        
        # Use form for Enter key functionality with beautiful button styling
        with st.form("insert_form", clear_on_submit=True):
            insert_val = st.text_input("Valeur à insérer", placeholder="Entrez un entier...")
            col_i1, col_i2 = st.columns([1, 1])
            with col_i1:
                submitted = st.form_submit_button("➕ Insérer", type="primary")
            with col_i2:
                clear_tree = st.form_submit_button("🗑️ Vider l'arbre", type="secondary")
        
        if submitted and insert_val:
            try:
                v = int(insert_val)
                # empêcher les doublons
                found, _ = st.session_state.tree23.search(v)
                if found:
                    st.error(f"La valeur {v} existe déjà dans l'arbre.")
                else:
                    if st.session_state.insertion_start_time is None:
                        st.session_state.insertion_start_time = time.perf_counter()
                    st.session_state.tree23.insert(v)
                    st.session_state.last_action = 'insert'
                    st.session_state.last_key = v
                    set_highlight_for_key(st.session_state.tree23, v, True)
                    st.success(f"Valeur {v} insérée")
                    st.rerun()
            except Exception as e:
                st.error(f"Valeur non valide ou erreur: {e}")
        
        if clear_tree:
            st.session_state.tree23 = Tree23()
            st.session_state.last_action = None
            st.session_state.last_key = None
            st.session_state.highlight_node_label = None
            st.session_state.insertion_start_time = None
            st.info("Arbre vidé")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Section 2: Chargement de fichier
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("📁 Charger depuis un fichier")
        uploaded_file = st.file_uploader("Choisir un fichier data.txt", type=['txt'], key="file_upload")
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                tokens = content.split()
                values = [int(t) for t in tokens]

                if st.button("📥 Charger les valeurs", key="load_values"):
                    if st.session_state.insertion_start_time is None:
                        st.session_state.insertion_start_time = time.perf_counter()

                    duplicates = []
                    inserted = 0
                    for v in values:
                        found, _ = st.session_state.tree23.search(v)
                        if found:
                            duplicates.append(v)
                        else:
                            st.session_state.tree23.insert(v)
                            inserted += 1

                    if inserted:
                        st.success(f"✅ {inserted} nouvelles valeurs insérées")
                    if duplicates:
                        st.error(f"Valeurs déjà présentes ignorées: {sorted(set(duplicates))}")
                    st.session_state.last_action = 'insert'
                    if inserted:
                        last_new = [v for v in values if v not in duplicates][-1]
                        st.session_state.last_key = last_new
                        set_highlight_for_key(st.session_state.tree23, last_new, True)
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("👁️ Aperçu de l'arbre")
        if st.session_state.tree23.root:
            fig_preview = draw_tree(st.session_state.tree23, highlight_label=st.session_state.highlight_node_label, figsize=(14, 8))
            st.pyplot(fig_preview)
            # Removed all metrics as requested - just show the tree
        else:
            st.info("Aucun arbre à afficher pour le moment")
        
        # Section Finaliser déplacée sous l'arbre à droite
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        if st.session_state.tree23.root:
            is_bal = st.session_state.tree23.is_balanced()
            if is_bal:
                st.success("✅ L'arbre est équilibré ")
                if st.button("Les Opérations", key="finalize_tree"):
                    st.session_state.tp2_page = 'visualisation'
                    st.rerun()
            else:
                st.warning("⚠️ L'arbre n'est pas équilibré — ajoutez d'autres valeurs ou corrigez.")
                stats = st.session_state.tree23.get_stats()
                st.write(f"Profondeurs des feuilles: {stats['leaf_depths']}")
        else:
            st.info("Veuillez insérer des valeurs d'abord")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ===== PAGE B - VISUALISATION FINALE =====
    # Bouton retour placé en haut à gauche
    left_header = st.columns([1, 6])[0]
    with left_header:
        if st.button("← Retour à l'insertion", key="back_to_edition"):
            st.session_state.tp2_page = 'edition'
            st.rerun()

    # Disposition: arbre à gauche (avec zone de saisie en dessous), infos à droite
    col_main, col_info = st.columns([3, 1])

    with col_main:
        fig = draw_tree(st.session_state.tree23, highlight_label=st.session_state.highlight_node_label, figsize=(18, 10))
        st.pyplot(fig)

        # Zone de saisie et boutons
        value = st.text_input("Valeur", key="tp2_action_value", placeholder="Entrez un entier...")
        col_cmd1, col_cmd2 = st.columns(2)
        with col_cmd1:
            if st.button("🔍 Rechercher", key="tp2_btn_search"):
                try:
                    v = int(value)
                    found, _node = st.session_state.tree23.search(v)
                    st.session_state.last_key = v
                    st.session_state.last_action = 'search_found' if found else 'search_not_found'
                    set_highlight_for_key(st.session_state.tree23, v, found)
                    if found:
                        st.success(f"✅ La valeur {v} a été trouvée dans l'arbre!")
                    else:
                        st.warning(f"⚠️ La valeur {v} n'existe pas dans l'arbre.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Entrée invalide: {e}")

        with col_cmd2:
            if st.button("❌ Supprimer", key="tp2_btn_delete"):
                try:
                    v = int(value)
                    before = st.session_state.tree23.display()
                    st.session_state.tree23.delete(v)
                    after = st.session_state.tree23.display()
                    st.session_state.last_key = v
                    st.session_state.last_action = 'delete'
                    st.session_state.highlight_node_label = None
                    if before != after:
                        st.warning(f"Suppression de {v} effectuée")
                    else:
                        st.info("Aucune modification (clé absente)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Entrée invalide: {e}")

        # Tests rapides en bas
        with st.expander("🧪 Tests rapides (TP)"):
            colt1, colt2, colt3 = st.columns(3)
            with colt1:
                if st.button("Insérer série [10,20,5,6,12,30,7,17]", key="tp2_test_insert_series"):
                    st.session_state.tree23 = Tree23()
                    st.session_state.insertion_start_time = time.perf_counter()
                    for v in [10, 20, 5, 6, 12, 30, 7, 17]:
                        st.session_state.tree23.insert(v)
                    st.session_state.last_action = 'insert'
                    st.session_state.last_key = 17
                    set_highlight_for_key(st.session_state.tree23, 17, True)
                    st.success("Série insérée")
                    st.rerun()
            with colt2:
                if st.button("Tester search(6) & search(25)", key="tp2_test_search"):
                    f6, _ = st.session_state.tree23.search(6)
                    f25, _ = st.session_state.tree23.search(25)
                    st.write(f"search(6) → {f6}")
                    st.write(f"search(25) → {f25}")
            with colt3:
                if st.button("delete(6), delete(10)", key="tp2_test_delete"):
                    st.session_state.tree23.delete(6)
                    st.session_state.tree23.delete(10)
                    st.session_state.last_action = 'delete'
                    st.session_state.last_key = 10
                    st.session_state.highlight_node_label = None
                    st.warning("Suppressions effectuées")
                    st.rerun()

    with col_info:
        # Information part moved to left side down
        st.markdown('<div class="properties-container">', unsafe_allow_html=True)
        st.subheader("📊 Informations")
        
        stats = st.session_state.tree23.get_stats()
        st.markdown(f'<div class="property-item">📏 <strong>Hauteur:</strong> {stats["height"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="property-item">🌳 <strong>Nœuds:</strong> {stats["total_nodes"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="property-item">🔑 <strong>Clés:</strong> {stats["total_keys"]}</div>', unsafe_allow_html=True)
        
        balance_text = "OUI" if stats['is_balanced'] else "NON"
        balance_color = "green" if stats['is_balanced'] else "red"
        st.markdown(f'<div class="property-item">⚖️ <strong>Équilibre:</strong> <span style="color: {balance_color}">{balance_text}</span></div>', unsafe_allow_html=True)
        
        if st.session_state.insertion_start_time:
            total_time = time.perf_counter() - st.session_state.insertion_start_time
            st.markdown(f'<div class="property-item">⏱️ <strong>Temps d\'insertion:</strong> {total_time:.3f}s</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🗑️ Vider l'arbre", key="tp2_btn_reset"):
            st.session_state.tree23 = Tree23()
            st.session_state.last_action = None
            st.session_state.last_key = None
            st.session_state.highlight_node_label = None
            st.session_state.insertion_start_time = None
            st.info("Arbre vidé")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

