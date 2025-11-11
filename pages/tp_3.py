import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from collections import deque

matplotlib.use("Agg")

# ===========================
# CONFIG & STYLE
# ===========================

st.set_page_config(page_title="TP3 - Arbre 2-3 & Quicksort", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: white;
        font-family: 'Arial', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: darkred;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .tp3-title {
        font-size: 26px;
        font-weight: bold;
        color: darkred;
        text-align: center;
        margin: 10px 0 5px 0;
    }

    .tp3-subtitle {
        font-size: 15px;
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }

    .bloc {
        background-color: #f8f9fa;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 18px;
    }

    .step-title {
        font-size: 18px;
        font-weight: bold;
        color: darkred;
        margin-bottom: 6px;
    }

    .code-like {
        font-family: "Courier New", monospace;
        background-color: #eee;
        padding: 6px 10px;
        border-radius: 5px;
        display: inline-block;
        margin: 2px 0;
    }

    .array-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .box {
        min-width: 34px;
        padding: 6px 6px;
        text-align: center;
        border-radius: 8px;
        border: 1px solid #555;
        font-weight: bold;
        font-size: 14px;
        background-color: white;
        transition: all 0.2s ease-in-out;
    }

    .box-normal {
        background-color: #ffffff;
    }

    .box-pivot {
        background-color: #b22222;
        color: white;
        border-color: #7f0000;
        transform: scale(1.08);
    }

    .box-swap {
        background-color: #ffd966;
        border-color: #ff9900;
    }

    .box-fixed {
        background-color: #d9ead3;
        border-color: #38761d;
    }

    .step-label {
        font-weight: bold;
        color: darkred;
    }

    .small-label {
        font-size: 11px;
        color: #555;
    }

    .stButton>button {
        background-color: darkred;
        color: white;
        border-radius: 18px;
        font-size: 14px;
        padding: 8px 18px;
        margin-top: 4px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #8B0000;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }

    [data-testid="stTextInput"] input {
        border: 2px solid darkred !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 8px rgba(255,0,0,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="tp3-title">TP3 - Arbre 2-3 & Tri Rapide (Quicksort)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tp3-subtitle">
Arbre 2-3 (Node23 / Tree23) → Tableau préfixe (NON trié) → Tableau trié par Quicksort → Nouvel arbre 2-3 (fixe) → Étapes du tri pour compréhension.
</div>
""", unsafe_allow_html=True)

# ===========================
# Implémentation Arbre 2-3 correcte
# ===========================

class Node23:
    def __init__(self, keys=None, children=None):
        self.keys = list(keys) if keys else []
        self.children = list(children) if children else []

    def is_leaf(self):
        return len(self.children) == 0

class Tree23:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node23([key])
            return self.root

        res = self._insert(self.root, key)
        if res is not None:
            left, middle_key, right = res
            self.root = Node23([middle_key], [left, right])
        return self.root

    def _insert(self, node, key):
        if node.is_leaf():
            node.keys.append(key)
            node.keys.sort()
            if len(node.keys) <= 2:
                return None
            return self._split_node(node)

        if key < node.keys[0]:
            idx = 0
        elif len(node.keys) == 1 or key < node.keys[1]:
            idx = 1
        else:
            idx = 2

        split_result = self._insert(node.children[idx], key)
        if split_result is None:
            return None

        left, middle_key, right = split_result
        node.keys.insert(idx, middle_key)
        node.children[idx] = left
        node.children.insert(idx + 1, right)

        if len(node.keys) <= 2:
            return None
        return self._split_node(node)

    def _split_node(self, node):
        k0, k1, k2 = node.keys
        if node.children:
            c0, c1, c2, c3 = node.children
            left = Node23([k0], [c0, c1])
            right = Node23([k2], [c2, c3])
        else:
            left = Node23([k0])
            right = Node23([k2])
        return left, k1, right

    def levels(self):
        if not self.root:
            return []
        res = []
        def dfs(node, depth=0):
            if len(res) <= depth:
                res.append([])
            res[depth].append(node)
            for c in node.children:
                dfs(c, depth + 1)
        dfs(self.root)
        return [[n.keys for n in lvl] for lvl in res]


# ===========================
# Fonctions arbre & tableaux
# ===========================

def build_23_tree_from_list(values):
    t = Tree23()
    for v in values:
        t.insert(v)
    return t

def preorder_traversal(node, result):
    if node is None:
        return
    for k in node.keys:
        result.append(k)
    for child in node.children:
        preorder_traversal(child, result)

def tree_to_unsorted_array_preorder(tree: Tree23):
    res = []
    preorder_traversal(tree.root, res)
    return res

def format_tree_levels(tree: Tree23):
    lvls = tree.levels()
    if not lvls:
        return ["(arbre vide)"]
    lines = []
    for i, nodes in enumerate(lvls):
        parts = ["[" + ", ".join(str(k) for k in keys) + "]" for keys in nodes]
        lines.append(f"Niveau {i} : " + " | ".join(parts))
    return lines


# ===========================
# Quicksort avec étapes (sur une COPIE)
# ===========================

def quicksort_with_steps(arr):
    """
    On travaille sur une copie locale du tableau pour ne PAS toucher
    au tableau original affiché avant / après tri.
    Retourne (sorted_array, steps)
    """
    a = arr[:]  # copie indépendante
    steps = []

    def add_step(t, **k):
        s = {"type": t, "array": a.copy()}
        s.update(k)
        steps.append(s)

    def _qs(low, high, depth):
        if low < high:
            pivot = a[high]
            add_step("partition", low=low, high=high, pivot=pivot, depth=depth)

            i = low - 1
            for j in range(low, high):
                if a[j] <= pivot:
                    i += 1
                    a[i], a[j] = a[j], a[i]
                    add_step("swap", i=i, j=j, low=low, high=high, pivot=pivot, depth=depth)

            a[i + 1], a[high] = a[high], a[i + 1]
            add_step("pivot_fixed", i=i + 1, pivot=pivot, low=low, high=high, depth=depth)

            p = i + 1
            _qs(low, p - 1, depth + 1)
            _qs(p + 1, high, depth + 1)

        elif low == high:
            add_step("single", index=low, depth=depth)

    if len(a) > 0:
        _qs(0, len(a) - 1, 0)

    return a, steps


def render_step(step):
    """Affichage visuel d'une étape de quicksort avec des cases colorées."""
    arr = step["array"]
    t = step["type"]

    if t == "partition":
        st.markdown(
            f"<span class='step-label'>Début de partition</span> "
            f"(low={step['low']}, high={step['high']}), pivot = <b>{step['pivot']}</b>.",
            unsafe_allow_html=True
        )
    elif t == "swap":
        st.markdown(
            f"<span class='step-label'>Échange</span> des éléments aux indices i={step['i']}, j={step['j']} "
            f"(pivot = <b>{step['pivot']}</b>).",
            unsafe_allow_html=True
        )
    elif t == "pivot_fixed":
        st.markdown(
            f"<span class='step-label'>Pivot fixé</span> : <b>{step['pivot']}</b> à la position {step['i']}.",
            unsafe_allow_html=True
        )
    elif t == "single":
        st.markdown(
            f"<span class='step-label'>Sous-tableau de taille 1</span> : élément à l'indice {step['index']} déjà trié.",
            unsafe_allow_html=True
        )

    html = "<div class='array-row'>"
    for i, v in enumerate(arr):
        cls = "box box-normal"
        if t == "partition" and i == step.get("high"):
            cls = "box box-pivot"
        if t == "swap" and (i == step.get("i") or i == step.get("j")):
            cls = "box box-swap"
        if t == "pivot_fixed" and i == step.get("i"):
            cls = "box box-fixed"
        if t == "single" and i == step.get("index"):
            cls = "box box-fixed"
        html += f"<div class='{cls}'>{v}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ===========================
# Arbre graphique (même style TP2)
# ===========================

def build_graph_from_tree(tree: Tree23):
    G = nx.DiGraph()
    pos, labels = {}, {}

    if not tree.root:
        return G, pos, labels

    def label(node): 
        return "[" + ", ".join(str(k) for k in node.keys) + "]"

    def layout(node, x, y, dx):
        node_lbl = label(node)
        G.add_node(node_lbl)
        labels[node_lbl] = node_lbl
        pos[node_lbl] = (x, y)
        for i, c in enumerate(node.children):
            cx = x + (i - (len(node.children) - 1)/2.0) * dx
            cy = y - 1.5
            child_lbl = label(c)
            G.add_edge(node_lbl, child_lbl)
            layout(c, cx, cy, dx/2)

    layout(tree.root, 0, 0, 6)
    return G, pos, labels

def draw_tree(tree: Tree23, title):
    G, pos, labels = build_graph_from_tree(tree)
    fig, ax = plt.subplots(figsize=(8, 5))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(Arbre vide)", ha='center', va='center')
        ax.axis('off')
        return fig
    nx.draw(
        G, pos,
        labels=labels,
        node_color='darkred',
        node_size=2200,
        font_color='white',
        edgecolors='black'
    )
    ax.set_title(title, fontsize=14, color='darkred')
    ax.axis('off')
    plt.tight_layout()
    return fig


# ===========================
# Session State
# ===========================

if "tree_initial" not in st.session_state:
    st.session_state.tree_initial = None
if "tree_final" not in st.session_state:
    st.session_state.tree_final = None
if "qs_unsorted" not in st.session_state:
    st.session_state.qs_unsorted = []
if "qs_sorted" not in st.session_state:
    st.session_state.qs_sorted = []
if "qs_steps" not in st.session_state:
    st.session_state.qs_steps = []
if "qs_index" not in st.session_state:
    st.session_state.qs_index = 0
if "show_steps" not in st.session_state:
    st.session_state.show_steps = False


# ===========================
# 1) Saisie + exécution globale
# ===========================

st.markdown('<div class="bloc"><div class="step-title">1️⃣ Saisie des valeurs</div>', unsafe_allow_html=True)

user_input = st.text_input(
    "Entrez les valeurs (ex: 8,3,1,6,4,10,14)",
    value="8,3,1,6,4,10,14"
)


if st.button("Lancer TP3"):
    try:
        values = [int(x.strip()) for x in user_input.split(",") if x.strip()]
        if not values:
            st.error("Veuillez entrer au moins une valeur.")
        else:
            # Arbre initial
            tree_initial = build_23_tree_from_list(values)
            st.session_state.tree_initial = tree_initial

            # Tableau NON trié (préfixe)
            unsorted_arr = tree_to_unsorted_array_preorder(tree_initial)
            st.session_state.qs_unsorted = unsorted_arr

            # Tri rapide (sur copie) + étapes
            sorted_arr, steps = quicksort_with_steps(unsorted_arr)
            st.session_state.qs_sorted = sorted_arr
            st.session_state.qs_steps = steps
            st.session_state.qs_index = 0
            st.session_state.show_steps = False  # on cache les étapes tant que le bouton n'est pas cliqué

            # Nouvel arbre 2-3 construit À PARTIR du tableau trié (fixe)
            tree_final = build_23_tree_from_list(sorted_arr)
            st.session_state.tree_final = tree_final

            st.success("TP3 exécuté : arbre initial, tableau avant/après tri et nouvel arbre générés.")
    except ValueError:
        st.error("Veuillez entrer uniquement des entiers valides.")
st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 2) Arbre initial
# ===========================

if st.session_state.tree_initial:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="bloc"><div class="step-title">2️⃣ Arbre 2-3 initial</div>', unsafe_allow_html=True)
        for line in format_tree_levels(st.session_state.tree_initial):
            st.code(line, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        fig1 = draw_tree(st.session_state.tree_initial, "Arbre 2-3 initial")
        st.pyplot(fig1)


# ===========================
# 3) Tableau avant / après tri (fixes)
# ===========================

if st.session_state.qs_unsorted:
    st.markdown('<div class="bloc"><div class="step-title">3️⃣ Tableau issu du parcours préfixe (NON trié)</div>', unsafe_allow_html=True)
    st.markdown(f"<span class='code-like'>{st.session_state.qs_unsorted}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.qs_sorted:
    st.markdown('<div class="bloc"><div class="step-title">4️⃣ Tableau après tri rapide (Quicksort)</div>', unsafe_allow_html=True)
    st.markdown(f"<span class='code-like'>{st.session_state.qs_sorted}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 4) Nouvel arbre 2-3 après tri (fixe)
# ===========================

if st.session_state.tree_final:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="bloc"><div class="step-title">5️⃣ Nouvel arbre 2-3 après tri + réinsertion</div>', unsafe_allow_html=True)
        for line in format_tree_levels(st.session_state.tree_final):
            st.code(line, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        fig2 = draw_tree(st.session_state.tree_final, "Nouvel arbre 2-3 (après tri)")
        st.pyplot(fig2)


# ===========================
# 5) Bouton pour afficher les étapes (sans toucher aux arbres)
# ===========================

if st.session_state.qs_steps:
    st.markdown('<div class="bloc">', unsafe_allow_html=True)

    if not st.session_state.show_steps:
        if st.button("Afficher les étapes du tri rapide (visualisation)"):
            st.session_state.show_steps = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-title">6️⃣ Étapes détaillées du tri rapide</div>', unsafe_allow_html=True)

        steps = st.session_state.qs_steps
        idx = st.session_state.qs_index

        col_prev, col_info, col_next = st.columns([1, 3, 1])

        with col_prev:
            if st.button("◀️ Précédent", disabled=(idx <= 0)):
                st.session_state.qs_index = max(0, idx - 1)
                st.rerun()

        with col_next:
            if st.button("Suivant ▶️", disabled=(idx >= len(steps) - 1)):
                st.session_state.qs_index = min(len(steps) - 1, idx + 1)
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align:center;' class='small-label'>Étape {idx + 1} / {len(steps)}</div>",
                unsafe_allow_html=True
            )

        current_step = steps[idx]
        render_step(current_step)

        st.markdown(
            f"<div class='small-label'>Le tableau trié final reste : <b>{st.session_state.qs_sorted}</b> (fixe, indépendant de la navigation).</div>",
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)