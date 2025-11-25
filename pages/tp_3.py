import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# ===========================
# CONFIG & STYLE
# ===========================

st.set_page_config(page_title="TP3 - Tri Rapide & Arbre 2-3", layout="wide")

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

    /* Nouveaux styles pour les listes gauche/droite lors du choix du pivot */
    .box-left {
        background-color: #d9ead3;  /* vert clair */
        border-color: #38761d;
    }

    .box-right {
        background-color: #f4cccc;  /* rouge très clair */
        border-color: #cc0000;
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
    
    /* ✅ Selectbox labels */
    .main [data-testid="stSelectbox"] label {
        color: #262730 !important;
    }
    
    /* ✅ Ensure buttons keep white text */
    .main .stButton>button {
        color: white !important;
    }
    
    /* ✅ Ensure styled titles keep their colors */
    .tp3-title, .step-title {
        color: darkred !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="tp3-title">TP3 - Tri Rapide (Quicksort) & Arbre 2-3</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tp3-subtitle">
L'utilisateur saisit un tableau → Tri rapide (pivot début / fin / médiane) → Visualisation des étapes + Arbre 2-3 construit nœud par nœud.
</div>
""", unsafe_allow_html=True)


# ===========================
# Implémentation Arbre 2-3
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

        def _insert(node, key):
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

            split_result = _insert(node.children[idx], key)
            if split_result is None:
                return None

            left, middle_key, right = split_result
            node.keys.insert(idx, middle_key)
            node.children[idx] = left
            node.children.insert(idx + 1, right)

            if len(node.keys) <= 2:
                return None
            return self._split_node(node)

        res = _insert(self.root, key)
        if res is not None:
            left, middle_key, right = res
            self.root = Node23([middle_key], [left, right])
        return self.root

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


def build_23_tree_from_list(values):
    t = Tree23()
    for v in values:
        t.insert(v)
    return t


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
# Choix du pivot (3 cas)
# ===========================

PIVOT_LABELS = {
    "Pivot au début": "start",
    "Pivot à la fin": "end",
    "Pivot médiane (début, milieu, fin)": "median"
}
DEFAULT_PIVOT_LABEL = "Pivot à la fin"
reverse_labels = {v: k for k, v in PIVOT_LABELS.items()}

def choose_pivot_index(arr, low, high, mode_key):
    """
    - start  -> low
    - end    -> high
    - median -> élément du milieu (indice (low + high)//2)
    """
    if mode_key == "start":
        return low
    if mode_key == "end":
        return high
    if mode_key == "median":
        return (low + high) // 2
    return high


# ===========================
# Quicksort avec étapes + valeurs fixées
# ===========================

def quicksort_with_steps(arr, pivot_mode_key):
    """
    Tri rapide sur une COPIE du tableau arr.
    Retourne (sorted_array, steps) avec :
      - array : état global du tableau
      - fixed : valeurs déjà fixées
    Chaque étape contient aussi low/high pour afficher uniquement la "table" courante.
    """
    a = arr[:]
    steps = []
    fixed_values = []

    def add_step(t, **k):
        s = {
            "type": t,
            "array": a.copy(),
            "fixed": fixed_values.copy()
        }
        s.update(k)
        steps.append(s)

    def _qs(low, high, depth):
        if low < high:
            chosen_idx = choose_pivot_index(a, low, high, pivot_mode_key)
            chosen_value = a[chosen_idx]

            # Étape : choix du pivot sur le sous-tableau [low..high]
            add_step(
                "choose_pivot",
                low=low,
                high=high,
                pivot=chosen_value,
                pivot_idx=chosen_idx,
                depth=depth
            )

            # On met le pivot à la fin (schéma de Lomuto)
            if chosen_idx != high:
                a[chosen_idx], a[high] = a[high], a[chosen_idx]
            pivot = a[high]
            pivot_idx = high

            add_step(
                "partition",
                low=low,
                high=high,
                pivot=pivot,
                pivot_idx=pivot_idx,
                depth=depth
            )

            i = low - 1
            for j in range(low, high):
                if a[j] <= pivot:
                    i += 1
                    a[i], a[j] = a[j], a[i]
                    add_step(
                        "swap",
                        i=i,
                        j=j,
                        low=low,
                        high=high,
                        pivot=pivot,
                        pivot_idx=high,
                        depth=depth
                    )

            a[i + 1], a[high] = a[high], a[i + 1]
            pivot_idx = i + 1
            fixed_values.append(a[pivot_idx])
            add_step(
                "pivot_fixed",
                i=pivot_idx,
                low=low,
                high=high,
                pivot=a[pivot_idx],
                pivot_idx=pivot_idx,
                depth=depth
            )

            p = pivot_idx
            _qs(low, p - 1, depth + 1)
            _qs(p + 1, high, depth + 1)

            # À ce moment-là, le sous-tableau [low..high] est trié → nouvelle "table"
            add_step(
                "segment_sorted",
                low=low,
                high=high,
                depth=depth
            )

        elif low == high:
            fixed_values.append(a[low])
            add_step(
                "single",
                index=low,
                low=low,
                high=high,
                pivot_idx=None,
                depth=depth
            )
            # Sous-tableau de taille 1 déjà trié → on peut aussi considérer que c'est une table finale
            add_step(
                "segment_sorted",
                low=low,
                high=high,
                depth=depth
            )

    if len(a) > 0:
        _qs(0, len(a) - 1, 0)

    return a, steps


# ===========================
# Affichage d'une étape
# ===========================

def render_step(step):
    arr = step["array"]
    t = step["type"]
    pivot_idx = step.get("pivot_idx", None)
    low = step.get("low", None)
    high = step.get("high", None)

    # ----- Cas spécial : choix du pivot -----
    if t == "choose_pivot":
        pivot = step["pivot"]
        low, high = step["low"], step["high"]

        st.markdown(
            f"<span class='step-label'>Choix du pivot</span> : "
            f"valeur <b>{pivot}</b> (indice {step['pivot_idx']}) "
            f"dans le sous-tableau [{low}..{high}].",
            unsafe_allow_html=True
        )

        # 1) Sous-tableau courant avec pivot en rouge
        html_sub = "<div class='array-row'>"
        for i in range(low, high + 1):
            v = arr[i]
            cls = "box box-normal"
            if i == pivot_idx:
                cls = "box box-pivot"
            html_sub += f"<div class='{cls}'>{v}</div>"
        html_sub += "</div>"

        st.markdown("<div class='small-label'>Sous-tableau courant :</div>", unsafe_allow_html=True)
        st.markdown(html_sub, unsafe_allow_html=True)

        # 2) Listes gauche/droite en travaillant sur ce sous-tableau uniquement
        segment = arr[low:high+1]
        left_elems = [x for x in segment if x < pivot]
        right_elems = [x for x in segment if x > pivot]

        st.markdown("<div class='small-label'>Éléments &lt; pivot (gauche) :</div>", unsafe_allow_html=True)
        if left_elems:
            html_left = "<div class='array-row'>"
            for v in left_elems:
                html_left += f"<div class='box box-left'>{v}</div>"
            html_left += "</div>"
            st.markdown(html_left, unsafe_allow_html=True)
        else:
            st.markdown("<div class='small-label'>Aucun élément strictement plus petit que le pivot.</div>", unsafe_allow_html=True)

        st.markdown("<div class='small-label'>Éléments &gt; pivot (droite) :</div>", unsafe_allow_html=True)
        if right_elems:
            html_right = "<div class='array-row'>"
            for v in right_elems:
                html_right += f"<div class='box box-right'>{v}</div>"
            html_right += "</div>"
            st.markdown(html_right, unsafe_allow_html=True)
        else:
            st.markdown("<div class='small-label'>Aucun élément strictement plus grand que le pivot.</div>", unsafe_allow_html=True)

        return

    # ----- Segment trié : nouvelle table verte -----
    if t == "segment_sorted":
        low, high = step["low"], step["high"]
        st.markdown(
            f"<span class='step-label'>Sous-tableau trié</span> sur l'intervalle [{low}..{high}] → nouvelle table :",
            unsafe_allow_html=True
        )
        html_sorted = "<div class='array-row'>"
        for i in range(low, high + 1):
            v = arr[i]
            html_sorted += f"<div class='box box-fixed'>{v}</div>"
        html_sorted += "</div>"
        st.markdown(html_sorted, unsafe_allow_html=True)
        return

    # ----- Autres types d'étapes : partition, swap, pivot_fixed, single -----
    if t == "partition":
        st.markdown(
            f"<span class='step-label'>Début de partition</span> "
            f"(low={low}, high={high}), pivot = <b>{step['pivot']}</b> "
            f"(déplacé à la fin du sous-tableau).",
            unsafe_allow_html=True
        )
    elif t == "swap":
        st.markdown(
            f"<span class='step-label'>Échange</span> des éléments aux indices i={step['i']}, j={step['j']} "
            f"(pivot = <b>{step['pivot']}</b> à droite).",
            unsafe_allow_html=True
        )
    elif t == "pivot_fixed":
        st.markdown(
            f"<span class='step-label'>Pivot fixé</span> : <b>{step['pivot']}</b> à la position {step['i']}.",
            unsafe_allow_html=True
        )
    elif t == "single":
        st.markdown(
            f"<span class='step-label'>Sous-tableau de taille 1</span> : "
            f"élément à l'indice {step['index']} déjà trié.",
            unsafe_allow_html=True
        )

    # On travaille uniquement sur la "table" courante [low..high]
    if low is not None and high is not None:
        html = "<div class='array-row'>"
        for i in range(low, high + 1):
            v = arr[i]
            cls = "box box-normal"

            if t in ("partition", "swap") and pivot_idx is not None and i == pivot_idx:
                cls = "box box-pivot"

            if t == "pivot_fixed" and i == step.get("i"):
                cls = "box box-fixed"

            if t == "single" and i == step.get("index"):
                cls = "box box-fixed"

            html += f"<div class='{cls}'>{v}</div>"
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)


# ===========================
# Arbre graphique
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
if "pivot_mode" not in st.session_state:
    st.session_state.pivot_mode = PIVOT_LABELS[DEFAULT_PIVOT_LABEL]
if "tree_from_sorted" not in st.session_state:
    st.session_state.tree_from_sorted = None


# ===========================
# 1) Saisie + exécution globale
# ===========================

st.markdown('<div class="bloc"><div class="step-title">1️⃣ Saisie du tableau</div>', unsafe_allow_html=True)

user_input = st.text_input(
    "Entrez les valeurs du tableau (ex: 8,3,1,6,4,10,14)",
    value="8,3,1,6,4,10,14"
)

current_label = reverse_labels.get(st.session_state.pivot_mode, DEFAULT_PIVOT_LABEL)
pivot_label_options = list(PIVOT_LABELS.keys())

pivot_label = st.selectbox(
    "Méthode de sélection du pivot pour le tri rapide :",
    pivot_label_options,
    index=pivot_label_options.index(current_label)
)

if st.button("Lancer le tri rapide"):
    try:
        values = [int(x.strip()) for x in user_input.split(",") if x.strip()]
        if not values:
            st.error("Veuillez entrer au moins une valeur.")
        else:
            st.session_state.pivot_mode = PIVOT_LABELS[pivot_label]

            st.session_state.qs_unsorted = values

            sorted_arr, steps = quicksort_with_steps(values, st.session_state.pivot_mode)
            st.session_state.qs_sorted = sorted_arr
            st.session_state.qs_steps = steps
            st.session_state.qs_index = 0
            st.session_state.show_steps = False

            st.session_state.tree_from_sorted = build_23_tree_from_list(sorted_arr)

            st.success(
                f"Tri rapide exécuté avec {pivot_label}. "
                "Tableau trié et arbre 2-3 final construits."
            )
    except ValueError:
        st.error("Veuillez entrer uniquement des entiers valides.")

st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 2) Tableau avant / après tri
# ===========================

if st.session_state.qs_unsorted:
    st.markdown('<div class="bloc"><div class="step-title">2️⃣ Tableau initial (avant tri)</div>', unsafe_allow_html=True)
    st.markdown(f"<span class='code-like'>{st.session_state.qs_unsorted}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.qs_sorted:
    st.markdown('<div class="bloc"><div class="step-title">3️⃣ Tableau après tri rapide</div>', unsafe_allow_html=True)
    st.markdown(f"<span class='code-like'>{st.session_state.qs_sorted}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 3) Arbre 2-3 final (tableau trié)
# ===========================

if st.session_state.tree_from_sorted:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="bloc"><div class="step-title">4️⃣ Arbre 2-3 final (tableau trié)</div>', unsafe_allow_html=True)
        for line in format_tree_levels(st.session_state.tree_from_sorted):
            st.code(line, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        fig2 = draw_tree(st.session_state.tree_from_sorted, "Arbre 2-3 (à partir du tableau trié)")
        st.pyplot(fig2)


# ===========================
# 4) Étapes détaillées + arbre nœud par nœud
# ===========================

if st.session_state.qs_steps:
    st.markdown('<div class="bloc">', unsafe_allow_html=True)

    if not st.session_state.show_steps:
        if st.button("Afficher les étapes du tri rapide (visualisation)"):
            st.session_state.show_steps = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-title">5️⃣ Étapes détaillées du tri rapide + Arbre 2-3 (nœud par nœud)</div>', unsafe_allow_html=True)

        steps = st.session_state.qs_steps
        idx = st.session_state.qs_index

        col_prev, col_info, col_next = st.columns([1, 3, 1])

        with col_prev:
            if st.button("◀ Précédent", disabled=(idx <= 0)):
                st.session_state.qs_index = max(0, idx - 1)
                st.rerun()

        with col_next:
            if st.button("Suivant ▶", disabled=(idx >= len(steps) - 1)):
                st.session_state.qs_index = min(len(steps) - 1, idx + 1)
                st.rerun()

        with col_info:
            label_for_mode = reverse_labels.get(st.session_state.pivot_mode, DEFAULT_PIVOT_LABEL)
            st.markdown(
                f"<div style='text-align:center;' class='small-label'>Étape {idx + 1} / {len(steps)} "
                f"({label_for_mode})</div>",
                unsafe_allow_html=True
            )

        current_step = steps[idx]

        col_a, col_b = st.columns(2)

        with col_a:
            render_step(current_step)

        with col_b:
            fixed_vals = current_step["fixed"]
            if fixed_vals:
                sorted_fixed = sorted(fixed_vals)
                tree_step = build_23_tree_from_list(sorted_fixed)
                fig_step = draw_tree(tree_step, "Arbre 2-3 (valeurs fixées triées)")
                st.pyplot(fig_step)
            else:
                fig_empty, ax_empty = plt.subplots(figsize=(8, 5))
                ax_empty.text(0.5, 0.5, "Aucun nœud encore inséré dans l'arbre",
                              ha='center', va='center')
                ax_empty.axis('off')
                st.pyplot(fig_empty)

        st.markdown(
            "<div class='small-label'>L'arbre affiché pour chaque étape est construit à partir "
            "des valeurs déjà fixées, rangées dans l'ordre trié.</div>",
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)
