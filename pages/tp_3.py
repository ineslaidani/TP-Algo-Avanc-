import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

st.set_page_config(page_title="Visualisation Quicksort", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; font-family: 'Arial', sans-serif; }
    [data-testid="stSidebar"] { background-color: darkred; }
    [data-testid="stSidebar"] * { color: white !important; }
    .tp3-title { font-size: 26px; font-weight: bold; color: darkred; text-align: center; margin: 10px 0 5px 0; }
    .bloc { background-color: #f8f9fa; padding: 14px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 18px; }
    .step-title { font-size: 18px; font-weight: bold; color: darkred; margin-bottom: 6px; }
    .array-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 10px; }
    .box { min-width: 45px; padding: 10px; text-align: center; border-radius: 4px; border: 2px solid #333; font-weight: bold; font-size: 18px; color: #262730; background-color: white; }
    .box-pivot { background-color: #ff9900; color: black; border-color: #e68a00; }
    .box-sorted { background-color: #d9ead3; border-color: #38761d; color: #274e13; }
    .stButton>button { background-color: darkred; color: white; border-radius: 18px; border: none; }
    p, span, label, div { color: #262730 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="tp3-title">Visualisation Arborescente du Quicksort</div>', unsafe_allow_html=True)

PIVOT_LABELS = {
    "Pivot au début": "start",
    "Pivot à la fin": "end",
    "Pivot médiane": "median"
}

def choose_pivot_index(arr, low, high, mode_key):
    if mode_key == "start": return low
    if mode_key == "end": return high
    return (low + high) // 2

def quicksort_tree_trace(arr, pivot_mode):
    steps = []
    tree_nodes = []
    
    def _qs(sub_arr, depth, left_bound, right_bound, parent_id=None):
        if not sub_arr: return
        
        node_id = len(tree_nodes)
        mid_x = (left_bound + right_bound) / 2
        
        chosen_idx = choose_pivot_index(sub_arr, 0, len(sub_arr) - 1, pivot_mode)
        pivot_val = sub_arr[chosen_idx]
        
        tree_nodes.append({
            "id": node_id,
            "parent": parent_id,
            "array": sub_arr.copy(),
            "pivot": pivot_val,
            "depth": depth,
            "x": mid_x,
            "y": -depth
        })

        steps.append({
            "node_id": node_id,
            "array": sub_arr.copy(),
            "pivot": pivot_val,
            "depth": depth
        })

        if len(sub_arr) <= 1: return

        left_part = []
        right_part = []
        for i, x in enumerate(sub_arr):
            if i == chosen_idx: continue
            if x < pivot_val or (x == pivot_val and i < chosen_idx):
                left_part.append(x)
            else:
                right_part.append(x)
        
        _qs(left_part, depth + 1, left_bound, mid_x, node_id)
        _qs(right_part, depth + 1, mid_x, right_bound, node_id)

    _qs(arr, 0, 0, 100, None)
    return steps, tree_nodes

def draw_recursive_tree(nodes, current_step_idx):
    fig, ax = plt.subplots(figsize=(16, 9))
    G = nx.DiGraph()
    
    visible_node_ids = set(range(current_step_idx + 1))
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes if n["id"] in visible_node_ids}
    
    for n in nodes:
        if n["id"] in visible_node_ids:
            G.add_node(n["id"])
            if n["parent"] is not None and n["parent"] in visible_node_ids:
                G.add_edge(n["parent"], n["id"])

    if not G.nodes:
        ax.axis('off')
        return fig

    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=25, edge_color='#CCCCCC', width=2, alpha=0.7)
    
    for node_id, (x, y) in pos.items():
        node_data = nodes[node_id]
        arr_str = "  ".join(map(str, node_data['array']))
        
        is_current = (node_id == current_step_idx)
        ec = "#FF9900" if is_current else "darkred"
        lw = 4 if is_current else 1.5
        
        ax.text(x, y, arr_str, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.8", fc="white", ec=ec, lw=lw))

    all_y = [n['y'] for n in nodes if n['id'] in visible_node_ids]
    ax.set_ylim(min(all_y) - 0.8, 0.5)
    ax.set_xlim(-5, 105)
    ax.axis('off')
    return fig

if "trace_steps" not in st.session_state:
    st.session_state.update({"trace_steps": [], "trace_nodes": [], "trace_index": 0, "input_arr": []})

st.markdown('<div class="bloc"><div class="step-title"></div>', unsafe_allow_html=True)
c_in1, c_in2 = st.columns([2, 1])
with c_in1:
    u_input = st.text_input("Valeurs", value="19,7,15,12,16,18,4,11,13")
with c_in2:
    p_choice = st.selectbox("Pivot", list(PIVOT_LABELS.keys()), index=1)

if st.button("Lancer la Visualisation"):
    try:
        vals = [int(x.strip()) for x in u_input.split(",") if x.strip()]
        if vals:
            st.session_state.input_arr = vals
            steps, nodes = quicksort_tree_trace(vals, PIVOT_LABELS[p_choice])
            st.session_state.update({"trace_steps": steps, "trace_nodes": nodes, "trace_index": 0})
    except ValueError:
        st.error("Erreur de format.")

if st.session_state.trace_steps:
    idx = st.session_state.trace_index
    steps = st.session_state.trace_steps
    
    c_p, c_i, c_n = st.columns([1, 2, 1])
    with c_p:
        if st.button("◀", disabled=(idx <= 0)):
            st.session_state.trace_index -= 1
            st.rerun()
    with c_n:
        if st.button("▶", disabled=(idx == len(steps)-1)):
            st.session_state.trace_index += 1
            st.rerun()
    with c_i:
        st.markdown(f"<div style='text-align:center; font-size:18px;'><b>Étape {idx+1} / {len(steps)}</b></div>", unsafe_allow_html=True)

    st.markdown('<div class="bloc">', unsafe_allow_html=True)
    curr = steps[idx]
    
    h_step = "<div class='array-row'>"
    for v in curr['array']:
        cls = "box box-pivot" if (v == curr['pivot'] and len(curr['array']) > 1) else "box"
        h_step += f"<div class='{cls}'>{v}</div>"
    h_step += "</div>"
    st.markdown(h_step, unsafe_allow_html=True)
    
    st.pyplot(draw_recursive_tree(st.session_state.trace_nodes, idx))
    
    if idx == len(steps) - 1:
        st.markdown("<div style='border-top:2px solid darkred; margin:20px 0;'></div><div class='step-title' style='text-align:center;'>Tri Terminé</div>", unsafe_allow_html=True)
        h_f = "<div class='array-row'>" + "".join([f"<div class='box box-sorted'>{x}</div>" for x in sorted(st.session_state.input_arr)]) + "</div>"
        st.markdown(h_f, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)