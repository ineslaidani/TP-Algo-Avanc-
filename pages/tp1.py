import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# استخدام backend غير تفاعلي لـ matplotlib
matplotlib.use('Agg')

# تخصيص الألوان والتصميم باستخدام CSS
st.markdown("""
<style>
    .stApp {
        background-color: white;
        font-family: 'Arial', sans-serif;
    }
            
        
    [data-testid="stSidebar"] {
        background-color: darkred;
    }
    
    /* تغيير لون النص في الـ Sidebar إلى الأبيض */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* تغيير لون خلفية عناصر الـ Sidebar */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #8B0000;
        color: white;
        border: 1px solid white;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #A52A2A;
        color: white;
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
</style>
""", unsafe_allow_html=True)

# تهيئة حالة الجلسة
if 'page' not in st.session_state:
    st.session_state.page = "arbre"
if 'sommets_confirmed' not in st.session_state:
    st.session_state.sommets_confirmed = False
if 'adj_matrix' not in st.session_state:
    st.session_state.adj_matrix = None

# زرين في الأعلى للاختيار
st.markdown("### Choisir la section:")
col1, col2 = st.columns(2)

with col1:
    if st.button("Arbre", use_container_width=True, key="btn_arbre"):
        st.session_state.page = "arbre"
        st.session_state.sommets_confirmed = False

with col2:
    if st.button("Graph", use_container_width=True, key="btn_graph"):
        st.session_state.page = "graph" 
        st.session_state.sommets_confirmed = False

# دوال مساعدة لحساب خصائص الشجرة
class TreeProperties:
    @staticmethod
    def calculate_tree_height(G):
        """حساب ارتفاع الشجرة (عدد الحواف في أطول مسار من الجذر إلى الورقة)"""
        if G.number_of_nodes() == 0:
            return 0
        
        # إيجاد الجذر (العقدة التي ليس لها أصل)
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        if not roots:
            return 0
        
        def dfs_height(node, visited):
            visited.add(node)
            # إذا كانت العقدة ورقة (لا أبناء)
            if G.out_degree(node) == 0:
                return 0
            max_height = 0
            for child in G.successors(node):
                if child not in visited:
                    max_height = max(max_height, dfs_height(child, visited))
            return max_height + 1
        
        return dfs_height(roots[0], set())

    @staticmethod
    def calculate_tree_degree(G):
        """حساب درجة الشجرة (أعلى درجة بين العقد)"""
        if G.number_of_nodes() == 0:
            return 0
        # استخدام out_degree للأشجار الموجهة للحصول على عدد الأبناء الحقيقي
        return max(dict(G.out_degree()).values())

    @staticmethod
    def calculate_tree_density(G):
        """حساب كثافة الشجرة"""
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        # في الشجرة المتجهة، الكثافة = عدد الحواف / (عدد العقد * (عدد العقد - 1))
        m = G.number_of_edges()
        return m / (n * (n - 1)) if n > 1 else 0.0

    @staticmethod
    def calculate_average_degree(G):
        """حساب متوسط درجة الشجرة"""
        if G.number_of_nodes() == 0:
            return 0.0
        # استخدام out_degree للأشجار الموجهة
        degrees = dict(G.out_degree()).values()
        return sum(degrees) / len(degrees)

# دوال مساعدة لحساب خصائص الرسم البياني
class GraphProperties:
    @staticmethod
    def calculate_graph_degree(G):
        """حساب درجة الرسم البياني (أعلى درجة بين العقد)"""
        if G.number_of_nodes() == 0:
            return 0
        return max(dict(G.degree()).values())

    @staticmethod
    def calculate_average_degree(G):
        """حساب متوسط درجة الرسم البياني"""
        if G.number_of_nodes() == 0:
            return 0.0
        degrees = dict(G.degree()).values()
        return sum(degrees) / len(degrees)

    @staticmethod
    def calculate_graph_density(G):
        """حساب كثافة الرسم البياني"""
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        m = G.number_of_edges()
        # للرسوم البيانية غير الموجهة
        if not G.is_directed():
            return (2 * m) / (n * (n - 1)) if n > 1 else 0.0
        # للرسوم البيانية الموجهة
        else:
            return m / (n * (n - 1)) if n > 1 else 0.0

    @staticmethod
    def get_degree_sequence(G):
        """الحصول على تسلسل الدرجات"""
        return sorted([d for n, d in G.degree()], reverse=True)

# دوال مساعدة لبناء الأشجار الحقيقية حسب القوانين الصحيحة
class TreeBuilder:
    @staticmethod
    def build_binary_search_tree(values):
        """بناء شجرة بحث ثنائية صحيحة حسب قوانين ABR"""
        G = nx.DiGraph()
        if not values:
            return G
        
        numeric_values = [int(v) for v in values]
        
        class Node:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None
        
        def insert(root, value):
            if root is None:
                return Node(value)
            if value < root.value:
                root.left = insert(root.left, value)
            else:
                root.right = insert(root.right, value)
            return root
        
        def build_graph(node, parent=None):
            if node is None:
                return
            G.add_node(str(node.value))
            if parent is not None:
                G.add_edge(str(parent.value), str(node.value))
            build_graph(node.left, node)
            build_graph(node.right, node)
        
        root = Node(numeric_values[0])
        for value in numeric_values[1:]:
            root = insert(root, value)
        
        build_graph(root)
        return G

    @staticmethod
    def build_avl_tree(values):
        """بناء شجرة AVL صحيحة - شجرة بحث ثنائية متوازنة"""
        G = nx.DiGraph()
        if not values:
            return G
        
        numeric_values = [int(v) for v in values]
        
        class AVLNode:
            def __init__(self, value):
                self.value = value
                self.left = None
                self.right = None
                self.height = 1
        
        def get_height(node):
            if not node:
                return 0
            return node.height
        
        def get_balance(node):
            if not node:
                return 0
            return get_height(node.left) - get_height(node.right)
        
        def rotate_right(y):
            x = y.left
            T2 = x.right
            
            x.right = y
            y.left = T2
            
            y.height = 1 + max(get_height(y.left), get_height(y.right))
            x.height = 1 + max(get_height(x.left), get_height(x.right))
            
            return x
        
        def rotate_left(x):
            y = x.right
            T2 = y.left
            
            y.left = x
            x.right = T2
            
            x.height = 1 + max(get_height(x.left), get_height(x.right))
            y.height = 1 + max(get_height(y.left), get_height(y.right))
            
            return y
        
        def insert(node, value):
            if not node:
                return AVLNode(value)
            
            if value < node.value:
                node.left = insert(node.left, value)
            else:
                node.right = insert(node.right, value)
            
            node.height = 1 + max(get_height(node.left), get_height(node.right))
            
            balance = get_balance(node)
            
            # Left Left Case
            if balance > 1 and value < node.left.value:
                return rotate_right(node)
            
            # Right Right Case
            if balance < -1 and value > node.right.value:
                return rotate_left(node)
            
            # Left Right Case
            if balance > 1 and value > node.left.value:
                node.left = rotate_left(node.left)
                return rotate_right(node)
            
            # Right Left Case
            if balance < -1 and value < node.right.value:
                node.right = rotate_right(node.right)
                return rotate_left(node)
            
            return node
        
        def build_graph(node, parent=None):
            if node is None:
                return
            G.add_node(str(node.value))
            if parent is not None:
                G.add_edge(str(parent.value), str(node.value))
            build_graph(node.left, node)
            build_graph(node.right, node)
        
        root = None
        for value in numeric_values:
            root = insert(root, value)
        
        build_graph(root)
        return G

    @staticmethod
    def build_heap_tree(values):
        """بناء شجرة TAS صحيحة - شجرة ثنائية كاملة مع خاصية الـ Heap"""
        G = nx.DiGraph()
        if not values:
            return G
        
        numeric_values = [int(v) for v in values]
        
        # بناء max-heap
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[left] > arr[largest]:
                largest = left
                
            if right < n and arr[right] > arr[largest]:
                largest = right
                
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        # بناء الـ heap
        n = len(numeric_values)
        for i in range(n // 2 - 1, -1, -1):
            heapify(numeric_values, n, i)
        
        # بناء الرسم البياني للشجرة
        for i in range(n):
            G.add_node(str(numeric_values[i]))
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                G.add_edge(str(numeric_values[i]), str(numeric_values[left]))
            if right < n:
                G.add_edge(str(numeric_values[i]), str(numeric_values[right]))
        
        return G

    @staticmethod
    def build_b_tree(values):
        """بناء شجرة B-arbre صحيحة - شجرة متعددة الأبناء متوازنة"""
        G = nx.DiGraph()
        if not values:
            return G
        
        numeric_values = sorted([int(v) for v in values])
        m = 3  # درجة الشجرة
        
        class BNode:
            def __init__(self):
                self.keys = []
                self.children = []
                self.is_leaf = True
            
            def add_key(self, key):
                self.keys.append(key)
                self.keys.sort()
        
        def build_b_tree_sorted(sorted_values, degree):
            if not sorted_values:
                return None
                
            root = BNode()
            n = len(sorted_values)
            
            if n <= degree:
                root.keys = sorted_values
                return root
                
            # تقسيم القيم
            mid = n // 2
            root.keys = [sorted_values[mid]]
            
            left_values = sorted_values[:mid]
            right_values = sorted_values[mid+1:]
            
            if left_values:
                left_child = build_b_tree_sorted(left_values, degree)
                if left_child:
                    root.children.append(left_child)
                    root.is_leaf = False
                    
            if right_values:
                right_child = build_b_tree_sorted(right_values, degree)
                if right_child:
                    root.children.append(right_child)
                    root.is_leaf = False
                    
            return root
        
        def build_graph(node, parent=None):
            if node is None:
                return
                
            node_label = "/".join(str(k) for k in node.keys)
            G.add_node(node_label)
            
            if parent is not None:
                G.add_edge(parent, node_label)
                
            for child in node.children:
                build_graph(child, node_label)
        
        root = build_b_tree_sorted(numeric_values, m)
        build_graph(root)
        return G

    @staticmethod
    def build_amr_tree(values):
        """بناء شجرة AMR صحيحة - Arbre m-aire de recherche"""
        G = nx.DiGraph()
        if not values:
            return G
        
        numeric_values = sorted([int(v) for v in values])
        m = 3  # درجة الشجرة
        
        class MNode:
            def __init__(self):
                self.values = []
                self.children = []
            
            def add_value(self, value):
                self.values.append(value)
                self.values.sort()
        
        def build_m_ary_search_tree(sorted_values, arity):
            if not sorted_values:
                return None
                
            node = MNode()
            n = len(sorted_values)
            
            if n <= arity:
                node.values = sorted_values
                return node
                
            # تقسيم القيم مع الحفاظ على ترتيب البحث
            chunk_size = max(1, n // arity)
            chunks = [sorted_values[i:i + chunk_size] for i in range(0, n, chunk_size)]
            
            if len(chunks) > arity:
                # دمج الأخيرين إذا تجاوزنا العدد
                chunks[-2:] = [chunks[-2] + chunks[-1]]
            
            for chunk in chunks:
                if len(chunk) == 1:
                    node.values.append(chunk[0])
                else:
                    child = build_m_ary_search_tree(chunk, arity)
                    if child:
                        node.children.append(child)
                        node.values.append(child.values[len(child.values)//2])
            
            node.values.sort()
            return node
        
        def build_graph(node, parent=None):
            if node is None:
                return
                
            node_label = "/".join(str(v) for v in node.values)
            G.add_node(node_label)
            
            if parent is not None:
                G.add_edge(parent, node_label)
                
            for child in node.children:
                build_graph(child, node_label)
        
        root = build_m_ary_search_tree(numeric_values, m)
        build_graph(root)
        return G

# دوال مساعدة لبناء الرسوم البيانية من مصفوفة adjacency
class GraphBuilder:
    @staticmethod
    def create_graph_from_matrix(sommets, matrix, ponderee=False, orientee=False):
        """إنشاء رسم بياني من مصفوفة adjacency"""
        if orientee:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # إضافة القمم
        for sommet in sommets:
            G.add_node(sommet)
        
        # إضافة الحواف
        n = len(sommets)
        for i in range(n):
            for j in range(n):
                weight = matrix[i][j]
                if weight != 0:  # إذا كانت هناك حافة
                    if ponderee:
                        G.add_edge(sommets[i], sommets[j], weight=float(weight))
                    else:
                        G.add_edge(sommets[i], sommets[j])
        
        return G

# عرض المحتوى حسب الاختيار
if st.session_state.page == "arbre":
    # محتوى لـ Arbre
    st.markdown('<div class="section-title">Section Arbre</div>', unsafe_allow_html=True)
    
    # اختيار type arbre avec checkbox
    st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
    st.subheader("Type d'Arbre")
    
    # Checkboxes لأنواع الأشجار
    col1, col2, col3 = st.columns(3)
    
    with col1:
        arbre_binaire_recherche = st.checkbox("Arbre Binaire de Recherche")
        arbre_avl = st.checkbox("Arbre AVL")
        
    with col2:
        arbre_TAS = st.checkbox("Arbre TAS")
        arbre_b_arbre = st.checkbox("Arbre B-arbre")
        
    with col3:
        arbre_AMR = st.checkbox("Arbre AMR")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # إدخال القيم (les sommets)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("Entrez les valeurs des sommets")
    
    # استخدام placeholder بدلاً من النص الخارجي
    valeurs = st.text_input(
        "Valeurs:",
        placeholder="Entrez les valeurs séparées par des virgules (ex: 1,2,3,4,5):",
        key="arbre_valeurs"
    )
    
    if valeurs:
        sommets = [v.strip() for v in valeurs.split(',')]
        st.write("Sommets saisis:", sommets)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # زر Construire
    if st.button("Construire"):
        if not valeurs:
            st.warning("Veuillez entrer des valeurs pour les sommets")
        else:
            try:
                sommets = [v.strip() for v in valeurs.split(',')]
                G = None
                type_selected = ""
                
                # تحديد نوع الشجرة المختار
                if arbre_binaire_recherche:
                    G = TreeBuilder.build_binary_search_tree(sommets)
                    type_selected = "Arbre Binaire de Recherche"
                    st.info("✅ ABR: Arbre binaire de recherche (gauche < racine < droite)")
                elif arbre_avl:
                    G = TreeBuilder.build_avl_tree(sommets)
                    type_selected = "Arbre AVL"
                    st.info("✅ AVL: Arbre binaire de recherche équilibré avec rotations")
                elif arbre_TAS:
                    G = TreeBuilder.build_heap_tree(sommets)
                    type_selected = "Arbre TAS"
                    st.info("✅ TAS: Arbre binaire complet avec propriété de max-heap")
                elif arbre_b_arbre:
                    G = TreeBuilder.build_b_tree(sommets)
                    type_selected = "Arbre B-arbre"
                    st.info("✅ B-arbre: Arbre m-aire équilibré avec nœuds multiples")
                elif arbre_AMR:
                    G = TreeBuilder.build_amr_tree(sommets)
                    type_selected = "Arbre AMR"
                    st.info("✅ AMR: Arbre m-aire de recherche")
                else:
                    st.warning("Veuillez sélectionner un type d'arbre")
                    st.stop()
                
                if G and G.number_of_nodes() > 0:
                    # رسم الشجرة بشكل هرمي منظم
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # إنشاء تخطيط هرمي يدوي محسّن
                    pos = {}
                    levels = {}
                    
                    def assign_levels(node, level=0):
                        if node not in levels:
                            levels[level] = levels.get(level, [])
                            levels[level].append(node)
                            for child in G.successors(node):
                                assign_levels(child, level + 1)
                    
                    # البدء من الجذر
                    racine = [node for node in G.nodes() if G.in_degree(node) == 0]
                    if racine:
                        assign_levels(racine[0])
                    
                    # حساب المواقع بشكل هرمي جميل
                    max_level = max(levels.keys()) if levels else 0
                    for level, nodes in levels.items():
                        y = -level * 1.5  # المسافة بين المستويات
                        x_positions = np.linspace(-len(nodes), len(nodes), len(nodes))
                        for i, node in enumerate(nodes):
                            pos[node] = (x_positions[i], y)
                    
                    # رسم الشجرة مع اللون الأحمر الغامق للعقد
                    node_colors = ['darkred'] * len(G.nodes())
                    
                    nx.draw_networkx_nodes(G, pos, 
                                         node_color=node_colors,
                                         node_size=1500, 
                                         ax=ax,
                                         edgecolors='black',
                                         linewidths=2)
                    
                    nx.draw_networkx_edges(G, pos, 
                                         edge_color='black', 
                                         arrows=True, 
                                         arrowsize=25, 
                                         arrowstyle='->',
                                         width=2,
                                         ax=ax)
                    
                    nx.draw_networkx_labels(G, pos, 
                                          font_size=12, 
                                          font_weight='bold',
                                          font_color='white',
                                          ax=ax)
                    
                    ax.set_title(f"{type_selected}\nValeurs: {', '.join(sommets)}", 
                                fontsize=16, fontweight='bold', pad=20)
                    ax.axis('off')
                    
                    # تحسين المسافات
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # معلومات عن الشجرة
                    st.success(f"✅ {type_selected} construit avec succès!")
                    
                    # معلومات عامة
                    racine = [node for node in G.nodes() if G.in_degree(node) == 0]
                    if racine:
                        st.write(f"**Racine:** {racine[0]}")
                    
                    st.write(f"**Nombre de nœuds:** {G.number_of_nodes()}")
                    st.write(f"**Nombre d'arêtes:** {G.number_of_edges()}")
                    
                    # عرض خصائص الشجرة
                    st.markdown('<div class="properties-container">', unsafe_allow_html=True)
                    st.subheader("📊 Propriétés de l'Arbre")
                    
                    # حساب الخصائص
                    hauteur = TreeProperties.calculate_tree_height(G)
                    degree_max = TreeProperties.calculate_tree_degree(G)
                    densite = TreeProperties.calculate_tree_density(G)
                    degree_moyen = TreeProperties.calculate_average_degree(G)
                    
                    # عرض الخصائص
                    st.markdown(f'<div class="property-item">📏 <strong>Hauteur:</strong> {hauteur}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="property-item">🎯 <strong>Degré maximal:</strong> {degree_max}</div>', unsafe_allow_html=True)
                    #st.markdown(f'<div class="property-item">📊 <strong>Degré moyen:</strong> {degree_moyen:.2f}</div>', unsafe_allow_html=True)
                    #st.markdown(f'<div class="property-item">📈 <strong>Densité:</strong> {densite:.4f}</div>', unsafe_allow_html=True)
                    
                    # تحقق إضافي من الدرجة
                    degrees_info = dict(G.out_degree())
                    st.markdown(f'<div class="property-item">🔍 <strong>Détail des degrés:</strong></div>', unsafe_allow_html=True)
                    for node, deg in degrees_info.items():
                        st.markdown(f'<div style="margin-left: 20px;">• {node}: degré {deg}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.error("Erreur: Impossible de construire l'arbre")
                
            except Exception as e:
                st.error(f"Erreur lors de la construction: {str(e)}")

else:  # st.session_state.page == "graph"
    # محتوى لـ Graph
    st.markdown('<div class="section-title">Section Graph</div>', unsafe_allow_html=True)
    
    # تحديد نوع الرسم البياني كـ "Graph Non-Arbre" دائماً
    st.session_state.graph_type = "graph_non_arbre"
    
    # اختيار خصائص الغراف
    st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
    st.subheader("Propriétés du Graph")
    
    col1, col2 = st.columns(2)
    with col1:
        graph_ponderee = st.checkbox("Pondérée", key="ponderee")
    with col2:
        graph_orientee = st.checkbox("Orientée", key="orientee")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # إدخال القيم (les sommets)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("Entrez les valeurs des sommets")
    
    valeurs_graph = st.text_input(
        "Sommets:",
        placeholder="Entrez les valeurs séparées par des virgules (ex: 1,2,3,4,5):",
        key="graph_sommets"
    )
    
    if valeurs_graph:
        sommets_graph = [v.strip() for v in valeurs_graph.split(',')]
        st.write("Sommets saisis:", sommets_graph)
    
    # زر Confirmer
    if st.button("Confirmer", key="confirmer_graph"):
        if not valeurs_graph:
            st.warning("Veuillez entrer des valeurs pour les sommets")
        else:
            st.session_state.sommets_confirmed = True
            st.session_state.sommets_graph = [v.strip() for v in valeurs_graph.split(',')]
            st.session_state.adj_matrix = None
            st.success("Sommets confirmés! Veuillez remplir la matrice d'adjacence.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # عرض مصفوفة adjacency بعد التأكيد
    if st.session_state.get('sommets_confirmed', False):
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        st.subheader("Matrice d'Adjacence")
        
        sommets = st.session_state.sommets_graph
        n = len(sommets)
        
        st.write("Remplissez la matrice d'adjacence avec des 0 et 1:")
        
        # إنشاء مصفوفة ابتدائية
        if st.session_state.adj_matrix is None:
            initial_matrix = [[0 for _ in range(n)] for _ in range(n)]
            st.session_state.adj_matrix = initial_matrix
        
        # عرض المصفوفة كجدول مبسط
        st.write("**Matrice d'adjacence:**")
        
        # إنشاء DataFrame للمصفوفة
        matrix_data = st.session_state.adj_matrix.copy()
        df = pd.DataFrame(matrix_data, index=sommets, columns=sommets)
        
        # استخدام data_editor للسماح للمستخدم بتعديل القيم
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            height=min(300, 35 * (n + 1)),
            key="matrix_editor"
        )
        
        # تحديث المصفوفة في حالة الجلسة
        st.session_state.adj_matrix = edited_df.values.tolist()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # زر Construire le Graph
        if st.button("Construire", key="construire_graph"):
            try:
                matrix = st.session_state.adj_matrix
                sommets = st.session_state.sommets_graph
                
                # معلومات عن الخيارات
                properties = ""
                if graph_orientee and graph_ponderee:
                    properties = "Orienté et Pondéré"
                elif graph_orientee:
                    properties = "Orienté"
                elif graph_ponderee:
                    properties = "Pondéré"
                else:
                    properties = "Non Orienté"
                
                # بناء Graph Non-Arbre
                st.info(f"Type: Graph Non-Arbre ({properties})")
                
                G = GraphBuilder.create_graph_from_matrix(
                    sommets, matrix, 
                    ponderee=graph_ponderee, 
                    orientee=graph_orientee
                )
                
                # رسم الرسم البياني
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # تحديد التخطيط
                pos = nx.spring_layout(G, seed=42)
                
                # رسم العقد
                nx.draw_networkx_nodes(G, pos, 
                                     node_color='darkred',
                                     node_size=800, 
                                     ax=ax,
                                     edgecolors='black',
                                     linewidths=2)
                
                # رسم الحواف
                if graph_orientee:
                    nx.draw_networkx_edges(G, pos, 
                                         edge_color='gray',
                                         arrows=True,
                                         arrowsize=20,
                                         arrowstyle='->',
                                         width=2,
                                         ax=ax)
                else:
                    nx.draw_networkx_edges(G, pos, 
                                         edge_color='gray',
                                         width=2,
                                         ax=ax)
                
                # رسم تسميات الأوزان إذا كانت موزونة
                if graph_ponderee:
                    edge_labels = nx.get_edge_attributes(G, 'weight')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
                
                # رسم التسميات
                nx.draw_networkx_labels(G, pos, 
                                      font_size=12, 
                                      font_weight='bold',
                                      font_color='white',
                                      ax=ax)
                
                ax.set_title(f"Graph Non-Arbre ({properties})\nSommets: {', '.join(sommets)}", 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # معلومات عن الرسم البياني
                st.success(f"✅ Graph Non-Arbre construit avec succès!")
                st.write(f"**Nombre de sommets:** {G.number_of_nodes()}")
                st.write(f"**Nombre d'arêtes:** {G.number_of_edges()}")
                
                # عرض خصائص الرسم البياني
                st.markdown('<div class="properties-container">', unsafe_allow_html=True)
                st.subheader("📊 Propriétés du Graph")
                
                # حساب خصائص الرسم البياني
                degree_max = GraphProperties.calculate_graph_degree(G)
                degree_moyen = GraphProperties.calculate_average_degree(G)
                densite = GraphProperties.calculate_graph_density(G)
                degree_sequence = GraphProperties.get_degree_sequence(G)
                
                # عرض الخصائص
                st.markdown(f'<div class="property-item">🎯 <strong>Degré maximal:</strong> {degree_max}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="property-item">📊 <strong>Degré moyen:</strong> {degree_moyen:.2f}</div>', unsafe_allow_html=True)
                #st.markdown(f'<div class="property-item">📈 <strong>Densité:</strong> {densite:.4f}</div>', unsafe_allow_html=True)
                #st.markdown(f'<div class="property-item">📋 <strong>Séquence des degrés:</strong> {degree_sequence}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # عرض مصفوفة adjacency النهائية
                st.write("**Matrice d'adjacence utilisée:**")
                st.dataframe(df)
                
            except Exception as e:
                st.error(f"Erreur lors de la construction du graph: {str(e)}")