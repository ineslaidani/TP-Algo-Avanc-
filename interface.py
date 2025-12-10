import streamlit as st

st.set_page_config(page_title="TP Algorithmique Avancé", layout="wide")

# ===== CSS general =====
st.markdown("""
<style>
    .stApp {
        background-color: white;
        font-family: 'Arial', sans-serif;
    }

    /* ✅ Sidebar (Navbar) Style */
    [data-testid="stSidebar"] {
        background-color: darkred;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton>button {
        background-color: #8B0000;
        color: white;
        border: 1px solid white;
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #A52A2A;
        color: white;
    }

    /* ✅ Main Title */
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: darkred;
        text-align: center;
        padding: 20px;
        border: 3px solid darkred;
        border-radius: 10px;
        margin: 20px auto;
        width: 60%;
        background-color: #f8f9fa;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* ✅ Group Info Box */
    .group-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid darkred;
        margin: 20px auto;
        text-align: center;
        width: 70%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .group-title {
        font-size: 24px;
        font-weight: bold;
        color: darkred;
        margin-bottom: 15px;
    }

    .member-name {
        background-color: white;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 15px;
        text-align: center;
        width: 80%;
        margin: auto;
        color: #333 !important;
    }

    /* ✅ Ensure main content text is visible - but exclude buttons and styled elements */
    .main .block-container {
        color: #262730;
    }
    
    /* Only target plain text elements, not buttons or styled divs */
    .main .block-container p:not(.stButton p):not(.stButton *),
    .main .block-container .stMarkdown p:not(.stButton *),
    .main .block-container .stText:not(.stButton *) {
        color: #262730 !important;
    }
    
    /* ✅ Input fields text color */
    .main [data-testid="stTextInput"] input {
        color: #262730 !important;
    }
    
    .main [data-testid="stTextInput"] label {
        color: #262730 !important;
    }
    
    /* ✅ Text input placeholder */
    .main [data-testid="stTextInput"] input::placeholder {
        color: #999 !important;
    }
    
    /* ✅ Ensure buttons keep white text */
    .main .stButton>button {
        color: white !important;
    }
    
    /* ✅ Ensure styled titles keep their colors */
    .main-title, .group-title, .section-title {
        color: darkred !important;
    }

    /* ✅ Button Style */
    .stButton>button {
        background-color: darkred;
        color: white;
        border-radius: 18px;
        font-size: 16px;
        padding: 12px 25px;
        width: 150px;
        height: 45px;
        margin: 10px auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: none;
        display: block;
    }

    .stButton>button:hover {
        background-color: #8B0000;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ===== TITRE PRINCIPAL =====
st.markdown('<div class="main-title">TP Algorithmique Avancé</div>', unsafe_allow_html=True)

# ===== INFORMATIONS DU GROUPE =====
st.markdown('<div class="group-info">', unsafe_allow_html=True)
st.markdown('<div class="group-title">Group 4</div>', unsafe_allow_html=True)

members = [
    "Bengrab Meriem", "Belhadj Aya", "Mehdid Malak",
    "Kalafat Fadoua", "Ziane Hiba", "Laidani Inès"
]

# قسم الأسماء إلى صفين كل صف يحتوي 3 أسماء
row1 = members[:3]
row2 = members[3:]

cols1 = st.columns(3)
for i, col in enumerate(cols1):
    col.markdown(f'<div class="member-name">{row1[i]}</div>', unsafe_allow_html=True)

cols2 = st.columns(3)
for i, col in enumerate(cols2):
    col.markdown(f'<div class="member-name">{row2[i]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===== BOUTONS DES TPs =====
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:darkred;'>Travaux Pratiques</h4>", unsafe_allow_html=True)

tp_buttons = [
    ("TP1", "pages/tp1.py"),
    ("TP2", "pages/tp_2.py"),
    ("TP3", "pages/tp_3.py"),
    ("TP4","pages/tp_41.py"),
    ("TP5", "pages/tp_4.py"),
    ("TP6", None),
]

# ✅ صفين متمركزين (3 أزرار في كل صف)
row1_btns = tp_buttons[:3]
row2_btns = tp_buttons[3:]

# صف أول (متمركز)
cols_btns1 = st.columns([1, 1, 1, 1, 1])  # لتوسيط الأزرار
for i, (label, page) in enumerate(row1_btns, start=1):
    with cols_btns1[i]:
        if st.button(label, key=f"btn_{label}_1"):
            if page:
                st.switch_page(page)
            else:
                st.write(f"{label} à venir...")

# صف ثاني (متمركز)
cols_btns2 = st.columns([1, 1, 1, 1, 1])
for i, (label, page) in enumerate(row2_btns, start=1):
    with cols_btns2[i]:
        if st.button(label, key=f"btn_{label}_2"):
            if page:
                st.switch_page(page)
            else:
                st.write(f"{label} à venir...")
