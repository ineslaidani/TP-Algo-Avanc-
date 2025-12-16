# 🌳 TP Algorithmique Avancé - Visualisateur d'Arbres et Graphes

**Groupe 4** | Application interactive éducative pour visualiser et comprendre les algorithmes fondamentaux d'informatique.

Développée avec **Streamlit** et **Python**, cette plateforme permet d'explorer visuellement les structures de données et algorithmes clés : arbres binaires, graphes, algorithmes de tri et de plus courts chemins.

## 🌐 Application en ligne

**👉 [Accéder à l'application](https://tpalgo-groupe4.streamlit.app)**

---

## ✨ Fonctionnalités

### 📊 TP1 - Arbres et Graphes

#### 🌲 **Section Arbres**
Visualisation interactive de différents types d'arbres avec leurs propriétés :

- **Arbre Binaire de Recherche (ABR)** : Insertion selon la propriété de recherche binaire
- **Arbre AVL** : Arbre binaire de recherche auto-équilibré avec rotations
- **Arbre TAS** : Structure de tas (heap) avec propriété de max-heap
- **Arbre B-arbre** : Arbre m-aire équilibré avec nœuds multiples
- **Arbre AMR** : Arbre m-aire de recherche

**Fonctionnalités :**
- Construction interactive à partir de valeurs saisies
- Visualisation hiérarchique avec NetworkX
- Calcul automatique des propriétés (hauteur, degré, densité)
- Affichage détaillé des degrés par nœud

#### 🔗 **Section Graphes**
Création et visualisation de graphes personnalisés :

- **Graphes orientés/non-orientés**
- **Graphes pondérés/non-pondérés**
- Saisie via matrice d'adjacence interactive
- Visualisation avec NetworkX et Matplotlib
- Calcul des propriétés (degré maximal, degré moyen, densité)

---

### 🌳 TP2 - Arbre 2-3

Implémentation complète de l'arbre 2-3 (B-arbre d'ordre 3) avec toutes les opérations :

**Fonctionnalités :**
- ✅ **Insertion** : Insertion manuelle valeur par valeur ou import depuis fichier
- ✅ **Recherche** : Recherche de clés avec visualisation du nœud trouvé
- ✅ **Suppression** : Suppression avec gestion des fusions et emprunts
- ✅ **Vérification d'équilibre** : Contrôle automatique de l'équilibre de l'arbre
- ✅ **Visualisation** : Représentation graphique avec NetworkX
- ✅ **Statistiques** : Hauteur, nombre de nœuds, nombre de clés, temps d'exécution

**Interface :**
- Mode édition : Construction progressive de l'arbre
- Mode visualisation : Opérations (recherche, suppression) avec mise en évidence
- Tests rapides intégrés pour validation

---

### 🔄 TP3 - Arbre 2-3 & Tri Rapide (Quicksort)

Combinaison de deux algorithmes fondamentaux :

**Workflow :**
1. **Construction d'un arbre 2-3** à partir de valeurs saisies
2. **Parcours préfixe** : Extraction des valeurs dans un tableau non trié
3. **Tri rapide (Quicksort)** : Tri du tableau avec visualisation étape par étape
4. **Reconstruction** : Création d'un nouvel arbre 2-3 à partir du tableau trié

**Fonctionnalités Quicksort :**
- Visualisation interactive des étapes de partition
- Navigation étape par étape (Précédent/Suivant)
- Mise en évidence des éléments pivot, échangés et fixés
- Affichage des sous-tableaux à chaque récursion

---

### 🎯 TP4 - Algorithmes PCC et Coloration

#### 🔍 **Onglet 1 : Algorithme de Bellman-Ford**

Implémentation complète de l'algorithme de Bellman-Ford pour les plus courts chemins :

**Fonctionnalités :**
- ✅ **Support des poids négatifs** : Détection et gestion des cycles de poids négatif
- ✅ **Tableau d'itérations** : Visualisation détaillée de chaque itération
- ✅ **Reconstruction des chemins** : Affichage des plus courts chemins depuis la source
- ✅ **Graphe partiel G°** : Visualisation de l'arborescence des plus courts chemins
- ✅ **Matrices interactives** : Saisie via matrices d'adjacence et de pondération
- ✅ **Exemples pré-remplis** : Cas standards et cas avec poids négatifs

**Affichage :**
- Tableau des itérations avec marquage des valeurs modifiées (*)
- Tableau des résultats finaux (distance, chemin)
- Visualisation du graphe partiel avec distances et poids

#### 🎨 **Onglet 2 : Algorithme de Coloration (Matula)**

Implémentation de l'algorithme de Matula pour la coloration optimale de graphes :

**Fonctionnalités :**
- ✅ **Smallest-Last Ordering** : Classement par degré croissant
- ✅ **Coloration gloutonne** : Coloration optimale suivant l'ordre inversé
- ✅ **Visualisation étape par étape** : Tableaux des degrés et des colorations
- ✅ **Graphe coloré** : Visualisation finale avec palette de couleurs
- ✅ **Statistiques** : Nombre de couleurs utilisées, distribution, temps d'exécution

**Interface :**
- Saisie des sommets
- Matrice d'adjacence interactive (graphe non orienté)
- Affichage des trois étapes : degrés initiaux, ordering, coloration
- Graphes initial et final coloré

---

## 🛠️ Technologies utilisées

- **Python 3.10+**
- **Streamlit** : Framework web pour applications interactives
- **NetworkX** : Manipulation et visualisation de graphes
- **Matplotlib** : Visualisation graphique
- **NumPy** : Calculs numériques et matrices
- **Pandas** : Manipulation de données et DataFrames

---

## 📦 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone <votre-url-repo>
   cd TP_algo
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   ```

3. **Activer l'environnement virtuel**
   
   **Sur Windows :**
   ```bash
   venv\Scripts\activate
   ```
   
   **Sur Linux/Mac :**
   ```bash
   source venv/bin/activate
   ```

4. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Utilisation

### Lancer l'application localement

```bash
streamlit run interface.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse :
- **http://localhost:8501**

### Navigation dans l'application

1. **Page principale** : Affiche les informations du groupe et les boutons d'accès aux TPs
2. **TP1** : Arbres et Graphes
3. **TP2** : Arbre 2-3
4. **TP3** : Arbre 2-3 & Quicksort
5. **TP4** : Bellman-Ford & Coloration (Matula)

### Guide rapide par TP

#### TP1 - Arbres et Graphes
- Choisir la section (Arbre ou Graph)
- Sélectionner le type d'arbre souhaité
- Entrer les valeurs séparées par des virgules
- Cliquer sur "Construire"
- Pour les graphes : remplir la matrice d'adjacence

#### TP2 - Arbre 2-3
- Mode édition : Insérer des valeurs une par une ou charger depuis un fichier
- Vérifier l'équilibre de l'arbre
- Passer en mode visualisation pour les opérations (recherche, suppression)

#### TP3 -  Quicksort
- Entrer les valeurs initiales
- Cliquer sur "Lancer TP3"
- Visualiser l'arbre initial, le tableau non trié, le tableau trié, et le nouvel arbre
- Cliquer sur "Afficher les étapes du tri rapide" pour la visualisation détaillée

#### TP4 - Bellman-Ford
- **Onglet 1** : Saisir les sommets, remplir les matrices, choisir la source, lancer l'algorithme
- **Onglet 2** : Saisir les sommets, créer les arêtes via la matrice, lancer la coloration

---

## 📁 Structure du projet

```
TP_algo/
│
├── interface.py              # Page principale de l'application
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation du projet
│
└── pages/                     # Pages Streamlit (TPs)
    ├── tp1.py                # TP1 - Arbres et Graphes
    ├── tp_2.py               # TP2 - Arbre 2-3
    ├── tp_3.py               # TP3 - Arbre 2-3 & Quicksort
    └── tp_4.py               # TP4 - Bellman-Ford & Coloration
```

---

## 👥 Membres du groupe

**Groupe 4**

- **Bengrab Meriem**
- **Belhadj Aya**
- **Mehdid Malak**
- **Kalafat Fadoua**
- **Ziane Hiba**
- **Laidani Inès**

---

## 🌐 Déploiement

### Déploiement sur Streamlit Cloud

1. **Pousser le code sur GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Créer un compte Streamlit Cloud**
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Se connecter avec votre compte GitHub

3. **Déployer l'application**
   - Cliquer sur "New app"
   - Sélectionner votre dépôt et la branche
   - Définir le fichier principal : `interface.py`
   - Cliquer sur "Deploy"

4. **Lien public**
   - Votre application sera accessible via un lien public
   - **Notre application :** [https://tpalgo-groupe4.streamlit.app](https://tpalgo-groupe4.streamlit.app)



Pour un déploiement local avec Streamlit :

```bash
streamlit run interface.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📝 Notes

- Les visualisations utilisent **NetworkX** pour la manipulation des graphes
- Les graphiques sont générés avec **Matplotlib** (backend non interactif)
- Tous les algorithmes sont implémentés en Python pur (pas de bibliothèques externes pour les algorithmes)
- L'interface est entièrement responsive et optimisée pour l'enseignement

---

## 📄 Licence

Ce projet est développé dans le cadre d'un travail pratique universitaire.

---


## 🔗 Liens utiles

- **Application en ligne :** [https://tpalgo-groupe4.streamlit.app](https://tpalgo-groupe4.streamlit.app)
- **Documentation Streamlit :** [https://docs.streamlit.io](https://docs.streamlit.io)

---

**Développé avec ❤️ par le Groupe 4**

