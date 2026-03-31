# 🔮 Prédiction du Churn Client — Telco Customer Churn

## 📋 Description du Projet

**Mini-Projet Systèmes Intelligents** — Conception d'un système intelligent de prédiction du churn client de bout en bout.

### Problème traité
Le **churn client** (attrition) est un enjeu majeur pour les entreprises de télécommunications. Un client qui "churne" est un client qui résilie son abonnement. L'objectif est de **prédire quels clients risquent de partir** afin de permettre à l'entreprise de prendre des mesures préventives (offres promotionnelles, support client renforcé, etc.).

### Dataset utilisé
- **Source** : [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Taille** : 7 043 clients × 21 variables
- **Variable cible** : `Churn` (Yes / No)
- **Features** : informations démographiques (genre, senior, partenaire), services souscrits (internet, streaming, sécurité), données contractuelles (type de contrat, facturation, charges)

---

## 🤖 Modèles Testés et Résultats

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| Régression Logistique | ~0.80 | ~0.65 | ~0.55 | ~0.60 | ~0.84 |
| Random Forest | ~0.79 | ~0.64 | ~0.48 | ~0.55 | ~0.82 |
| SVM | ~0.80 | ~0.66 | ~0.52 | ~0.58 | ~0.83 |
| k-NN | ~0.77 | ~0.57 | ~0.52 | ~0.54 | ~0.77 |
| Arbre de Décision | ~0.73 | ~0.48 | ~0.52 | ~0.50 | ~0.66 |

> **Note** : Les valeurs exactes seront mises à jour après exécution des notebooks.

### Analyse non supervisée
- **K-Means Clustering** : 3 segments de clients identifiés avec des taux de churn différenciés
- **PCA** : Réduction de dimensionnalité pour la visualisation et l'interprétation

---

## 🏗️ Structure du Projet

```
mini-projet-si/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── data/
│   └── telco_churn.csv               # Dataset Telco Customer Churn
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA et visualisations
│   ├── 02_preprocessing.ipynb        # Prétraitement des données
│   ├── 03_modeling.ipynb             # Entraînement et comparaison des modèles
│   └── 04_clustering.ipynb           # Clustering (K-Means) et PCA
├── src/
│   ├── preprocessing.py              # Module de prétraitement
│   ├── models.py                     # Module de modélisation
│   └── utils.py                      # Utilitaires et visualisations
├── app/
│   └── streamlit_app.py              # Interface Streamlit
├── models/                           # Modèles sauvegardés (.pkl)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── rapport/
    └── rapport.pdf                   # Rapport PDF (7-10 pages)
```

---

## 🚀 Comment Exécuter le Projet

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USER/mini-projet-si.git
cd mini-projet-si

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution des Notebooks

```bash
# Option 1 : Jupyter Notebook
jupyter notebook notebooks/

# Option 2 : Exécution séquentielle en scripts
python notebooks/01_exploration.py
python notebooks/02_preprocessing.py
python notebooks/03_modeling.py
python notebooks/04_clustering.py
```

### Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à l'adresse `http://localhost:8501`.

---

## 📦 Dépendances

| Package | Version | Usage |
|---------|---------|-------|
| pandas | ≥1.5 | Manipulation des données |
| numpy | ≥1.23 | Calculs numériques |
| matplotlib | ≥3.6 | Visualisations |
| seaborn | ≥0.12 | Visualisations statistiques |
| scikit-learn | ≥1.2 | ML (modèles, métriques, prétraitement) |
| streamlit | ≥1.28 | Interface web interactive |
| joblib | ≥1.2 | Sérialisation des modèles |

---

## 📊 Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Exploration │───▶│ Prétraitement│───▶│ Modélisation │───▶│  Interface   │
│    (EDA)     │    │  (Cleaning)  │    │ (5 modèles)  │    │ (Streamlit)  │
└──────────────┘    └──────────────┘    └──────┬───────┘    └──────────────┘
                                               │
                                        ┌──────▼───────┐
                                        │  Clustering  │
                                        │ (K-Means+PCA)│
                                        └──────────────┘
```

---

## 👥 Auteurs

- **Étudiant 1** — [Nom]
- **Étudiant 2** — [Nom]

Mini-Projet Systèmes Intelligents — 2025/2026
