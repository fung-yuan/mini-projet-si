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

Nous avons entraîné et comparé 5 modèles d'apprentissage supervisé. Voici les résultats sur le jeu de test :

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| **Logistic Regression** | **0.8034** | **0.6678** | 0.5160 | **0.5822** | **0.8465** |
| Random Forest | 0.7857 | 0.6277 | 0.4733 | 0.5396 | 0.8228 |
| SVM | 0.8027 | 0.6690 | 0.5080 | 0.5775 | 0.7980 |
| k-NN | 0.7715 | 0.5743 | 0.5374 | 0.5552 | 0.7769 |
| Decision Tree | 0.7537 | 0.5354 | 0.5455 | 0.5404 | 0.7174 |

> Le modèle final retenu (sauvegardé pour l'interface web) est la **Régression Logistique**, qui offre le meilleur compromis (meilleur F1-Score et AUC).

### Analyse non supervisée
- **K-Means Clustering** : 3 segments de clients identifiés, permettant de cerner différents profils et taux de churn.
- **PCA** : Projection en 2D pour visualiser et interpréter la séparation des clusters et du churn réel.

---

## 🏗️ Structure du Projet

```text
mini-projet-si/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── data/
│   └── telco_churn.csv                # Dataset Telco Customer Churn
├── notebooks/
│   └── projet_churn.ipynb             # Notebook unique et complet (EDA → Prétraitement → Modèles → Clustering)
├── src/
│   ├── preprocessing.py               # Module de traitement et feature engineering
│   ├── models.py                      # Module d'entraînement, comparaison et cross-validation
│   └── utils.py                       # Utilitaires de visualisations (matrice confusion, ROC, etc.)
├── app/
│   └── streamlit_app.py               # Interface web interactive Streamlit finale
└── models/                            # Modèles sauvegardés (générés après l'entraînement)
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## 🚀 Comment Exécuter le Projet

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/fung-yuan/mini-projet-si.git
cd mini-projet-si

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution du Pipeline ML (Notebook)

Pour afficher l'exploration de données, l'entraînement des modèles et les graphiques :
```bash
jupyter notebook notebooks/projet_churn.ipynb
```
*(Vous pouvez exécuter toutes les cellules de ce notebook pour recréer les fichiers de modèles dans le dossier `models/`)*

### Lancer l'application web Streamlit

Notre application web interactive et stylisée permet d'estimer en temps réel le risque de churn d'un client !

```bash
streamlit run app/streamlit_app.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📦 Dépendances Principales

- `pandas` & `numpy` : Manipulation des données
- `matplotlib` & `seaborn` : Visualisations (matrices de corrélation, features importance, etc.)
- `scikit-learn` : Machine Learning (modèles, métriques, prétraitement, clustering)
- `streamlit` : Interface web interactive
- `joblib` : Sauvegarde et chargement des modèles sérialisés

---

## 📊 Workflow / Pipeline

```text
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
