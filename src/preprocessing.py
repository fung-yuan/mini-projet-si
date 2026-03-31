"""
Module de prétraitement des données Telco Customer Churn.
Contient les fonctions de nettoyage, encodage et normalisation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Charge le dataset depuis un fichier CSV."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    Nettoie les données :
    - Convertit TotalCharges en numérique
    - Gère les valeurs manquantes
    - Supprime la colonne customerID
    """
    df = df.copy()
    
    # Convertir TotalCharges en numérique (certaines valeurs sont des espaces vides)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remplir les valeurs manquantes de TotalCharges avec la médiane
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Supprimer customerID (pas utile pour la modélisation)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    return df


def encode_features(df):
    """
    Encode les variables catégorielles :
    - LabelEncoding pour les variables binaires
    - One-Hot Encoding pour les variables multi-catégories
    """
    df = df.copy()
    
    # Variables binaires (2 catégories) → Label Encoding
    binary_cols = []
    multi_cols = []
    
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)
    
    # Label Encoding pour les variables binaires
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-Hot Encoding pour les variables multi-catégories
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    return df


def feature_engineering(df):
    """
    Crée de nouvelles features :
    - Ratio charges mensuelles / ancienneté
    - Indicateur de nouveau client
    """
    df = df.copy()
    
    # Ratio charges / ancienneté (éviter division par 0)
    df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # Client nouveau (tenure <= 6 mois)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    
    return df


def scale_features(X_train, X_test):
    """Standardise les features numériques."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(filepath, test_size=0.2, random_state=42):
    """
    Pipeline complet de préparation des données.
    Retourne X_train, X_test, y_train, y_test, scaler et le dataframe nettoyé.
    """
    # Chargement
    df = load_data(filepath)
    
    # Nettoyage
    df_clean = clean_data(df)
    
    # Feature engineering
    df_clean = feature_engineering(df_clean)
    
    # Encodage
    df_encoded = encode_features(df_clean)
    
    # Séparation features / target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Split train/test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalisation
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_clean, X.columns.tolist()
