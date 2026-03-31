"""
Module de modélisation pour la prédiction du churn client.
Contient les fonctions d'entraînement et d'évaluation des modèles.
"""

import numpy as np
import pandas as pd
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score


def get_models():
    """Retourne un dictionnaire des modèles à entraîner."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    return models


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Entraîne un modèle et calcule toutes les métriques d'évaluation.
    Retourne un dictionnaire de résultats.
    """
    # Entraînement avec mesure du temps
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métriques
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'Training Time (s)': round(train_time, 4),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return results


def compare_models(X_train, X_test, y_train, y_test):
    """
    Entraîne et compare tous les modèles.
    Retourne un DataFrame de comparaison et un dictionnaire de résultats détaillés.
    """
    models = get_models()
    all_results = {}
    comparison_data = []
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entraînement : {name}")
        print(f"{'='*50}")
        
        results = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        all_results[name] = {**results, 'model': model}
        
        comparison_data.append({
            'Modèle': name,
            'Accuracy': f"{results['Accuracy']:.4f}",
            'Precision': f"{results['Precision']:.4f}",
            'Recall': f"{results['Recall']:.4f}",
            'F1-Score': f"{results['F1-Score']:.4f}",
            'AUC': f"{results['AUC']:.4f}" if results['AUC'] else 'N/A',
            'Temps (s)': results['Training Time (s)']
        })
        
        print(f"  Accuracy:  {results['Accuracy']:.4f}")
        print(f"  F1-Score:  {results['F1-Score']:.4f}")
        print(f"  AUC:       {results['AUC']:.4f}" if results['AUC'] else "  AUC: N/A")
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df, all_results


def cross_validate_models(X_train, y_train, cv=5):
    """Effectue une validation croisée sur tous les modèles."""
    models = get_models()
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        cv_results[name] = {
            'Mean F1': scores.mean(),
            'Std F1': scores.std(),
            'Scores': scores
        }
        print(f"{name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    return cv_results


def save_best_model(all_results, filepath='models/best_model.pkl'):
    """Sauvegarde le meilleur modèle basé sur le F1-Score."""
    best_name = max(all_results, key=lambda x: all_results[x]['F1-Score'])
    best_model = all_results[best_name]['model']
    
    joblib.dump(best_model, filepath)
    print(f"\nMeilleur modèle : {best_name} (F1={all_results[best_name]['F1-Score']:.4f})")
    print(f"Sauvegardé dans : {filepath}")
    
    return best_name, best_model
