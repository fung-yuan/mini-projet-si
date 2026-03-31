"""
Module utilitaire pour les visualisations et fonctions d'aide.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_churn_distribution(df, target='Churn'):
    """Affiche la distribution de la variable cible."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Countplot
    counts = df[target].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(counts.index.astype(str), counts.values, color=colors, edgecolor='white')
    axes[0].set_title('Distribution du Churn', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Nombre de clients')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['Non Churn', 'Churn'], 
                autopct='%1.1f%%', colors=colors, startangle=90,
                explode=(0, 0.05), shadow=True)
    axes[1].set_title('Proportion du Churn', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/churn_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_numerical_distributions(df, numerical_cols):
    """Affiche les distributions des variables numériques."""
    n = len(numerical_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue='Churn', kde=True, ax=axes[i], 
                     palette=['#2ecc71', '#e74c3c'], alpha=0.6)
        axes[i].set_title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/numerical_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_categorical_churn_rate(df, categorical_cols, max_cols=6):
    """Affiche le taux de churn par variable catégorielle."""
    cols_to_plot = categorical_cols[:max_cols]
    n = len(cols_to_plot)
    rows = (n + 2) // 3
    
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, col in enumerate(cols_to_plot):
        churn_rate = df.groupby(col)['Churn'].apply(
            lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
        )
        colors = sns.color_palette("husl", len(churn_rate))
        churn_rate.plot(kind='bar', ax=axes[i], color=colors, edgecolor='white')
        axes[i].set_title(f'Taux de Churn par {col}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Taux de Churn')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
    
    # Cacher les axes non utilisés
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('models/churn_rate_by_category.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df):
    """Affiche la matrice de corrélation."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Matrice de Corrélation', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(all_results):
    """Affiche les matrices de confusion de tous les modèles."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    
    for i, (name, results) in enumerate(all_results.items()):
        cm = results['Confusion Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Non Churn', 'Churn'],
                    yticklabels=['Non Churn', 'Churn'])
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Prédit')
        axes[i].set_ylabel('Réel')
    
    plt.suptitle('Matrices de Confusion', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(all_results, y_test):
    """Affiche les courbes ROC de tous les modèles."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(all_results))
    
    for i, (name, results) in enumerate(all_results.items()):
        if results['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ligne de base')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.set_title('Courbes ROC — Comparaison des Modèles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15):
    """Affiche l'importance des features (pour Random Forest / Decision Tree)."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("viridis", top_n)
        ax.barh(range(top_n), importances[indices][::-1], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices][::-1])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Features les Plus Importantes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("Ce modèle n'a pas d'attribut feature_importances_.")


def plot_model_comparison(comparison_df):
    """Affiche un graphique de comparaison des modèles."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.18
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].astype(float)
        bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i], edgecolor='white')
    
    ax.set_xlabel('Modèle', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des Modèles', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Modèle'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
